"""dynamics.py 主循环测试（健全性 + 口径）。"""
import numpy as np
import pytest

from src.config import Config
from src import theory
from src.dynamics import simulate
from src.hypergraph import generate_powerlaw_hypergraph


def _cfg(**kw):
    base = dict(N=200, m=2, C=2.0, seed=0)
    base.update(kw)
    return Config(**base)


def test_b_zero_never_collapses():
    # b≡0 → edge_p=0 → 永不坍塌，S(t) 恒定 = S(0)
    cfg = _cfg(C=2.0, t_max=50)
    b = np.zeros(cfg.N)
    r = simulate(cfg, b=b)
    assert r.Tc_sim is None
    assert r.T_density is None
    assert np.allclose(r.S, r.S[0])
    assert np.allclose(r.Sh, 1.0)


def test_huge_capacity_degenerates_to_random_annihilation():
    # C → ∞：无级联，只剩 Stage-1 随机湮灭 → S_h(t) ≈ (1-p)^t ≈ e^{-pt}（L213）
    cfg = _cfg(C=1e6, t_max=3000)
    r = simulate(cfg)
    p = theory.collapse_prob_p(cfg.N, cfg.m, r.b_mean)
    # 事件驱动下 ts 是事件时间；S_h(t) = 存活边/E0 ≈ (1-p)^t（期望）
    expected = (1.0 - p) ** r.ts
    # 随机游走涨落 ~sqrt(Sh(1-Sh)/E0) ≈ 1.6%/点；逐点 3-4σ 偏差正常，
    # 用统计式断言：中位相对偏差小、最大偏差 < 5σ
    rel = np.abs(r.Sh - expected) / expected
    assert np.median(rel) < 0.02
    assert rel.max() < 0.08


def test_Tc_sim_and_Tdensity_reasonable():
    # 真实配置：两个 C（三阶段/两阶段）各跑一次，应得到非空、量级合理的 T_c
    for C in (2.0, 8.0):
        cfg = _cfg(C=C, seed=1)
        r = simulate(cfg)
        assert r.Tc_sim is not None and r.Tc_sim > 0
        assert r.T_density is not None and r.T_density > 0
        assert r.rho_c > 0
        assert r.Tc_th > 0
        assert r.regime in ("case_I", "case_II")
        # S(t) 单调不增（有记录序列）
        assert np.all(np.diff(r.S) <= 1e-12)
        # 两个口径都应在模拟范围内拿到；相对顺序无保证
        # （小 N 近临界时 S 可在 S_h 跌破 ρ_c 前就 < eps）
        assert r.Tc_sim <= r.ts[-1] and r.T_density <= r.ts[-1]


def test_reuse_edge_nodes_and_b():
    # 复用给定 H 与 b：不重新生成，结果可复现；同图同参数两次运行一致
    rng = np.random.default_rng(7)
    edge_nodes, _ = generate_powerlaw_hypergraph(200, 2, gamma=2.5, k_min=1, rng=rng)
    b = np.linspace(0.1, 0.9, 200)
    cfg = _cfg(C=2.0, seed=99)
    r1 = simulate(cfg, edge_nodes=edge_nodes, b=b)
    r2 = simulate(cfg, edge_nodes=edge_nodes, b=b)
    assert r1.E0 == len(edge_nodes)
    assert np.array_equal(r1.ts, r2.ts)
    assert np.allclose(r1.S, r2.S)
    assert np.allclose(r1.Sh, r2.Sh)


def test_missing_C_raises():
    # 新框架要求显式 C（实验取两个 C 跨阈值）
    cfg = Config(N=200, m=2, C=None, t_max=10)
    with pytest.raises(ValueError):
        simulate(cfg)
