"""fastsim（淬火时钟引擎）的正确性测试。

重点是与既有、已验证的实现逐点等价：关联表、级联规则、巨分量口径。
统计层面的等价（三段时刻分布）由 experiments/validate_fastsim.py 负责。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import theory
from src.cascade import stage2_redistribute_cascade
from src.fastsim import (
    build_incidence_lists,
    cascade_sparse,
    edge_rates,
    giant_component_fraction_fast,
    quenched_clocks,
    sample_quenched_uniforms,
    simulate_quenched,
)
from src.hypergraph import (
    build_incidence,
    generate_powerlaw_hypergraph,
    giant_component_fraction,
    hyperdegree_moments,
    sample_vulnerability,
)


def _toy(N=300, m=2, k_min=3, seed=0):
    rng = np.random.default_rng(seed)
    edge_nodes, _ = generate_powerlaw_hypergraph(N, m, gamma=2.5, k_min=k_min, rng=rng)
    b = sample_vulnerability(N, "uniform", rng)
    return edge_nodes, b, rng


def test_incidence_lists_match_sets():
    edge_nodes, _, _ = _toy()
    N = 300
    ref = build_incidence(edge_nodes, N)
    got = build_incidence_lists(edge_nodes, N)
    assert len(got) == N
    for v in range(N):
        assert got[v] == sorted(ref[v])


def test_edge_rates_match_formula():
    edge_nodes, b, _ = _toy()
    N, m = 300, 2
    got = edge_rates(edge_nodes, b, N, m)
    ref = math.factorial(m) / N ** m * b[edge_nodes].sum(axis=1)
    assert np.allclose(got, ref)


def test_quenched_clocks_geometric_mean_and_support():
    rng = np.random.default_rng(3)
    p = 1e-3
    u = sample_quenched_uniforms(200_000, rng)
    tau = quenched_clocks(np.full(200_000, p), u)
    assert tau.min() >= 1
    assert abs(tau.mean() / (1.0 / p) - 1.0) < 0.02   # E[Geom] = 1/p


def test_quenched_clocks_monotone_in_rate():
    """CRN 的关键性质：p_e 增大 -> tau_e 不增（同一 u 下）。"""
    rng = np.random.default_rng(4)
    u = sample_quenched_uniforms(5000, rng)
    lo = quenched_clocks(np.full(5000, 1e-4), u)
    hi = quenched_clocks(np.full(5000, 3e-4), u)
    assert np.all(hi <= lo)
    assert (hi < lo).mean() > 0.9


@pytest.mark.parametrize("C", [1.5, 2.0, 3.0, 1e9])
def test_cascade_sparse_matches_reference(C):
    """稀疏级联与 cascade.stage2_redistribute_cascade 在同一状态下结果一致。"""
    N = 300
    edge_nodes, _, rng = _toy(N=N)
    E = len(edge_nodes)
    node_edges_sets = build_incidence(edge_nodes, N)
    node_edges_lists = build_incidence_lists(edge_nodes, N)
    en_list = [tuple(int(v) for v in row) for row in edge_nodes]

    for trial in range(5):
        rg = np.random.default_rng(100 + trial)
        alive0 = rg.random(E) > 0.3
        load0 = 1.0 + 0.4 * rg.random(E)
        seeds = np.array(sorted(rg.choice(np.nonzero(alive0)[0], size=3,
                                          replace=False).tolist()))

        load_a, alive_a = load0.copy(), alive0.copy()
        col_a, lost_a = stage2_redistribute_cascade(
            edge_nodes, node_edges_sets, load_a, alive_a, C, seeds)

        load_b, alive_b = load0.tolist(), alive0.tolist()
        col_b, lost_b = cascade_sparse(en_list, node_edges_lists, load_b, alive_b,
                                       C, seeds.tolist())

        assert sorted(col_a) == sorted(col_b)
        assert alive_a.tolist() == alive_b
        assert np.allclose(load_a, np.array(load_b))
        assert lost_a == pytest.approx(lost_b)


def test_giant_component_fast_matches_reference():
    N = 300
    edge_nodes, _, _ = _toy(N=N)
    E = len(edge_nodes)
    for seed in range(4):
        rg = np.random.default_rng(seed)
        alive = rg.random(E) > 0.4
        ref = giant_component_fraction(edge_nodes, alive, N)
        got = giant_component_fraction_fast(edge_nodes, alive, N)
        assert got == pytest.approx(ref)


def test_giant_component_node_removal_shrinks():
    N = 300
    edge_nodes, _, _ = _toy(N=N)
    full = giant_component_fraction_fast(edge_nodes, None, N)
    keep = np.ones(N, dtype=bool)
    keep[:30] = False
    cut = giant_component_fraction_fast(edge_nodes, None, N, node_mask=keep)
    assert 0.0 < cut <= full


def test_simulate_quenched_threshold_ordering():
    """S_h 单调下降 => t_sh1 <= t_shE <= t_rhoc。"""
    N, m = 500, 2
    edge_nodes, b, rng = _toy(N=N)
    node_edges = build_incidence_lists(edge_nodes, N)
    k0, k2 = hyperdegree_moments(build_incidence(edge_nodes, N))
    C = 1.5 * theory.C_threshold(k0, m)
    edge_p = edge_rates(edge_nodes, b, N, m)
    u = sample_quenched_uniforms(len(edge_nodes), rng)
    r = simulate_quenched(edge_nodes, node_edges, edge_p, C,
                          theory.S_h_1star(k0, m, C), theory.S_h_E(k0),
                          theory.rho_c(k0, k2, m), u, collect_extras=True)
    assert r["t_sh1_sim"] is not None and r["t_rhoc_sim"] is not None
    assert r["t_sh1_sim"] <= r["t_shE_sim"] <= r["t_rhoc_sim"]
    # 初始段应当是"纯随机"：几乎没有额外坍塌
    assert np.mean(r["extras"]["init"]) < 0.05


def test_simulate_quenched_no_cascade_when_capacity_infinite():
    """C -> inf 退化：无级联，事件数 = 坍塌边数，全部落在 extras=0。"""
    N, m = 400, 2
    edge_nodes, b, rng = _toy(N=N)
    node_edges = build_incidence_lists(edge_nodes, N)
    k0, k2 = hyperdegree_moments(build_incidence(edge_nodes, N))
    edge_p = edge_rates(edge_nodes, b, N, m)
    u = sample_quenched_uniforms(len(edge_nodes), rng)
    r = simulate_quenched(edge_nodes, node_edges, edge_p, 1e12,
                          theory.S_h_1star(k0, m, 1e12), theory.S_h_E(k0),
                          theory.rho_c(k0, k2, m), u, collect_extras=True)
    all_extras = r["extras"]["init"] + r["extras"]["casc"] + r["extras"]["tail"]
    assert max(all_extras) == 0


def test_crn_perturbation_only_shortens_collapse():
    """把某节点 b_i 提到 1（其余不变、u 固定）不会推迟 T_c —— CRN 单调性。"""
    N, m = 400, 2
    edge_nodes, b, rng = _toy(N=N)
    node_edges = build_incidence_lists(edge_nodes, N)
    k0, k2 = hyperdegree_moments(build_incidence(edge_nodes, N))
    C = 1.5 * theory.C_threshold(k0, m)
    sh1, shE, rc = (theory.S_h_1star(k0, m, C), theory.S_h_E(k0),
                    theory.rho_c(k0, k2, m))
    u = sample_quenched_uniforms(len(edge_nodes), rng)

    base = simulate_quenched(edge_nodes, node_edges,
                             edge_rates(edge_nodes, b, N, m), C, sh1, shE, rc, u)
    hub = int(np.argmax([len(s) for s in node_edges]))
    b2 = b.copy()
    b2[hub] = 1.0
    pert = simulate_quenched(edge_nodes, node_edges,
                             edge_rates(edge_nodes, b2, N, m), C, sh1, shE, rc, u)
    assert pert["t_rhoc_sim"] <= base["t_rhoc_sim"]
