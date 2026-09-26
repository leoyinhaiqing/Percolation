"""主时间步循环与指标记录（Track A 仿真引擎）。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import theory
from .config import Config
from .cascade import stage1_random_collapse, stage2_redistribute_cascade
from .hypergraph import (
    build_incidence,
    generate_powerlaw_hypergraph,
    giant_component_fraction,
    hyperdegree_moments,
    sample_vulnerability,
)


@dataclass
class Result:
    """单次仿真的输出（初始指标 + 时间序列 + 理论对照）。"""

    cfg: Config
    # 初始/静态量
    E0: int
    k0: float
    k2: float
    b_mean: float
    S0: float
    B0: float
    p: float
    rho_c: float
    C_threshold: float
    C_crit: float
    regime: str
    Tc_th: float
    # 时间序列
    ts: np.ndarray
    S: np.ndarray
    Sh: np.ndarray
    L: np.ndarray
    ncol: np.ndarray
    # 结果
    Tc_sim: Optional[int]            # 首次 S<eps 的步
    Tc_sim2: Optional[int]           # 首次 S<eps2 的步
    T_density: Optional[int]         # 首次 S_h<rho_c 的步
    lost_load: float = 0.0
    edge_nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.int64))


def simulate(cfg: Config,
             edge_nodes: Optional[np.ndarray] = None,
             b: Optional[np.ndarray] = None,
             C: Optional[float] = None) -> Result:
    """主循环：Stage 1 随机坍塌 → Stage 2 级联 → 记录巨分量/密度/负载。

    - 生成（或复用给定）幂律超图，报告初始指标与理论量；
    - 逐时间步推进；**无坍塌步跳过重算**（S/S_h/L 不变）；
    - 记录 T_c_sim（S<eps）、T_c_sim2（S<eps2）、T_density（S_h<rho_c）。
    - edge_nodes/b/C 可复用：Track B 在固定 H 上做扰动时传入。
    """
    rng = np.random.default_rng(cfg.seed)

    # --- 生成与初始化 ---
    if edge_nodes is None:
        edge_nodes, _ = generate_powerlaw_hypergraph(
            cfg.N, cfg.m, gamma=cfg.gamma, k_min=cfg.k_min, k_max=cfg.k_max, rng=rng)
    E0 = len(edge_nodes)
    node_edges = build_incidence(edge_nodes, cfg.N)
    k0, k2 = hyperdegree_moments(node_edges)

    if b is None:
        b = sample_vulnerability(cfg.N, cfg.b_dist, rng, cfg.b_param)
    b_mean = float(b.mean())

    coef = math.factorial(cfg.m) / cfg.N ** cfg.m
    edge_p = coef * b[edge_nodes].sum(axis=1)  # P(e) = m!/N^m * Σ_{i∈e} b_i（L66）

    load = np.full(E0, cfg.L0, dtype=float)
    alive = np.ones(E0, dtype=bool)
    Cv = cfg.C if C is None else float(C)
    if Cv is None:
        raise ValueError("Config.C 必须显式给出（实验取两个 C 跨 C* 阈值）")

    # --- 理论量（main.tex 0803 修正版）---
    p = theory.collapse_prob_p(cfg.N, cfg.m, b_mean)
    rhoc = theory.rho_c(k0, k2, cfg.m)
    B0 = theory.branching_B(k0, cfg.m, S_h=1.0)
    C_thr = theory.C_threshold(k0, cfg.m)
    Cc = theory.C_crit(k0, k2, cfg.m)
    rgm = theory.regime(k0, cfg.m, Cv)
    # p=0（b_mean=0）永不坍塌：Tc_th 无定义（记 inf），t_max 用显式值
    if p == 0.0:
        Tc_th = float("inf")
    else:
        Tc_th = theory.collapse_time_Tc(cfg.N, cfg.m, k0, k2, b_mean, Cv,
                                        cfg.b_dist, cfg.b_param)

    # t_max 安全上限：理论 T_c 的 5 倍 + 余量；p=0 时用显式值
    if cfg.t_max is not None:
        t_max = cfg.t_max
    elif p == 0.0:
        t_max = 100
    else:
        t_max = int(5.0 * Tc_th) + 100
    rec = max(1, cfg.record_every)

    # --- 记录器 ---
    ts: List[int] = []
    S: List[float] = []
    Sh: List[float] = []
    L: List[float] = []
    ncol: List[int] = []

    def record(t: int, nc: int) -> float:
        s = giant_component_fraction(edge_nodes, alive, cfg.N)
        n_alive = int(alive.sum())
        ts.append(t)
        S.append(s)
        Sh.append(n_alive / E0)
        L.append(float(load[alive].mean()) if n_alive > 0 else 0.0)
        ncol.append(nc)
        return s

    s0 = record(0, 0)

    Tc_sim: Optional[int] = None
    Tc_sim2: Optional[int] = None
    T_density: Optional[int] = None
    lost_total = 0.0

    # --- 事件驱动主循环（Gillespie 风格）---
    # 空步（无坍塌）对系统无任何副作用（S/S_h/L 不变、无级联触发），故可直接采样
    # "下一次至少一条边坍塌"的等待时间 dt ~ Geom(p_total)，跳到事件步再处理级联。
    # 这是逐步仿真的精确加速（p_e 极小，同一步多条边同时坍塌概率 ~ p_total^2 可忽略）。
    # 每事件步都记录（事件间状态不变，无需按 record_every 采样）。
    t = 0
    while t < t_max and alive.any():
        w = edge_p[alive]
        p_total = float(w.sum())
        if p_total == 0.0:
            break  # b=0：永不坍塌
        dt = int(rng.geometric(p_total))  # 等待步数（>=1）
        t += dt
        if t > t_max:
            break
        alive_idx = np.nonzero(alive)[0]
        e = int(rng.choice(alive_idx, p=w / p_total))  # 按 P(e) 加权选坍塌边
        collapsed, lost = stage2_redistribute_cascade(
            edge_nodes, node_edges, load, alive, Cv, np.array([e]))
        lost_total += lost

        s = record(t, len(collapsed))
        sh = Sh[-1]
        if Tc_sim is None and s < cfg.eps:
            Tc_sim = t
        if Tc_sim2 is None and s < cfg.eps2:
            Tc_sim2 = t
        if T_density is None and sh < rhoc:
            T_density = t
        # 已拿到所需阈值且巨分量已基本消失 → 提前停止以省时
        if not alive.any() or (Tc_sim is not None and T_density is not None
                               and s < cfg.eps2):
            break

    return Result(
        cfg=cfg,
        E0=E0,
        k0=k0,
        k2=k2,
        b_mean=b_mean,
        S0=s0,
        B0=B0,
        p=p,
        rho_c=rhoc,
        C_threshold=C_thr,
        C_crit=Cc,
        regime=rgm,
        Tc_th=Tc_th,
        ts=np.array(ts),
        S=np.array(S),
        Sh=np.array(Sh),
        L=np.array(L),
        ncol=np.array(ncol),
        Tc_sim=Tc_sim,
        Tc_sim2=Tc_sim2,
        T_density=T_density,
        lost_load=lost_total,
        edge_nodes=edge_nodes,
    )
