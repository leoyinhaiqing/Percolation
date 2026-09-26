"""事件驱动 vs 逐步 Bernoulli 仿真对照（验证稀有事件近似误差）。

逐步版：每个时间步对每条存活边独立 Bernoulli(p_e)（main.tex 的精确离散模型），
不跳空步；仅在有坍塌的步重算 S（空步状态不变）。
事件驱动版：src.dynamics.simulate（Geom(Σp_e) 跳步 + 每事件选一条初始坍塌边）。

输出：每 (N, k_min, C) 下两种实现的 T_c^sim / T_density 均值与相对差异。
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import theory  # noqa: E402
from src.config import Config  # noqa: E402
from src.dynamics import simulate as event_simulate  # noqa: E402
from src.hypergraph import (  # noqa: E402
    build_incidence, generate_powerlaw_hypergraph, giant_component_fraction,
    hyperdegree_moments, sample_vulnerability,
)
from src.cascade import stage1_random_collapse, stage2_redistribute_cascade  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def stepwise_simulate(cfg: Config, C: float):
    """逐步 Bernoulli 主循环（不跳步）。返回 (T_c_sim, T_density)。"""
    rng = np.random.default_rng(cfg.seed)
    edge_nodes, _ = generate_powerlaw_hypergraph(cfg.N, cfg.m, gamma=cfg.gamma,
                                                 k_min=cfg.k_min, rng=rng)
    node_edges = build_incidence(edge_nodes, cfg.N)
    k0, k2 = hyperdegree_moments(node_edges)
    b = sample_vulnerability(cfg.N, "uniform", rng)
    E0 = len(edge_nodes)
    edge_p = math.factorial(cfg.m) / cfg.N ** cfg.m * b[edge_nodes].sum(axis=1)
    load = np.full(E0, 1.0)
    alive = np.ones(E0, dtype=bool)
    rhoc = theory.rho_c(k0, k2, cfg.m)
    t_max = int(5.0 * theory.collapse_time_Tc(cfg.N, cfg.m, k0, k2, b.mean(), C)) + 100

    Tc_sim = T_density = None
    for t in range(1, t_max + 1):
        D = stage1_random_collapse(edge_p, alive, rng)
        if len(D):
            stage2_redistribute_cascade(edge_nodes, node_edges, load, alive, C, D)
            # 仅在有坍塌的步重算 S（空步状态不变，首次跌破必在坍塌步）
            if Tc_sim is None:
                s = giant_component_fraction(edge_nodes, alive, cfg.N)
                if s < cfg.eps:
                    Tc_sim = t
            if T_density is None and alive.sum() / E0 < rhoc:
                T_density = t
        if not alive.any() or (Tc_sim is not None and T_density is not None):
            break
    return Tc_sim, T_density


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[200, 500])
    ap.add_argument("--k_mins", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()

    rows = []
    for N in args.Ns:
        for kmin in args.k_mins:
            probe = event_simulate(Config(N=N, m=2, gamma=2.5, k_min=kmin,
                                          C=1.0, seed=1))
            Cstar = theory.C_threshold(probe.k0, 2)
            for C in (round(0.75 * Cstar, 4), round(1.5 * Cstar, 4)):
                tc_e, tc_s, td_e, td_s = [], [], [], []
                for sd in range(args.seeds):
                    cfg = Config(N=N, m=2, gamma=2.5, k_min=kmin, C=C, seed=sd)
                    r = event_simulate(cfg)
                    tc_e.append(r.Tc_sim if r.Tc_sim else np.nan)
                    td_e.append(r.T_density if r.T_density else np.nan)
                    ts, tds = stepwise_simulate(cfg, C)
                    tc_s.append(ts if ts else np.nan)
                    td_s.append(tds if tds else np.nan)
                tc_e, tc_s = np.nanmean(tc_e), np.nanmean(tc_s)
                td_e, td_s = np.nanmean(td_e), np.nanmean(td_s)
                row = dict(N=N, k_min=kmin, C=C,
                           Tc_event=tc_e, Tc_stepwise=tc_s,
                           Tc_rel_diff=abs(tc_e - tc_s) / tc_s * 100 if tc_s else np.nan,
                           Td_event=td_e, Td_stepwise=td_s,
                           Td_rel_diff=abs(td_e - td_s) / td_s * 100 if td_s else np.nan)
                rows.append(row)
                print(f"N={N} kmin={kmin} C={C:.3f} | Tc: event={tc_e:.0f} "
                      f"stepwise={tc_s:.0f} Δ={row['Tc_rel_diff']:.2f}% | "
                      f"Td: event={td_e:.0f} stepwise={td_s:.0f} "
                      f"Δ={row['Td_rel_diff']:.2f}%")

    # 存 CSV
    import csv
    path = os.path.join(RESULTS, "event_vs_stepwise.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
