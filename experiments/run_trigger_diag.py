"""触发事件诊断：波动驱动机制的统计证据（多 seed × 两 regime）。

对每 (N, k_min, C) 跑多 seed 事件驱动仿真，定位**首次大级联触发事件步**
（S_h 首次跌破 S_h^1 的事件），记录：
  - 触发前负载分位数（p50/p90/p99/max）与 C 的比值
  - 坍塌边自身负载
  - 该事件步级联移除边数（一次级联的规模）
  - S_h 跳幅（触发前 → 触发后）
输出 CSV 到 results/trigger_diag_{tag}.csv。

用法:
    .venv/Scripts/python.exe experiments/run_trigger_diag.py --N 1000 --k_min 3 --seeds 5
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import theory  # noqa: E402
from src.config import Config  # noqa: E402
from src.hypergraph import (  # noqa: E402
    build_incidence, generate_powerlaw_hypergraph, hyperdegree_moments,
    sample_vulnerability,
)
from src.cascade import stage2_redistribute_cascade  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def trigger_diag(N: int, m: int, k_min: int, C: float, seed: int) -> dict:
    """单 seed：事件驱动跑到首次大级联触发步，返回诊断量（未触发则全 None）。"""
    rng = np.random.default_rng(seed)
    edge_nodes, _ = generate_powerlaw_hypergraph(N, m, gamma=2.5, k_min=k_min,
                                                 rng=rng)
    node_edges = build_incidence(edge_nodes, N)
    k0, k2 = hyperdegree_moments(node_edges)
    b = sample_vulnerability(N, "uniform", rng)
    E0 = len(edge_nodes)
    edge_p = math.factorial(m) / N ** m * b[edge_nodes].sum(axis=1)
    load = np.full(E0, 1.0)
    alive = np.ones(E0, dtype=bool)
    sh1 = theory.S_h_1star(k0, m, C)
    rc = theory.rho_c(k0, k2, m)
    t_max = int(5.0 * theory.collapse_time_Tc(N, m, k0, k2, b.mean(), C)) + 100

    t = 0
    Sh_prev = 1.0
    while alive.any() and t < t_max:
        w = edge_p[alive]
        ptot = float(w.sum())
        if ptot == 0:
            break
        t += int(rng.geometric(ptot))
        ai = np.nonzero(alive)[0]
        e = int(rng.choice(ai, p=w / ptot))
        e_load = float(load[e])
        l_before = load[alive]
        q = np.percentile(l_before, [50, 90, 99, 100])
        collapsed, lost = stage2_redistribute_cascade(
            edge_nodes, node_edges, load, alive, C, np.array([e]))
        Sh = alive.sum() / E0
        if Sh <= sh1:  # 首次触发
            return dict(seed=seed, N=N, k_min=k_min, C=C,
                        regime=theory.regime(k0, m, C),
                        t_trigger=t, Sh_before=Sh_prev, Sh_after=Sh,
                        n_cascade=len(collapsed),
                        e_load=e_load, q50=q[0], q90=q[1], q99=q[2], qmax=q[3])
        Sh_prev = Sh
    return dict(seed=seed, N=N, k_min=k_min, C=C, regime="?", t_trigger=None,
                Sh_before=None, Sh_after=None, n_cascade=None,
                e_load=None, q50=None, q90=None, q99=None, qmax=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    probe = generate_powerlaw_hypergraph(args.N, args.m, gamma=2.5,
                                         k_min=args.k_min,
                                         rng=np.random.default_rng(1))[0]
    node_edges = build_incidence(probe, args.N)
    k0, _ = hyperdegree_moments(node_edges)
    Cstar = theory.C_threshold(k0, args.m)
    cs = [round(0.75 * Cstar, 4), round(1.5 * Cstar, 4)]

    rows = []
    for C in cs:
        print(f"--- C={C} ({theory.regime(k0, args.m, C)}) ---")
        for sd in range(args.seed0, args.seed0 + args.seeds):
            row = trigger_diag(args.N, args.m, args.k_min, C, sd)
            rows.append(row)
            if row["t_trigger"] is not None:
                print(f"  seed={sd}: t={row['t_trigger']} 级联移除 {row['n_cascade']} 边 "
                      f"(E0~{int(args.N*k0/3)}) | S_h {row['Sh_before']:.3f}->{row['Sh_after']:.3f} | "
                      f"负载 p50/p90/p99/max = {row['q50']/C:.2f}/{row['q90']/C:.2f}/"
                      f"{row['q99']/C:.2f}/{row['qmax']/C:.2f} C")
            else:
                print(f"  seed={sd}: 未触发")

    path = os.path.join(RESULTS, f"trigger_diag_N{args.N}_m{args.m}_kmin{args.k_min}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
