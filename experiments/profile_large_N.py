"""大 N 的耗时归属：超图生成 vs 关联表 vs 仿真主循环。

N=1e5 的全套（15 seeds × 2 C）实测 490m47s，远高于按事件数外推的预期，
本脚本把单个 (生成 / 关联表 / 一次仿真) 的墙钟时间分开量出来，确定瓶颈在哪
（决定"再往上推 N"该优化什么）。

用法：
  .venv/Scripts/python.exe experiments/profile_large_N.py --Ns 20000 50000 100000
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import theory  # noqa: E402
from src.fastsim import (  # noqa: E402
    build_incidence_lists, edge_rates, sample_quenched_uniforms, simulate_quenched,
)
from src.hypergraph import (  # noqa: E402
    build_incidence, generate_powerlaw_hypergraph, hyperdegree_moments,
    sample_vulnerability,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[20000, 50000, 100000])
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"{'N':>8} {'E0':>9} {'gen(s)':>9} {'incid(s)':>9} {'moments(s)':>11} "
          f"{'sim_I(s)':>9} {'sim_II(s)':>10} {'total/seed(s)':>14}")
    for N in args.Ns:
        rng = np.random.default_rng(args.seed)
        t0 = time.perf_counter()
        edge_nodes, _ = generate_powerlaw_hypergraph(N, args.m, gamma=2.5,
                                                     k_min=args.k_min, rng=rng)
        t_gen = time.perf_counter() - t0

        t0 = time.perf_counter()
        node_edges = build_incidence_lists(edge_nodes, N)
        t_inc = time.perf_counter() - t0

        t0 = time.perf_counter()
        k0, k2 = hyperdegree_moments(build_incidence(edge_nodes, N))
        t_mom = time.perf_counter() - t0

        b = sample_vulnerability(N, "uniform", rng)
        edge_p = edge_rates(edge_nodes, b, N, args.m)
        u = sample_quenched_uniforms(len(edge_nodes), rng)
        rc, shE = theory.rho_c(k0, k2, args.m), theory.S_h_E(k0)
        Cstar = theory.C_threshold(k0, args.m)

        ts = []
        for frac in (1.5, 0.75):
            C = frac * Cstar
            t0 = time.perf_counter()
            simulate_quenched(edge_nodes, node_edges, edge_p, C,
                              theory.S_h_1star(k0, args.m, C), shE, rc, u)
            ts.append(time.perf_counter() - t0)

        total = t_gen + t_inc + t_mom + sum(ts)
        print(f"{N:>8} {len(edge_nodes):>9} {t_gen:>9.2f} {t_inc:>9.2f} "
              f"{t_mom:>11.2f} {ts[0]:>9.2f} {ts[1]:>10.2f} {total:>14.2f}")


if __name__ == "__main__":
    main()
