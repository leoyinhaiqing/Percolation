"""ΔT_c 的信噪比诊断：排序要稳定下来需要多少个淬火时钟？

单节点扰动只把 k_i 条边的时钟提前 (2b0+1)/(3b0) 倍，而 T_c 由 S_h 何时跌破 rho_c
决定——在 T_c 附近只剩 ~rho_c 比例的边还活着，所以"被提前的边里有没有恰好是晚期
幸存者"本身是个近乎二值的抽签。CRN 消掉了网络实现与时钟的共同涨落，但消不掉这个
抽签方差，只能靠时钟数 R 平均掉。

本脚本对每个活跃节点跑满 R_max 个时钟并保留逐时钟配对差，再按前 R 个截断，报告：
  - rho(dTc_R, k)      与超度的秩相关（低不代表错，只说明动态排序不是静态排序）
  - rho(dTc_R, dTc_max) 与"满时钟"排序的秩相关 —— 这才是收敛判据
  - 中位 sem/|dTc|      单节点估计的相对误差

用法：
  .venv/Scripts/python.exe experiments/run_vital_diag.py --N 400 --R_max 120
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_vital_nodes import Setup, spearman  # noqa: E402
from src.centrality import hyperdegree_centrality  # noqa: E402
from src.hypergraph import generate_powerlaw_hypergraph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--b0", type=float, default=0.4)
    ap.add_argument("--C_frac", type=float, default=1.5)
    ap.add_argument("--R_max", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    edge_nodes, _ = generate_powerlaw_hypergraph(
        args.N, args.m, gamma=2.5, k_min=args.k_min,
        rng=np.random.default_rng(args.seed))
    k = hyperdegree_centrality(edge_nodes, args.N)
    S = Setup(edge_nodes, args.N, args.m, args.b0, args.C_frac, args.R_max,
              args.seed + 1)
    active = [i for i in range(args.N) if S.node_edges[i]]

    base = np.array(S.Tc(S.edge_p, S.clocks), float)
    print(f"E0={S.E0} regime={S.regime} T_c base mean={base.mean():.4g} "
          f"cv={base.std() / base.mean():.3f}")

    t0 = time.perf_counter()
    D = np.empty((len(active), args.R_max))
    for j, i in enumerate(active):
        D[j] = base - np.array(S.Tc(S.perturbed_rates([i]), S.clocks), float)
    print(f"{len(active)} nodes x {args.R_max} clocks in "
          f"{time.perf_counter() - t0:.1f}s")

    ref = D.mean(axis=1)
    ka = k[active]
    print(f"{'R':>5} {'rho(dTc,k)':>11} {'rho(dTc,ref)':>13} "
          f"{'mean dTc/Tc':>12} {'med sem/|d|':>12} {'frac>0':>7}")
    for R in (3, 5, 10, 20, 40, 80, args.R_max):
        if R > args.R_max:
            continue
        d = D[:, :R].mean(axis=1)
        sem = D[:, :R].std(axis=1, ddof=1) / np.sqrt(R)
        print(f"{R:>5} {spearman(d, ka):>+11.3f} {spearman(d, ref):>+13.3f} "
              f"{d.mean() / base.mean():>12.2e} "
              f"{np.median(sem / np.abs(d)):>12.2f} {(d > 0).mean():>7.2f}")


if __name__ == "__main__":
    main()
