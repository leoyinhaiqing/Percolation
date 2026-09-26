"""核验 fastsim 淬火时钟引擎与 canonical 事件驱动引擎的统计等价性 + 加速比。

两个引擎在同一 (N, C, seed) 上共用**同一张超图与同一组 b**（run_one 内 rng 消耗
顺序相同），差别只在坍塌时刻的抽样方式：
  - event    每事件采样等待步 dt~Geom(sum p_e) 并按 p_e 选一条边（稀有事件近似，
             同一步至多一条边自发坍塌）；
  - quenched 每条边一次性抽 tau_e~Geom(p_e) 后按序处理（离散模型的精确仿真）。
因此逐 seed 的数值不会相同，需要比的是**每个 N 的分布**：均值之差是否落在配对
噪声内（报告 Welch t 与相对偏差），以及三段口径是否一致。

用法：
  .venv/Scripts/python.exe experiments/validate_fastsim.py --Ns 500 1000 2000 --seeds 15
  .venv/Scripts/python.exe experiments/validate_fastsim.py --Ns 5000 10000 --seeds 8
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_tail_diag import run_one  # noqa: E402
from src import theory  # noqa: E402
from src.hypergraph import (  # noqa: E402
    build_incidence, generate_powerlaw_hypergraph, hyperdegree_moments,
)

METRICS = ("t_sh1_sim", "t_shE_sim", "t_rhoc_sim")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t 统计量（不假设等方差）。"""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    s = math.sqrt(va + vb)
    return float("nan") if s == 0.0 else float((a.mean() - b.mean()) / s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=15)
    args = ap.parse_args()

    print(f"{'N':>7} {'regime':>8} {'metric':>12} {'event':>12} {'quenched':>12} "
          f"{'rel diff':>9} {'Welch t':>8}")
    timing: dict[str, float] = {"event": 0.0, "quenched": 0.0}

    for N in args.Ns:
        probe = generate_powerlaw_hypergraph(N, args.m, gamma=2.5, k_min=args.k_min,
                                             rng=np.random.default_rng(1))[0]
        k0p, _ = hyperdegree_moments(build_incidence(probe, N))
        Cstar = theory.C_threshold(k0p, args.m)
        for C in (round(0.75 * Cstar, 4), round(1.5 * Cstar, 4)):
            out: dict[str, list[dict]] = {}
            for eng in ("event", "quenched"):
                t0 = time.perf_counter()
                out[eng] = [run_one(N, args.m, args.k_min, C, sd, engine=eng)
                            for sd in range(args.seeds)]
                timing[eng] += time.perf_counter() - t0
            regime = out["event"][0]["regime"]
            for metric in METRICS:
                a = np.array([r[metric] for r in out["event"]], float)
                b = np.array([r[metric] for r in out["quenched"]], float)
                rel = b.mean() / a.mean() - 1.0
                print(f"{N:>7} {regime:>8} {metric:>12} {a.mean():>12.4g} "
                      f"{b.mean():>12.4g} {rel:>+8.2%} {welch_t(a, b):>8.2f}")

    speedup = timing["event"] / timing["quenched"] if timing["quenched"] else float("nan")
    print(f"\nwall clock: event {timing['event']:.1f}s, "
          f"quenched {timing['quenched']:.1f}s  ->  speedup x{speedup:.1f}")


if __name__ == "__main__":
    main()
