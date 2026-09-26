"""Step 1：生成幂律超图并报告初始指标。

对应 experiment_plan.md Track A Step 1：
生成 gamma=2.5 幂律超图，报告 <k>_0, <k^2>_0, |E|, S(0), rho_c, B(0),
S_h*（B=1 点）、C* 阈值与 C_crit、每个 C 的 S_h^1/S_h^2 与 regime，脆弱度与 <b>。

用法（venv 解释器）：
  .venv/Scripts/python.exe experiments/run_step1.py --N 1000 --m 2 --seed 1
  .venv/Scripts/python.exe experiments/run_step1.py --N 1000 --C 2.0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import hypergraph as hg  # noqa: E402
from src import theory  # noqa: E402


def build_report(N: int, m: int, gamma: float, k_min: int, k_max, seed: int,
                 C=None) -> dict:
    rng = np.random.default_rng(seed)
    edge_nodes, discarded = hg.generate_powerlaw_hypergraph(
        N, m, gamma=gamma, k_min=k_min, k_max=k_max, rng=rng)
    E = len(edge_nodes)
    node_edges = hg.build_incidence(edge_nodes, N)
    k0, k2 = hg.hyperdegree_moments(node_edges)
    alive = np.ones(E, dtype=bool)
    S0 = hg.giant_component_fraction(edge_nodes, alive, N)

    b = hg.sample_vulnerability(N, "uniform", rng)
    b_mean = float(b.mean())

    rc = theory.rho_c(k0, k2, m)
    B0 = theory.branching_B(k0, m, S_h=1.0)
    S_h_star_B1 = theory.S_h_star(k0, m)
    C_star = theory.C_threshold(k0, m)
    Cc = theory.C_crit(k0, k2, m)
    p = theory.collapse_prob_p(N, m, b_mean)

    rep = dict(
        N=N, m=m, gamma=gamma, k_min=k_min, E=E, discarded=discarded,
        k0=k0, k2=k2, S0=S0, b_mean=b_mean,
        rho_c=rc, B0=B0, S_h_star_B1=S_h_star_B1, C_threshold=C_star,
        C_crit=Cc, p=p,
        has_gc=theory.has_initial_gc(k0, k2, m),
        mr_ratio=theory.molloy_reed_ratio(k0, k2),
    )
    # 若给定 C（或用示例的两个 C 跨阈值），报告 S_h^1/S_h^2 与 regime
    # 示例取 0.75C*（三阶段）与 1.5C*（两阶段）：避开 S_h^1>=1 退化区（C 过小）
    if C is not None:
        cs = [float(C)]
    else:
        cs = [round(0.75 * C_star, 4), round(1.5 * C_star, 4)]
    rep["C_cases"] = [
        dict(C=c,
             S_h_1star=theory.S_h_1star(k0, m, c),
             S_h_2star=theory.S_h_2star(k0, m, c),
             regime=theory.regime(k0, m, c))
        for c in cs
    ]
    return rep


def print_report(r: dict) -> None:
    print("=" * 60)
    print(f"Step 1 report  (N={r['N']} m={r['m']} gamma={r['gamma']} "
          f"k_min={r['k_min']})")
    print("=" * 60)
    print(f"  |E|                = {r['E']}")
    print(f"  discarded stubs    = {r['discarded']:.3%}")
    print(f"  <k>_0              = {r['k0']:.4f}")
    print(f"  <k^2>_0            = {r['k2']:.4f}")
    print(f"  <k(k-1)>/<k>       = {r['mr_ratio']:.4f}   (> 1/m={1.0/r['m']:.4f} "
          f"→ initial GC: {r['has_gc']})")
    print(f"  S(0)               = {r['S0']:.4f}")
    print(f"  <b>                = {r['b_mean']:.4f}")
    print(f"  rho_c              = {r['rho_c']:.4f}")
    print(f"  B(0)               = {r['B0']:.4f}   (cascade branching; needs >0)")
    print(f"  S_h* (B=1)         = {r['S_h_star_B1']:.4f}")
    print(f"  C* = 2m/(m+1)<k>_0 = {r['C_threshold']:.4f}   "
          f"(C>=C* two-stage / C<C* three-stage)")
    print(f"  C_crit             = {r['C_crit']:.4f}   (rho_c=S_h^2; 需 L(0)<C<C_crit)")
    print(f"  p                  = {r['p']:.3e}")
    for cc in r["C_cases"]:
        print(f"  C={cc['C']:.4f}: S_h^1={cc['S_h_1star']:.4f} "
              f"S_h^2={cc['S_h_2star']:.4f}  regime={cc['regime']}")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=2.5)
    ap.add_argument("--k_min", type=int, default=1)
    ap.add_argument("--k_max", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    r = build_report(args.N, args.m, args.gamma, args.k_min, args.k_max,
                     args.seed, C=args.C)
    print_report(r)


if __name__ == "__main__":
    main()
