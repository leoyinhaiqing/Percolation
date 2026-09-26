"""分段与尾段弱级联直接诊断（audit 后续，findings ⑬⑭ 证据）。

对每 (N, C, seed) 事件驱动跑完整轨迹（不算巨分量，只跟踪 S_h 阈值穿越），记录：
  1) 三段实测时长：t_sh1（S_h 首穿 S_h^1）、t_shE（首穿 S_h^E）、t_rhoc（首穿 rho_c）
     —— 与理论 t^1 / t^E / T_c 的分段直接对照（此前只能从 T_density 端到端反推）；
  2) 各区域每事件级联规模（extras = 本事件坍塌边数 - 1）：
     - 初始段（S_h > S_h^1）：验证"纯随机"假设（extras 应≈0）
     - 级联段（S_h^E <= S_h <= S_h^1）
     - 尾段（rho_c < S_h < S_h^E）：直接测弱级联（替代此前仅触发时刻 5 事件的外推）
  3) 理论对照：平均场 t^1 与 Laplace 修正 t^1
     - 闭式（b~U(0,1) 的 Irwin-Hall）：S_h(t) = ((1-e^{-ct})/(ct))^{m+1}，c = m!/N^m
     - 经验（该图实测 edge_p）：S_h(t) = mean_e (1-p_e)^t —— 检验修正 ansatz 上限
     survivor bias：p_e 异质、高危边先死 → S_h 衰减慢于 e^{-pt}，t^1 被平均场低估。

用法（当前 canonical CSV 的产出命令）：
  .venv/Scripts/python.exe experiments/run_tail_diag.py --Ns 500 1000 2000 5000 10000 --seeds 15
实测运行时间：全五 N × 15 seeds × 2 C 共 5m18s（N=10^4 × 5 seeds × 2 C 单独跑为 71s）。
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
from src.cascade import stage2_redistribute_cascade  # noqa: E402
from src.fastsim import (  # noqa: E402
    build_incidence_lists, sample_quenched_uniforms, simulate_quenched,
)
from src.hypergraph import (  # noqa: E402
    build_incidence, generate_powerlaw_hypergraph, hyperdegree_moments,
    sample_vulnerability,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def solve_t_by_bisection(f, target: float, t_hi: float) -> float:
    """f 单调递减，求 f(t)=target 的 t（二分，相对精度 1e-6）。"""
    lo, hi = 0.0, t_hi
    while f(hi) > target:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-6 * hi:
            break
    return 0.5 * (lo + hi)


def t1_corrected_closed_form(N: int, m: int, sh1: float) -> float:
    """Laplace 修正 t^1（闭式）：((1-e^{-ct})/(ct))^{m+1} = S_h^1，c=m!/N^m。"""
    c = math.factorial(m) / N ** m

    def sh(t: float) -> float:
        x = c * t
        return 1.0 if x == 0.0 else ((1.0 - math.exp(-x)) / x) ** (m + 1)

    return solve_t_by_bisection(sh, sh1, 10.0 / c)


def t1_corrected_empirical(edge_p: np.ndarray, sh1: float) -> float:
    """Laplace 修正 t^1（经验版）：mean_e (1-p_e)^t = S_h^1（用实测边率分布）。"""
    log1m = np.log1p(-edge_p)

    def sh(t: float) -> float:
        return float(np.exp(log1m * t).mean())

    return solve_t_by_bisection(sh, sh1, 10.0 / float(edge_p.mean()))


def run_one(N: int, m: int, k_min: int, C: float, seed: int,
            engine: str = "event") -> dict:
    """单 seed 完整轨迹：分段时长 + 各区弱级联统计 + 理论对照。

    engine="event"    —— 事件驱动 Gillespie（canonical CSV 的产出路径，勿改）；
    engine="quenched" —— fastsim 淬火时钟引擎（离散模型的精确仿真，大 N 用）。
    两者在同一 (N, C, seed) 上共用同一张超图与同一组 b，只有坍塌时刻的抽样方式不同；
    统计等价性由 experiments/validate_fastsim.py 逐 N 核验。
    """
    rng = np.random.default_rng(seed)
    edge_nodes, _ = generate_powerlaw_hypergraph(N, m, gamma=2.5, k_min=k_min,
                                                 rng=rng)
    node_edges = build_incidence(edge_nodes, N)
    k0, k2 = hyperdegree_moments(node_edges)
    b = sample_vulnerability(N, "uniform", rng)
    b_mean = float(b.mean())
    E0 = len(edge_nodes)
    edge_p = math.factorial(m) / N ** m * b[edge_nodes].sum(axis=1)
    load = np.full(E0, 1.0)
    alive = np.ones(E0, dtype=bool)

    sh1 = theory.S_h_1star(k0, m, C)
    shE = theory.S_h_E(k0)
    rc = theory.rho_c(k0, k2, m)
    p = theory.collapse_prob_p(N, m, b_mean)
    t1_mf = theory.t_1(N, m, b_mean, sh1)
    tE_mf = theory.t_E(N, m, k0, b_mean, C)
    tail_mf = (1.0 / p) * math.log(1.0 / (rc * k0)) if rc * k0 < 1.0 else float("nan")
    t1_ih = t1_corrected_closed_form(N, m, sh1)
    t1_emp = t1_corrected_empirical(edge_p, sh1)
    t_max = int(5.0 * theory.collapse_time_Tc(N, m, k0, k2, b_mean, C)) + 100

    if engine == "quenched":
        res = simulate_quenched(
            edge_nodes, build_incidence_lists(edge_nodes, N), edge_p, C,
            sh1, shE, rc, sample_quenched_uniforms(E0, rng), collect_extras=True)
        t_sh1, t_shE, t_rhoc = (res["t_sh1_sim"], res["t_shE_sim"],
                                res["t_rhoc_sim"])
        extras = res["extras"]
        return _assemble(N, m, C, seed, k0, k2, b_mean, E0, t_sh1, t_shE, t_rhoc,
                         t1_mf, t1_ih, t1_emp, tE_mf, tail_mf, extras)
    if engine != "event":
        raise ValueError(f"unknown engine: {engine!r}")

    t = 0
    t_sh1 = t_shE = t_rhoc = None
    extras = {"init": [], "casc": [], "tail": []}
    while alive.any() and t < t_max and t_rhoc is None:
        w = edge_p[alive]
        ptot = float(w.sum())
        if ptot == 0.0:
            break
        t += int(rng.geometric(ptot))
        if t > t_max:
            break
        ai = np.nonzero(alive)[0]
        e = int(rng.choice(ai, p=w / ptot))
        Sh_before = alive.sum() / E0
        collapsed, _ = stage2_redistribute_cascade(
            edge_nodes, node_edges, load, alive, C, np.array([e]))
        region = ("init" if Sh_before > sh1
                  else "casc" if Sh_before >= shE else "tail")
        extras[region].append(len(collapsed) - 1)
        Sh = alive.sum() / E0
        if t_sh1 is None and Sh <= sh1:
            t_sh1 = t
        if t_shE is None and Sh <= shE:
            t_shE = t
        if t_rhoc is None and Sh < rc:
            t_rhoc = t

    return _assemble(N, m, C, seed, k0, k2, b_mean, E0, t_sh1, t_shE, t_rhoc,
                     t1_mf, t1_ih, t1_emp, tE_mf, tail_mf, extras)


def _assemble(N, m, C, seed, k0, k2, b_mean, E0, t_sh1, t_shE, t_rhoc,
              t1_mf, t1_ih, t1_emp, tE_mf, tail_mf, extras) -> dict:
    """把三段时刻与分区级联统计装配成 CSV 行（两个引擎共用同一 schema）。"""

    def stats(xs: list) -> tuple:
        if not xs:
            return (0, float("nan"), float("nan"), float("nan"))
        a = np.array(xs, float)
        return (len(a), float(a.mean()), float(np.percentile(a, 90)), float(a.max()))

    n_i, mean_i, p90_i, max_i = stats(extras["init"])
    n_c, mean_c, p90_c, max_c = stats(extras["casc"])
    n_t, mean_t, p90_t, max_t = stats(extras["tail"])
    return dict(
        N=N, C=C, regime=theory.regime(k0, m, C), seed=seed,
        k0=k0, k2=k2, b_mean=b_mean, E0=E0,
        t_sh1_sim=t_sh1, t_shE_sim=t_shE, t_rhoc_sim=t_rhoc,
        t1_mf=t1_mf, t1_laplace_ih=t1_ih, t1_laplace_emp=t1_emp,
        casc_mf=tE_mf - t1_mf, tail_mf=tail_mf,
        n_events_init=n_i, extras_mean_init=mean_i,
        n_events_casc=n_c, extras_mean_casc=mean_c, extras_p90_casc=p90_c,
        extras_max_casc=max_c,
        n_events_tail=n_t, extras_mean_tail=mean_t, extras_p90_tail=p90_t,
        extras_max_tail=max_t,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--engine", choices=("event", "quenched"), default="event",
                    help="event=canonical 事件驱动；quenched=fastsim 淬火时钟（大 N）")
    ap.add_argument("--out", default=None,
                    help="results/ 下的输出文件名（默认 tail_diag_m{m}_kmin{k_min}.csv）")
    args = ap.parse_args()

    rows = []
    for N in args.Ns:
        probe = generate_powerlaw_hypergraph(N, args.m, gamma=2.5,
                                             k_min=args.k_min,
                                             rng=np.random.default_rng(1))[0]
        k0p, _ = hyperdegree_moments(build_incidence(probe, N))
        Cstar = theory.C_threshold(k0p, args.m)
        for C in (round(0.75 * Cstar, 4), round(1.5 * Cstar, 4)):
            for sd in range(args.seeds):
                r = run_one(N, args.m, args.k_min, C, sd, engine=args.engine)
                rows.append(r)
                trig = (f"{r['t_sh1_sim'] / r['t1_mf']:.3f}"
                        if r["t_sh1_sim"] else "n/a")
                print(f"N={N} C={C:.3f} [{r['regime']}] seed={sd}: "
                      f"t_sh1/t1_mf={trig} "
                      f"t1_laplace/t1_mf={r['t1_laplace_ih'] / r['t1_mf']:.3f} | "
                      f"tail events={r['n_events_tail']} "
                      f"extras mean={r['extras_mean_tail']:.2f} "
                      f"p90={r['extras_p90_tail']:.1f} max={r['extras_max_tail']:.0f}")

    out_name = args.out or f"tail_diag_m{args.m}_kmin{args.k_min}.csv"
    path = os.path.join(RESULTS, out_name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
