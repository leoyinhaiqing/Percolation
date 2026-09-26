"""单次（或多次重复）仿真：输出 S(t)、S_h(t)、L(t) 曲线与理论对照。

新框架（main.tex 0806 修正版）：
  - 幂律超图（配置模型），显式容量 C（可给一个或多个，跨 C* 阈值）；
  - 未指定 C 时自动取 0.75*C*（三阶段）与 1.5*C*（两阶段）两个示例值；
  - 叠加理论竖线（t^1、T_c^th）与横线（rho_c、S_h^1、S_h^2、S_h*、eps、C）。

用法:
    .venv/Scripts/python.exe experiments/run_single.py --N 200 --m 2 --seed 1
    .venv/Scripts/python.exe experiments/run_single.py --N 200 --C 2.0 3.0 --repeat 5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import theory  # noqa: E402
from src.config import Config  # noqa: E402
from src.dynamics import simulate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def align(ts_list, val_list):
    """把多条（不等长）时间序列插值到公共网格 [0, min_last_t]。"""
    tmax = min(ts[-1] for ts in ts_list)
    grid = np.arange(tmax + 1)
    aligned = np.array([np.interp(grid, ts, v) for ts, v in zip(ts_list, val_list)])
    return grid, aligned


def report_one(C: float, results) -> dict:
    r0 = results[0]
    t1 = theory.t_1(r0.cfg.N, r0.cfg.m, r0.b_mean,
                    theory.S_h_1star(r0.k0, r0.cfg.m, C))
    print(f"--- C={C:.4f}  regime={r0.regime}  ({len(results)} repeats) ---")
    print(f"E0={r0.E0}  <k>0={r0.k0:.4f}  <k^2>0={r0.k2:.4f}  <b>={r0.b_mean:.4f}")
    print(f"S(0)={r0.S0:.4f}  B(0)={r0.B0:.4f}  rho_c={r0.rho_c:.4f}")
    print(f"C* = 2m/(m+1)<k>0 = {r0.C_threshold:.4f}  C_crit={r0.C_crit:.4f}")
    print(f"p={r0.p:.3e}  t^1(avg k0)={t1:.1f}")
    tcs = np.array([r.Tc_sim if r.Tc_sim else np.nan for r in results])
    tds = np.array([r.T_density if r.T_density else np.nan for r in results])
    print(f"T_c^th={r0.Tc_th:.1f}  "
          f"T_c^sim mean={np.nanmean(tcs):.1f}±{np.nanstd(tcs):.1f} "
          f"(n={np.sum(~np.isnan(tcs))})  ratio={np.nanmean(tcs)/r0.Tc_th:.2f}")
    print(f"T_density mean={np.nanmean(tds):.1f}±{np.nanstd(tds):.1f}  "
          f"lost_load={r0.lost_load:.3f}")
    return dict(C=C, Tc_th=r0.Tc_th, Tc_sim=np.nanmean(tcs), Tc_sim_std=np.nanstd(tcs),
                T_density=np.nanmean(tds), ratio=np.nanmean(tcs) / r0.Tc_th)


def plot_one(C: float, results, tag: str, save_npz: bool = True) -> str:
    r0 = results[0]
    cfg = r0.cfg
    t1 = theory.t_1(cfg.N, cfg.m, r0.b_mean, theory.S_h_1star(r0.k0, cfg.m, C))
    sh1 = theory.S_h_1star(r0.k0, cfg.m, C)
    sh2 = theory.S_h_2star(r0.k0, cfg.m, C)
    shs = theory.S_h_star(r0.k0, cfg.m)

    grid, S = align([r.ts for r in results], [r.S for r in results])
    _, Sh = align([r.ts for r in results], [r.Sh for r in results])
    _, L = align([r.ts for r in results], [r.L for r in results])
    s_mean, s_std = S.mean(0), S.std(0)
    sh_mean, sh_std = Sh.mean(0), Sh.std(0)
    l_mean, l_std = L.mean(0), L.std(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # S(t)
    ax = axes[0]
    ax.plot(grid, s_mean, "b-", lw=1.5, label="mean S(t)")
    ax.fill_between(grid, s_mean - s_std, s_mean + s_std, alpha=0.25)
    ax.axhline(cfg.eps, color="gray", ls=":", label=f"eps={cfg.eps}")
    ax.axvline(t1, color="brown", ls="--", lw=1, label=f"t^1={t1:.0f}")
    ax.axvline(r0.Tc_th, color="blue", ls="-.", label=f"T_c^th={r0.Tc_th:.0f}")
    tc = np.nanmean([r.Tc_sim if r.Tc_sim else np.nan for r in results])
    if np.isfinite(tc):
        ax.axvline(tc, color="r", ls="--", label=f"T_c^sim~{tc:.0f}")
    ax.set_xlabel("t"); ax.set_ylabel("S")
    ax.set_title(f"Giant component S(t), C={C:.3g}")
    ax.legend(fontsize=7)
    # S_h(t)
    ax = axes[1]
    ax.plot(grid, sh_mean, "g-", lw=1.5, label="mean S_h(t)")
    ax.fill_between(grid, sh_mean - sh_std, sh_mean + sh_std, alpha=0.25)
    for val, name, color in [(sh1, "S_h^1", "purple"), (sh2, "S_h^2", "orange"),
                             (shs, "S_h*", "magenta"), (r0.rho_c, "rho_c", "k")]:
        ax.axhline(val, color=color, ls=":", lw=1, label=f"{name}={val:.3f}")
    ax.set_xlabel("t"); ax.set_ylabel("S_h")
    ax.set_title(f"Hyperedge density S_h(t), C={C:.3g}")
    ax.legend(fontsize=7)
    # L(t)
    ax = axes[2]
    ax.plot(grid, l_mean, "r-", lw=1.5, label="mean L(t)")
    ax.fill_between(grid, l_mean - l_std, l_mean + l_std, alpha=0.25)
    ax.axhline(C, color="gray", ls=":", label=f"C={C:.3g}")
    ax.set_xlabel("t"); ax.set_ylabel("L")
    ax.set_title(f"Mean load L(t), C={C:.3g}")
    ax.legend(fontsize=7)

    fig.suptitle(f"N={cfg.N} m={cfg.m} gamma={cfg.gamma} | {r0.regime} | "
                 f"<k>0={r0.k0:.2f} <k^2>0={r0.k2:.1f} | seed={cfg.seed}")
    fig.tight_layout()
    path = os.path.join(FIGURES, f"single_{tag}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    if save_npz:
        np.savez(os.path.join(RESULTS, f"single_{tag}.npz"),
                 ts=grid, S=s_mean, S_std=s_std, Sh=sh_mean, Sh_std=sh_std,
                 L=l_mean, L_std=l_std, Tc_th=r0.Tc_th,
                 Tc_sim=np.nanmean([r.Tc_sim if r.Tc_sim else np.nan for r in results]),
                 T_density=np.nanmean([r.T_density if r.T_density else np.nan for r in results]))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=2.5)
    ap.add_argument("--k_min", type=int, default=1,
                    help="超度下截断（k_min=3 时 k0≈6，贴近论文 ⟨k⟩₀≈8 的参数区）")
    ap.add_argument("--C", type=float, nargs="+", default=None,
                    help="容量（可多个，跨 C* 阈值）；缺省自动取 0.75C*/1.5C* 两例")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    # 先跑一次拿实测 k0，确定默认 C 示例（避开 S_h^1>=1 退化区）。
    # 生成与 C 无关，probe 用临时 C=1.0 即可。
    probe = simulate(Config(N=args.N, m=args.m, gamma=args.gamma, k_min=args.k_min,
                            C=1.0, seed=args.seed))
    if args.C is None:
        Cstar = theory.C_threshold(probe.k0, args.m)
        cs = [round(0.75 * Cstar, 4), round(1.5 * Cstar, 4)]
    else:
        cs = [float(c) for c in args.C]

    for C in cs:
        results = [simulate(Config(N=args.N, m=args.m, gamma=args.gamma,
                                   k_min=args.k_min, C=C, seed=args.seed + i))
                   for i in range(args.repeat)]
        report_one(C, results)
        tag = f"N{args.N}_m{args.m}_kmin{args.k_min}_C{C:g}_s{args.seed}"
        p = plot_one(C, results, tag)
        print(f"saved figure: {p}")


if __name__ == "__main__":
    main()
