"""扫描 N × 容量 C，对照 T_c^sim 与 T_c^th（main.tex 0806 修正版）。

每个 N 独立生成幂律超图（配置模型），按该 N 实测 k0 取两个 C 示例
（0.75*C* 三阶段 / 1.5*C* 两阶段，避开退化区）；每 (N, C) 重复多次，
报告 T_c^sim(mean±std)、T_density、T_c^th、ratio=T_c^sim/T_c^th。

输出：
  results/sweep_m{m}_g{gamma}.csv
  figures/sweep_m{m}_g{gamma}_Tcsim_vs_Tcth.png   (y=x 参考线，按 N 标注)
  figures/sweep_m{m}_g{gamma}_ratio_vs_N.png

用法:
    .venv/Scripts/python.exe experiments/run_sweep.py --m 2 --repeat 5 --Ns 200 500
"""
from __future__ import annotations

import argparse
import csv
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


def run_one(N: int, m: int, gamma: float, k_min: int, C: float, repeat: int,
            seed_base: int) -> dict:
    """对 (N, C) 跑 repeat 次，返回汇总行。"""
    rec = max(1, N // 100)  # 大 N 降采样记录（不影响 T_c 判定精度量级）
    tcs, tds, k0s, k2s, tcths = [], [], [], [], []
    for i in range(repeat):
        r = simulate(Config(N=N, m=m, gamma=gamma, k_min=k_min, C=C,
                            seed=seed_base + i, record_every=rec))
        tcs.append(r.Tc_sim if r.Tc_sim else np.nan)
        tds.append(r.T_density if r.T_density else np.nan)
        k0s.append(r.k0)
        k2s.append(r.k2)
        tcths.append(r.Tc_th)
    tcs = np.array(tcs, float)
    tds = np.array(tds, float)
    Tc_th = float(np.nanmean(tcths))
    Tc_sim_mean = float(np.nanmean(tcs))
    Td_mean = float(np.nanmean(tds))
    return dict(
        N=N, C=C, regime=theory.regime(float(np.mean(k0s)), m, C),
        k0_actual=float(np.mean(k0s)), k2_actual=float(np.mean(k2s)),
        Tc_th=Tc_th,
        Tc_sim_mean=Tc_sim_mean,
        Tc_sim_std=float(np.nanstd(tcs)),
        T_density_mean=Td_mean,
        ratio=Tc_sim_mean / Tc_th if Tc_th > 0 else np.nan,      # S<eps 口径
        ratio_density=Td_mean / Tc_th if Tc_th > 0 else np.nan,  # S_h=rho_c 口径
        n_valid=int(np.sum(~np.isnan(tcs))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=2.5)
    ap.add_argument("--k_min", type=int, default=1,
                    help="超度下截断（k_min=3 时 k0≈6，贴近论文 ⟨k⟩₀≈8 的参数区）")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--Ns", type=int, nargs="+", default=[200, 500, 1000])
    ap.add_argument("--Cs", type=float, nargs="+", default=None,
                    help="覆盖 C 示例（否则每 N 自动取 0.75C*/1.5C*）")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = []
    for N in args.Ns:
        # probe 拿实测 k0（生成与 C 无关），确定 C 示例
        probe = simulate(Config(N=N, m=args.m, gamma=args.gamma, k_min=args.k_min,
                                C=1.0, seed=args.seed))
        if args.Cs is None:
            Cstar = theory.C_threshold(probe.k0, args.m)
            cs = [round(0.75 * Cstar, 4), round(1.5 * Cstar, 4)]
        else:
            cs = [float(c) for c in args.Cs]
        for C in cs:
            row = run_one(N, args.m, args.gamma, args.k_min, C, args.repeat,
                          args.seed)
            rows.append(row)
            print(f"N={N} C={C:.4f} [{row['regime']}]: "
                  f"Tc_sim={row['Tc_sim_mean']:.0f}±{row['Tc_sim_std']:.0f} | "
                  f"Tc_th={row['Tc_th']:.0f} | ratio(S)={row['ratio']:.3f} "
                  f"ratio(S_h)={row['ratio_density']:.3f} | "
                  f"T_density={row['T_density_mean']:.0f} | "
                  f"k0={row['k0_actual']:.2f} k2={row['k2_actual']:.1f}")

    tag = f"m{args.m}_g{args.gamma}_kmin{args.k_min}"
    # CSV
    csv_path = os.path.join(RESULTS, f"sweep_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved csv: {csv_path}")

    # 图 1: Tc_sim vs Tc_th（y=x）
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for reg, marker in [("case_II", "o"), ("case_I", "s")]:
        sub = [r for r in rows if r["regime"] == reg]
        if not sub:
            continue
        ax.errorbar([r["Tc_th"] for r in sub], [r["Tc_sim_mean"] for r in sub],
                    yerr=[r["Tc_sim_std"] for r in sub],
                    fmt=marker, ms=6, capsize=3, label=reg)
        for r in sub:
            ax.annotate(f"N={r['N']}", (r["Tc_th"], r["Tc_sim_mean"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    lim = [min(r["Tc_th"] for r in rows) * 0.8, max(r["Tc_th"] for r in rows) * 1.2]
    ax.plot(lim, lim, "k--", lw=1, label="y=x (theory = sim)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("T_c^th (mean-field)"); ax.set_ylabel("T_c^sim")
    ax.set_title(f"T_c^sim vs T_c^th, m={args.m}, gamma={args.gamma}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(FIGURES, f"sweep_{tag}_Tcsim_vs_Tcth.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # 图 2: ratio vs N（两条 C 线）
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for reg, color in [("case_II", "tab:blue"), ("case_I", "tab:red")]:
        sub = [r for r in rows if r["regime"] == reg]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: r["N"])
        ax.errorbar([r["N"] for r in sub], [r["ratio"] for r in sub],
                    yerr=[r["Tc_sim_std"] / r["Tc_th"] for r in sub],
                    fmt="o-", color=color, capsize=3, label=reg)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="ratio=1 (theory=sim)")
    ax.set_xlabel("N"); ax.set_ylabel("T_c^sim / T_c^th")
    ax.set_title(f"Ratio T_c^sim/T_c^th vs N, m={args.m}, gamma={args.gamma}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(FIGURES, f"sweep_{tag}_ratio_vs_N.png")
    fig.savefig(p2, dpi=120)
    plt.close(fig)

    print(f"saved figures: {p1}, {p2}")


if __name__ == "__main__":
    main()
