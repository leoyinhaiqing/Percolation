"""参数依赖图组（fig5）：T_c 对 C、m、<k>_0、N、<b> 的依赖 + 全局一致性。

读 results/param_sweep_m2_kmin3.csv（experiments/run_param_sweep.py 产出）。
六个面板：
  (a) T_c vs C/C*     —— regime 交界；纵轴线性以看清 C* 附近的形状变化
  (b) T_c vs <k>_0    —— 两支各一条曲线
  (c) T_c vs m        —— 对数纵轴，附 N^m 参考斜率
  (d) T_c vs N        —— 对数双轴，附最小二乘标度指数
  (e) T_c vs <b>      —— 对数双轴，附 1/<b> 参考线
  (f) T_c^th/T_c^sim  —— 只画 C sweep（精度唯一显著变化的方向）；其余四个 sweep 的
                          Case I 最大偏差放在图脚注记（它是整张图的注记，不属于 (f)）

配色沿用论文图组：Case I 蓝、Case II 朱红、Case II 校准绿。

用法：
  .venv/Scripts/python.exe experiments/make_param_figures.py
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.make_figures import BLUE, COL2, GREEN, INK, MUTED, VERM, save, set_style  # noqa: E402
from src import theory  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

FLOATS = ("x", "x_ctrl", "k0", "k2", "b_mean", "C", "C_star", "C_over_Cstar",
          "rho_c", "S_h_E", "S_h_1", "Tc_sim", "Tc_th", "Tc_th_calib", "E0")


def load(name: str) -> list[dict]:
    path = name if os.path.isabs(name) else os.path.join(RESULTS, name)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            r = dict(raw)
            for c in FLOATS:
                r[c] = float(r[c]) if r.get(c) not in (None, "") else math.nan
            r["N"], r["m"] = int(float(r["N"])), int(float(r["m"]))
            rows.append(r)
    return rows


def aggregate(rows: list[dict], sweep: str) -> dict:
    """按 (x_ctrl, regime) 聚合，返回 {regime: dict of arrays}（按 x 排序）。"""
    g = defaultdict(list)
    for r in rows:
        if r["sweep"] == sweep:
            g[(r["x_ctrl"], r["regime"])].append(r)
    out: dict = {}
    for rgm in ("case_I", "case_II"):
        keys = sorted(k for k in g if k[1] == rgm)
        if not keys:
            continue
        d = defaultdict(list)
        for k in keys:
            rs = g[k]
            sim = np.array([r["Tc_sim"] for r in rs])
            # 校准列就地重算，不读 CSV 的 Tc_th_calib：那一列写死了产出时的
            # lambda(N)，改了 src/theory.py 的校准式后会与图不一致。
            # Case I 无校准（lambda 只进 Case II），恒等于 th。
            cal = np.array([r["Tc_th"] * (theory.lambda_calibration(r["N"])
                                          if rgm == "case_II" else 1.0)
                            for r in rs])
            d["x"].append(np.mean([r["x"] for r in rs]))
            d["N"].append(np.mean([r["N"] for r in rs]))
            d["sim"].append(sim.mean())
            d["sem"].append(sim.std(ddof=1) / math.sqrt(len(sim)) if len(sim) > 1
                            else 0.0)
            d["th"].append(np.mean([r["Tc_th"] for r in rs]))
            d["cal"].append(cal.mean())
            d["n"].append(len(rs))
        out[rgm] = {k: np.array(v, float) for k, v in d.items()}
    return out


def slope(x: np.ndarray, y: np.ndarray) -> float:
    """log-log 最小二乘斜率。"""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def max_dev(d: dict, mask=None) -> float:
    """|T_c^th/T_c^sim - 1| 的最大值（mask 可限定 x 的子区间）。"""
    r = np.abs(d["th"] / d["sim"] - 1.0)
    return float(r.max() if mask is None else r[mask].max())


def _series(ax, D, logy=True, logx=False, show_calib=True):
    """两支的 sim（点+误差棒）与 th（线）。"""
    style = {"case_I": (BLUE, "o", "Case I sim"), "case_II": (VERM, "s", "Case II sim")}
    for rgm, (col, mk, lab) in style.items():
        if rgm not in D:
            continue
        d = D[rgm]
        ax.errorbar(d["x"], d["sim"], yerr=d["sem"], fmt=mk, ms=4.0, mfc="white",
                    mec=col, mew=1.0, ecolor=col, elinewidth=0.7, capsize=1.6,
                    ls="none", zorder=3, label=lab)
        ax.plot(d["x"], d["th"], color=col, lw=1.0, ls="--", zorder=2,
                label=lab.replace("sim", "theory"))
        if show_calib and rgm == "case_II":
            ax.plot(d["x"], d["cal"], color=GREEN, lw=1.0, ls="-", zorder=2,
                    label=r"Case II $\lambda(N)t^1_{\rm L}$")
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="param_sweep_m2_kmin3.csv")
    args = ap.parse_args()

    set_style()
    rows = load(args.csv)
    A = {s: aggregate(rows, s) for s in ("C", "m", "k0", "N", "b")}

    fig, axes = plt.subplots(2, 3, figsize=(COL2, COL2 * 0.62))
    (axC, axk, axm), (axN, axb, axr) = axes

    # (a) C/C*：regime 交界
    dC = A["C"]
    for rgm, col, mk in (("case_II", VERM, "s"), ("case_I", BLUE, "o")):
        if rgm not in dC:
            continue
        d = dC[rgm]
        axC.errorbar(d["x"], d["sim"] / 1e6, yerr=d["sem"] / 1e6, fmt=mk, ms=3.6,
                     mfc="white", mec=col, mew=1.0, ecolor=col, elinewidth=0.7,
                     capsize=1.5, ls="-", color=col, lw=1.0, zorder=3)
        axC.plot(d["x"], d["th"] / 1e6, color=col, lw=1.0, ls="--", zorder=2)
    axC.axvline(1.0, color=MUTED, lw=0.7, ls=":", zorder=1)
    axC.text(1.06, 0.06, r"$C=C^{*}$", transform=axC.get_xaxis_transform(),
             fontsize=6.5, color=MUTED, rotation=90, va="bottom")
    axC.set_xlim(0, 3.1)
    axC.set_xlabel(r"capacity  $C/C^{*}$")
    axC.set_ylabel(r"$T_c$  ($10^{6}$ steps)")
    axC.set_title("(a) capacity", fontsize=8, pad=3)
    axC.plot([], [], "s-", color=VERM, mfc="white", ms=3.6, label="Case II sim")
    axC.plot([], [], "o-", color=BLUE, mfc="white", ms=3.6, label="Case I sim")
    axC.plot([], [], ls="--", color=MUTED, lw=1.0, label="theory (dashed)")
    axC.legend(loc="upper left", borderaxespad=0.3)

    # (b) <k>_0
    _series(axk, A["k0"], logy=False, show_calib=False)
    axk.set_xlabel(r"mean hyperdegree  $\langle k\rangle_0$")
    axk.set_ylabel(r"$T_c$  (steps)")
    axk.set_yscale("log")
    axk.set_title(r"(b) hyperdegree", fontsize=8, pad=3)
    axk.legend(loc="lower right", borderaxespad=0.3, fontsize=6)

    # (c) m：附 N^m 参考斜率
    _series(axm, A["m"], show_calib=False)
    d = A["m"]["case_I"]
    axm.set_xlabel(r"hyperedge order  $m$")
    axm.set_ylabel(r"$T_c$  (steps)")
    axm.set_xticks(d["x"])
    axm.set_title(r"(c) order  ($T_c\sim N^{m}$)", fontsize=8, pad=3)
    axm.legend(loc="upper left", borderaxespad=0.3, fontsize=6)

    # (d) N：拟合标度指数
    _series(axN, A["N"], logx=True, show_calib=False)
    txt = []
    for rgm, lab in (("case_I", "I"), ("case_II", "II")):
        d = A["N"][rgm]
        txt.append(rf"Case {lab}: $T_c\propto N^{{{slope(d['x'], d['sim']):.2f}}}$")
    axN.set_xlabel(r"system size  $N$")
    axN.set_ylabel(r"$T_c$  (steps)")
    axN.set_title("(d) size", fontsize=8, pad=3)
    axN.text(0.04, 0.96, "\n".join(txt), transform=axN.transAxes, fontsize=6.2,
             va="top", ha="left", color=INK)
    axN.legend(loc="lower right", borderaxespad=0.3, fontsize=6)

    # (e) <b>：1/<b> 参考线
    _series(axb, A["b"], logx=True, show_calib=False)
    d = A["b"]["case_I"]
    grid = np.array([d["x"].min(), d["x"].max()])
    axb.plot(grid, d["sim"][-1] * d["x"][-1] / grid, color=MUTED, lw=0.8, ls=":",
             zorder=1, label=r"$\propto 1/\langle b\rangle$")
    axb.set_xlabel(r"mean vulnerability  $\langle b\rangle$")
    axb.set_ylabel(r"$T_c$  (steps)")
    axb.set_xticks([0.1, 0.2, 0.3, 0.5])
    axb.set_xticklabels(["0.1", "0.2", "0.3", "0.5"])
    axb.minorticks_off()
    axb.set_title("(e) vulnerability", fontsize=8, pad=3)
    axb.legend(loc="upper right", borderaxespad=0.3, fontsize=6)

    # (f) 精度 vs C/C*：容量是唯一让理论精度显著变化的参数（其余四个见文中表）
    axr.axhline(1.0, color=MUTED, lw=0.7, zorder=1)
    axr.axhspan(0.9, 1.1, color=MUTED, alpha=0.12, lw=0, zorder=0)
    axr.axvline(1.0, color=MUTED, lw=0.7, ls=":", zorder=1)
    for rgm, col, mk, lab in (("case_II", VERM, "s", r"Case II  $t^1_{\rm L}$"),
                              ("case_I", BLUE, "o", r"Case I  (zero-parameter)")):
        d = dC[rgm]
        axr.plot(d["x"], d["th"] / d["sim"], mk + "-", ms=3.6, mfc="white", mec=col,
                 mew=1.0, color=col, lw=1.0, zorder=3, label=lab)
    d = dC["case_II"]
    axr.plot(d["x"], d["cal"] / d["sim"], "^-", ms=3.6, mfc="white", mec=GREEN,
             mew=1.0, color=GREEN, lw=1.0, zorder=3,
             label=r"Case II  $\lambda(N)\,t^1_{\rm L}$")
    axr.set_xlim(0, 3.1)
    axr.set_ylim(0.25, 1.45)
    axr.set_xlabel(r"capacity  $C/C^{*}$")
    axr.set_ylabel(r"$T_c^{\rm th}/T_c^{\rm sim}$")
    axr.set_title(r"(f) accuracy vs capacity", fontsize=8, pad=3)
    axr.legend(loc="lower left", borderaxespad=0.3, fontsize=6)

    # 其余四个 sweep 的 Case I 精度：这是**整张图**的注记（覆盖 (b)-(e)），不属于
    # 任何单个面板，故放在图脚而不是塞进 (f)——塞进 (f) 会压住它自己的图例。
    # ⟨k⟩₀ 的最大偏差全部来自最小的一点（k_min=1 的退化区），只报总最大值会掩盖
    # "⟨k⟩₀≳4 后与 ⟨k⟩₀ 无关"这个结论，故两个数都给。
    dk = A["k0"]["case_I"]
    note = (r"Case I  $\max|T_c^{\rm th}/T_c^{\rm sim}-1|$ in panels (b)–(e):   "
            rf"$\langle k\rangle_0\!\geq\!4$: {max_dev(dk, dk['x'] >= 4):.0%} "
            rf"({max_dev(dk, dk['x'] < 4):.0%} at the degenerate point "
            rf"$\langle k\rangle_0\!=\!{dk['x'].min():.1f}$),   "
            rf"$m$: {max_dev(A['m']['case_I']):.0%},   "
            rf"$N$: {max_dev(A['N']['case_I']):.0%},   "
            rf"$\langle b\rangle$: {max_dev(A['b']['case_I']):.0%}")
    fig.text(0.5, -0.012, note, ha="center", va="top", fontsize=6.2, color=MUTED)

    fig.subplots_adjust(wspace=0.34, hspace=0.42)
    save(fig, "fig5_parameter_dependence")


if __name__ == "__main__":
    main()
