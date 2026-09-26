"""关键节点识别图组（fig6）：动态 ΔT_c 排序 vs 三条静态中心性。

读 results/vital_<tag>_curves.csv 与 vital_<tag>_nodes.csv
（experiments/run_vital_nodes.py 产出）。两行对应两个容量 regime，三列：
  (左) 静态鲁棒性：移除 top-k 节点（连带其超边）后的巨分量相对大小
  (中) 动态效果：把 top-k 节点的 b 提到 1 后的 T_c 相对下降 —— 与本文口径一致
  (右) ΔT_c vs 超度散点：动态排序在多大程度上"不只是度"

配色在论文三色基础上加一个 Okabe-Ito 的紫红给 PageRank，四条曲线仍 CVD-safe。

用法：
  .venv/Scripts/python.exe experiments/make_vital_figures.py --tag synthetic_N1000_m2
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

PURPLE, GOLD, SKY = "#CC79A7", "#E69F00", "#56B4E9"
STYLE = {
    "dTc":             (BLUE,   "o", "-",  r"$\Delta T_c$"),
    "degree_x_vuln":   (SKY,    "*", "-",  r"$k_i(1-b_i)$"),
    "hyperdegree":     (VERM,   "s", "--", "hyperdegree"),
    "betweenness":     (GREEN,  "^", "-.", "betweenness"),
    "pagerank":        (PURPLE, "v", ":",  "PageRank"),
    "dTc_greedy":      (INK,    "D", "-",  r"$\Delta T_c$ greedy (set-aware)"),
    "coverage_greedy": (GOLD,   "P", "--", "coverage greedy (static control)"),
}
REGIME_TITLE = {"case_I": r"Case I  (ductile, $C=1.5\,C^{*}$)",
                "case_II": r"Case II  (brittle, $C=0.75\,C^{*}$)"}


def load(path: str, floats: tuple) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            r = dict(raw)
            for c in floats:
                if c in r and r[c] not in (None, ""):
                    r[c] = float(r[c])
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="synthetic_N1000_m2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    curves = load(os.path.join(RESULTS, f"vital_{args.tag}_curves.csv"),
                  ("C_frac", "k", "gc", "gc_rel", "Tc", "Tc_rel"))
    nodes = load(os.path.join(RESULTS, f"vital_{args.tag}_nodes.csv"),
                 ("C_frac", "dTc", "hyperdegree", "betweenness", "pagerank",
                  "n_clocks"))
    N = len({r["node"] for r in nodes})

    regimes = []
    for r in curves:
        if r["regime"] not in regimes:
            regimes.append(r["regime"])
    regimes.sort(key=lambda s: 0 if s == "case_I" else 1)

    set_style()
    nr = len(regimes)
    fig, axes = plt.subplots(nr, 3, figsize=(COL2, COL2 * 0.34 * nr),
                             squeeze=False)

    for row, rgm in enumerate(regimes):
        axg, axt, axs = axes[row]
        sel = [r for r in curves if r["regime"] == rgm]
        by = defaultdict(list)
        for r in sel:
            by[r["method"]].append(r)

        for name, (col, mk, ls, lab) in STYLE.items():
            if not by.get(name):
                continue
            rs = sorted(by[name], key=lambda r: r["k"])
            kk = np.array([r["k"] for r in rs]) / N * 100
            lw = 1.4 if name in ("dTc", "dTc_greedy") else 1.0
            axg.plot(kk, [r["gc_rel"] for r in rs], mk, ls=ls, ms=3.4, mfc="white",
                     mec=col, mew=1.0, color=col, lw=lw, label=lab)
            axt.plot(kk, [r["Tc_rel"] for r in rs], mk, ls=ls, ms=3.4, mfc="white",
                     mec=col, mew=1.0, color=col, lw=lw, label=lab)

        # regime 不进 panel 标题（太长会压到相邻 panel 的 y 轴标签上），
        # 改成整行上方的一条行标题，panel 标题只留字母 + 口径。
        pa, pb, pc = (chr(ord("a") + 3 * row + j) for j in range(3))

        axg.set_xlabel(r"removed fraction  $k/N$  (%)")
        axg.set_ylabel(r"relative giant component  $S(k)/S(0)$")
        axg.set_title(f"({pa}) structural", fontsize=8, pad=3)
        axg.legend(loc="lower left", borderaxespad=0.3, fontsize=6)

        axt.axhline(1.0, color=MUTED, lw=0.6, zorder=1)
        axt.set_xlabel(r"perturbed fraction  $k/N$  (%)")
        axt.set_ylabel(r"relative collapse time  $T_c(k)/T_c(0)$")
        axt.set_title(f"({pb}) dynamic ($b_i\\to1$ for top-$k$)", fontsize=8, pad=3)
        axt.legend(loc="lower left", borderaxespad=0.3, fontsize=6)

        nd = [r for r in nodes if r["regime"] == rgm]
        k_i = np.array([r["hyperdegree"] for r in nd])
        d = np.array([r["dTc"] for r in nd])
        nc = np.array([r.get("n_clocks", 0) for r in nd], float)
        base = float(np.nanmax(np.abs(d))) or 1.0
        # 高精度（第二段多时钟）与粗估（第一段少时钟）分开画：两者噪声水平差一个量级。
        # 旧 CSV 没有 n_clocks 列时退化为单一散点（不假装区分不存在的精度信息）。
        has_nc = np.nanmax(nc) > 0 and len(np.unique(nc[nc > 0])) > 1
        fine = (nc == np.nanmax(nc)) if has_nc else np.ones_like(nc, bool)
        if has_nc:
            axs.plot(np.maximum(k_i[~fine], 0.5), d[~fine] / base, "o", ms=2.2,
                     mfc="none", mec=MUTED, mew=0.5, alpha=0.35,
                     label=f"screening ({int(np.nanmin(nc[nc > 0]))} clocks)")
        axs.plot(np.maximum(k_i[fine], 0.5), d[fine] / base, "o", ms=3.0 if has_nc
                 else 2.4, mfc="none", mec=BLUE, mew=0.9 if has_nc else 0.6,
                 alpha=1.0 if has_nc else 0.55,
                 label=(f"shortlist ({int(np.nanmax(nc))} clocks)") if has_nc
                 else "all nodes")
        pos = fine & (d > 0)
        if pos.sum() > 3:
            cf = np.polyfit(np.log(np.maximum(k_i[pos], 1)), np.log(d[pos] / base), 1)
            xg = np.array([max(1.0, k_i[fine].min()), k_i.max()])
            axs.plot(xg, np.exp(cf[1]) * xg ** cf[0], color=INK, lw=0.9, ls="--",
                     label=rf"$\Delta T_c\propto k^{{{cf[0]:.2f}}}$")
        axs.legend(loc="lower left", borderaxespad=0.3, fontsize=5.6,
                   frameon=True, framealpha=0.85, edgecolor="none")
        axs.set_xscale("log")
        axs.set_yscale("symlog", linthresh=1e-3)
        axs.set_xlabel(r"hyperdegree  $k_i$")
        axs.set_ylabel(r"$\Delta T_c$  (normalised)")
        axs.set_title(f"({pc}) score vs degree", fontsize=8, pad=3)

    fig.subplots_adjust(wspace=0.34, hspace=0.62, top=0.90)
    for row, rgm in enumerate(regimes):        # 每行上方一条行标题
        bb0, bb2 = axes[row][0].get_position(), axes[row][2].get_position()
        fig.text(0.5 * (bb0.x0 + bb2.x1), bb0.y1 + 0.055, REGIME_TITLE.get(rgm, rgm),
                 ha="center", va="bottom", fontsize=9.5, color=INK)
    save(fig, args.out or f"fig6_vital_nodes_{args.tag}")


if __name__ == "__main__":
    main()
