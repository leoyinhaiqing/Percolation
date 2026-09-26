"""生成论文图组（REVTeX 单/双栏尺寸，矢量 PDF + 预览 PNG）。

四张图：
  fig1_Tc_validation   T_c^sim vs T_c^th 散点 + 对角线（两个 regime，六个 N）
  fig2_relative_error  (a) 相对误差 vs N（±10% 参考带）(b) Case I 误差的三段可加分解
  fig3_lambda_vs_N     Case II 的 lambda vs N（观测 / 上界 / 校准曲线）
  fig4_trajectories    S_h(t) 轨迹：仿真 vs 平均场 vs Laplace 修正（ductile / brittle 对照）

前三张只读 results/ 下的 tail_diag CSV，不跑仿真；fig4 需要时间序列，故用
`dynamics.simulate` 现跑两条轨迹（N 小，成本低）。

配色用 Okabe-Ito 的三色子集（#0072B2 / #D55E00 / #009E73），色觉障碍安全
（validate_palette.js 六项全 PASS）；同一实体在所有图里保持同一颜色。

用法：
  .venv/Scripts/python.exe experiments/make_figures.py
  .venv/Scripts/python.exe experiments/make_figures.py --traj_N 2000 --traj_seed 1
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_tail_diag import group_by_config, load_rows, sem  # noqa: E402
from src import theory  # noqa: E402
from src.config import Config  # noqa: E402
from src.dynamics import simulate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")

# REVTeX：单栏 3.375 in，双栏 6.9 in
COL1, COL2 = 3.375, 6.9

# 固定实体配色（Okabe-Ito 子集，CVD-safe）
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED = "#1a1a1a", "#666666"


def n_label(n: float) -> str:
    """N 的紧凑刻度标签：<10^4 用整数，更大用 a×10^b。"""
    n = int(round(n))
    if n < 10_000:
        return str(n)
    e = int(math.floor(math.log10(n)))
    a = n / 10 ** e
    mant = "" if abs(a - 1.0) < 1e-9 else rf"{a:g}\times"
    return rf"${mant}10^{{{e}}}$"


def set_style() -> None:
    """期刊风格：衬线字体、四边内向刻度、无网格、细线条。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, name: str) -> None:
    """同名存 PDF（入正文）与 PNG（预览）。"""
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved: figures/{name}.pdf + .png")


# --------------------------------------------------------------------------
# 数据装配
# --------------------------------------------------------------------------

def collect(csvs: list[str], lam_a: float, lam_b: float,
            max_N: float | None = None) -> dict:
    """把两个 regime 的逐 seed 量整理成按 N 排序的数组。"""
    rows = load_rows(csvs)
    if max_N is not None:
        rows = [r for r in rows if r["N"] <= max_N]
    groups = group_by_config(rows)
    out: dict = {}
    for regime in ("case_I", "case_II"):
        keys = sorted(k for k in groups if k[1] == regime)
        Ns, sim_m, sim_e, th_m, err_m, err_e, lam_m, lam_e = ([] for _ in range(8))
        for k in keys:
            rs = groups[k]
            sim = np.array([r["t_rhoc_sim"] for r in rs])
            t1L = np.array([r["t1_laplace_ih"] for r in rs])
            th = t1L + np.array([r["casc_mf"] for r in rs]) if regime == "case_I" else t1L
            err = th / sim - 1.0
            lam = sim / t1L
            Ns.append(k[0])
            sim_m.append(sim.mean()); sim_e.append(sem(sim))
            th_m.append(th.mean())
            err_m.append(err.mean()); err_e.append(sem(err))
            lam_m.append(lam.mean()); lam_e.append(sem(lam))
        out[regime] = dict(
            N=np.array(Ns, float), sim=np.array(sim_m), sim_sem=np.array(sim_e),
            th=np.array(th_m), err=np.array(err_m), err_sem=np.array(err_e),
            lam=np.array(lam_m), lam_sem=np.array(lam_e),
        )
    out["case_I"].update(error_decomposition(groups))
    d = out["case_II"]
    # 饱和式 lambda(N)=a+b/ln N 在全部 8 个 N 上一次拟合（findings ㉕），故无需
    # 再区分"标定域内/域外"——整条曲线都是同一次拟合的结果。
    d["lam_pred"] = np.array([theory.lambda_calibration(n, a=lam_a, b=lam_b)
                              for n in d["N"]])
    # 校准后的 Case II 预测：T_c = lambda(N) * t^1_L
    d["th_calib"] = d["th"] * d["lam_pred"]
    d["err_calib"] = d["th_calib"] / d["sim"] - 1.0
    return out


def error_decomposition(groups: dict) -> dict:
    """把 Case I 的相对误差拆成三个**可加**的实测分量（按 N 排序）。

      err = (t^1_L + Δt^E_MF - T_c^sim)/T_c^sim
          = (t^1_L - t^1_sim)/T_c^sim            <- trigger（Laplace 触发时刻残差）
          + (Δt^E_MF - Δt^E_sim)/T_c^sim         <- cascade（平均场级联段偏长）
          - (T_c^sim - t^E_sim)/T_c^sim          <- tail（公式把尾段置零而实测非零）

    三项之和恒等于总误差，故可直接读出"哪一项在把误差往哪边推"。
    """
    keys = sorted(k for k in groups if k[1] == "case_I")
    trig, casc, tail = [], [], []
    for k in keys:
        rs = groups[k]
        sim = np.array([r["t_rhoc_sim"] for r in rs])
        t1L = np.array([r["t1_laplace_ih"] for r in rs])
        t1s = np.array([r["t_sh1_sim"] for r in rs])
        tEs = np.array([r["t_shE_sim"] for r in rs])
        cmf = np.array([r["casc_mf"] for r in rs])
        trig.append(((t1L - t1s) / sim).mean())
        casc.append(((cmf - (tEs - t1s)) / sim).mean())
        tail.append((-(sim - tEs) / sim).mean())
    return dict(dec_trigger=np.array(trig), dec_cascade=np.array(casc),
                dec_tail=np.array(tail))


# --------------------------------------------------------------------------
# Fig. 1 — T_c^sim vs T_c^th
# --------------------------------------------------------------------------

def fig_validation(D: dict) -> None:
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.86))
    I, II = D["case_I"], D["case_II"]

    lo = 0.6 * min(I["sim"].min(), II["sim"].min())
    hi = 2.1 * max(I["th"].max(), II["th"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color=MUTED, zorder=1,
            label=r"$T_c^{\rm sim}=T_c^{\rm th}$")

    ax.errorbar(I["th"], I["sim"], yerr=I["sim_sem"], fmt="o", ms=4.5, mfc="white",
                mec=BLUE, mew=1.1, ecolor=BLUE, elinewidth=0.8, capsize=1.8,
                ls="none", zorder=3, label=r"Case I: $t^1_{\rm L}+\Delta t^E_{\rm MF}$")
    ax.errorbar(II["th"], II["sim"], yerr=II["sim_sem"], fmt="s", ms=4.2, mfc="white",
                mec=VERM, mew=1.1, ecolor=VERM, elinewidth=0.8, capsize=1.8,
                ls="none", zorder=3, label=r"Case II: $t^1_{\rm L}$ (bound)")
    ax.errorbar(II["th_calib"], II["sim"], yerr=II["sim_sem"], fmt="^", ms=4.5,
                mfc="white", mec=GREEN, mew=1.1, ecolor=GREEN, elinewidth=0.8,
                capsize=1.8, ls="none", zorder=3,
                label=r"Case II: $\lambda(N)\,t^1_{\rm L}$")

    # 只锚定最小的 N（右上角没有放标签的余量）；N 递增方向写进图注
    ax.annotate(rf"$N={int(I['N'][0])}$", (I["th"][0], I["sim"][0]),
                textcoords="offset points", xytext=(11, 2), fontsize=6.5,
                color=MUTED, ha="left", va="bottom")
    ax.annotate("", xy=(0.60, 0.42), xytext=(0.48, 0.30), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.6))
    ax.text(0.62, 0.40, r"increasing $N$", transform=ax.transAxes, fontsize=6.5,
            color=MUTED, ha="left", va="top")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"theory  $T_c^{\rm th}$")
    ax.set_ylabel(r"simulation  $T_c^{\rm sim}$")
    ax.legend(loc="upper left", borderaxespad=0.4)
    save(fig, "fig1_Tc_validation")


# --------------------------------------------------------------------------
# Fig. 2 — 相对误差 vs N
# --------------------------------------------------------------------------

def fig_error(D: dict, band: float = 10.0) -> None:
    """(a) 相对误差 vs N（±band% 参考带）；(b) Case I 误差的三段可加分解。

    右panel 是左panel 非单调性的机制解释：级联段（平均场偏长）把误差往上推、
    尾段（公式置零而实测非零）往下拉，两者随 N 反向增长并交叉——所以 Case I 的
    误差有界但不单调，不能按任一端外推（findings ⑱）。
    """
    # 竖排单栏：两个 panel 上下叠放，图进正文时用 \columnwidth 的 figure
    # （跨栏 figure* 在 revtex 双栏下只能浮到页顶，会被推进参考文献里）。
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(COL1, COL1 * 1.62), sharex=True)
    I, II = D["case_I"], D["case_II"]

    ax.axhspan(-band, band, color=MUTED, alpha=0.12, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.6, ls="-", zorder=1)
    ax.text(I["N"][-1] * 1.06, band, rf"$\pm{band:g}\%$", fontsize=6.5,
            color=MUTED, va="center", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

    ax.errorbar(I["N"], 100 * I["err"], yerr=100 * I["err_sem"], fmt="o-", ms=4.5,
                mfc="white", mec=BLUE, mew=1.1, color=BLUE, ecolor=BLUE,
                elinewidth=0.8, capsize=1.8, zorder=3, label="Case I (zero-parameter)")
    ax.errorbar(II["N"], 100 * II["err"], yerr=100 * II["err_sem"], fmt="s-", ms=4.2,
                mfc="white", mec=VERM, mew=1.1, color=VERM, ecolor=VERM,
                elinewidth=0.8, capsize=1.8, zorder=3, label="Case II (upper bound)")
    ax.errorbar(II["N"], 100 * II["err_calib"], yerr=100 * II["err_sem"],
                fmt="^-", ms=4.5, mfc="white", mec=GREEN, mew=1.1, color=GREEN,
                ecolor=GREEN, elinewidth=0.8, capsize=1.8, zorder=3,
                label=r"Case II (calibrated $\lambda(N)$)")

    ax.set_xscale("log")
    ax.set_ylabel("relative error\n"
                  r"$(T_c^{\rm th}-T_c^{\rm sim})/T_c^{\rm sim}$  (%)")
    ax.minorticks_off()
    ax.set_xlim(I["N"][0] * 0.80, I["N"][-1] * 1.40)   # 右侧留出参考带标签的位置
    ax.legend(loc="lower left", borderaxespad=0.4, fontsize=6.2)
    ax.set_title("(a) theory vs simulation", fontsize=8, pad=4)

    axd.axhline(0, color=MUTED, lw=0.6, zorder=1)
    axd.plot(I["N"], 100 * I["dec_cascade"], "s-", ms=4.0, mfc="white", mec=VERM,
             mew=1.1, color=VERM, zorder=3,
             label=r"cascade  $(\Delta t^E_{\rm MF}-\Delta t^E_{\rm sim})/T_c^{\rm sim}$")
    axd.plot(I["N"], 100 * I["dec_tail"], "v-", ms=4.0, mfc="white", mec=GREEN,
             mew=1.1, color=GREEN, zorder=3,
             label=r"tail  $-(T_c^{\rm sim}-t^E_{\rm sim})/T_c^{\rm sim}$")
    axd.plot(I["N"], 100 * I["dec_trigger"], "d-", ms=3.8, mfc="white", mec=MUTED,
             mew=1.0, color=MUTED, zorder=2,
             label=r"trigger  $(t^1_{\rm L}-t^1_{\rm sim})/T_c^{\rm sim}$")
    axd.plot(I["N"], 100 * I["err"], "o-", ms=4.5, mfc="white", mec=BLUE, mew=1.1,
             color=BLUE, zorder=4, label="sum = Case I error")

    axd.set_xscale("log")
    axd.set_xlabel(r"system size  $N$")
    axd.set_ylabel("contribution to\nrelative error  (%)")
    axd.set_xticks(I["N"])
    axd.set_xticklabels([n_label(n) for n in I["N"]])
    axd.minorticks_off()
    axd.legend(loc="lower left", borderaxespad=0.4, fontsize=6.2)
    axd.set_title("(b) decomposition of the Case I error", fontsize=8, pad=4)

    fig.subplots_adjust(hspace=0.16)
    save(fig, "fig2_relative_error")


# --------------------------------------------------------------------------
# Fig. 3 — lambda vs N
# --------------------------------------------------------------------------

def fig_lambda(D: dict, lam_a: float, lam_b: float) -> None:
    """lambda 观测 vs 饱和式拟合 lambda(N)=a+b/ln N（findings ㉕）。

    拟合在全部 8 个 N 上一次完成，故整条曲线是同一条实线——不再有"标定域内实线 /
    域外虚线"的分割（那是被弃用的对数式才需要的限定）。
    """
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.78))
    II = D["case_II"]

    ax.axhline(1.0, color=VERM, lw=1.0, ls="--", zorder=1)
    ax.text(II["N"][-1], 1.006, r"parameter-free bound  $\lambda=1$", fontsize=6.5,
            color=VERM, ha="right", va="bottom")

    grid = np.logspace(math.log10(II["N"][0] * 0.85), math.log10(II["N"][-1] * 1.18), 200)
    lam_grid = np.array([theory.lambda_calibration(n, a=lam_a, b=lam_b) for n in grid])
    ax.plot(grid, lam_grid, color=GREEN, lw=1.1, zorder=2,
            label=rf"fit  $\lambda(N)={lam_a:g}+{lam_b:g}/\ln N$")
    ax.errorbar(II["N"], II["lam"], yerr=II["lam_sem"], fmt="o", ms=4.5, mfc="white",
                mec=BLUE, mew=1.1, ecolor=BLUE, elinewidth=0.8, capsize=1.8,
                ls="none", zorder=3, label=r"simulation (15 seeds, mean $\pm$ s.e.m.)")

    # 拟合质量就地复算（sem 加权），不写死数字：dof = 点数 - 2 个系数
    chi2 = float((((II["lam"] - II["lam_pred"]) / II["lam_sem"]) ** 2).sum())
    dof = max(1, len(II["N"]) - 2)
    ax.text(0.97, 0.80, f"{len(II['N'])} sizes,  max dev "
            f"{np.abs(II['lam_pred'] / II['lam'] - 1).max():.1%}\n"
            rf"$\chi^2/{{\rm dof}}={chi2 / dof:.1f}$",
            transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="right", va="top")

    ax.set_xscale("log")
    ax.set_xlabel(r"system size  $N$")
    ax.set_ylabel(r"$\lambda \equiv T_c^{\rm sim}/t^1_{\rm L}$")
    ax.set_xticks(II["N"])
    ax.set_xticklabels([n_label(n) for n in II["N"]])
    ax.minorticks_off()
    ax.set_ylim(min(0.70, float(min(II["lam"].min(), lam_grid.min())) - 0.03), 1.045)
    ax.legend(loc="lower left", borderaxespad=0.4)
    save(fig, "fig3_lambda_vs_N")


# --------------------------------------------------------------------------
# Fig. 4 — S_h(t) 轨迹（ductile vs brittle）
# --------------------------------------------------------------------------

def trajectory(N: int, m: int, k_min: int, frac: float, seed: int) -> dict:
    """跑一条轨迹并返回绘图所需的量（C = frac * C*）。"""
    from src.hypergraph import build_incidence, generate_powerlaw_hypergraph, hyperdegree_moments
    rng = np.random.default_rng(1)
    probe, _ = generate_powerlaw_hypergraph(N, m, gamma=2.5, k_min=k_min, rng=rng)
    k0p, _ = hyperdegree_moments(build_incidence(probe, N))
    C = round(frac * theory.C_threshold(k0p, m), 4)
    res = simulate(Config(N=N, m=m, k_min=k_min, C=C, seed=seed))
    sh1 = theory.S_h_1star(res.k0, m, C)
    return dict(res=res, C=C, sh1=sh1, shE=theory.S_h_E(res.k0), rho_c=res.rho_c,
                p=res.p, c=math.factorial(m) / N ** m,
                t1L=theory.t_1_laplace(N, m, sh1))


def fig_trajectories(traj_N: int, traj_seed: int, k_min: int, m: int) -> None:
    """主图看整条衰减，插图放大坍塌窗口——ductile 的阶梯 vs brittle 的一步穿透。"""
    from matplotlib.ticker import FixedLocator, NullFormatter

    panels = [(1.5, r"(a) Case I  (ductile,  $C=1.5\,C^{*}$)"),
              (0.75, r"(b) Case II  (brittle,  $C=0.75\,C^{*}$)")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.42), sharey=True)

    for ax, (frac, title) in zip(axes, panels):
        d = trajectory(traj_N, m, k_min, frac, traj_seed)
        res, t1L = d["res"], d["t1L"]
        x = res.ts / t1L
        grid = np.linspace(1e-6, max(x.max(), 1.05), 400) * t1L
        ct = d["c"] * grid

        ax.plot(grid / t1L, np.exp(-res.p * grid), color=VERM, lw=1.1, ls="--",
                zorder=2, label=r"mean field  $e^{-pt}$")
        ax.plot(grid / t1L, ((1 - np.exp(-ct)) / ct) ** (m + 1), color=GREEN, lw=1.1,
                zorder=2, label=r"Laplace  $[(1-e^{-ct})/ct]^{m+1}$")
        ax.plot(x, res.Sh, color=BLUE, lw=1.3, zorder=3, label=r"simulation  $S_h(t)$")

        for y, lab in ((d["sh1"], r"$S_h^{1}$"), (d["shE"], r"$S_h^{E}$"),
                       (d["rho_c"], r"$\rho_c$")):
            ax.axhline(y, color=MUTED, lw=0.55, ls=":", zorder=1)
            ax.text(1.008, y, lab, transform=ax.get_yaxis_transform(), fontsize=6.5,
                    color=MUTED, va="center", ha="left")

        ax.set_yscale("log")
        ax.set_xlim(0, max(1.06, float(x.max()) * 1.02))
        ax.set_ylim(0.6 * d["rho_c"], 1.5)
        ax.yaxis.set_major_locator(FixedLocator([0.03, 0.05, 0.1, 0.2, 0.5, 1.0]))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$t/t^1_{\rm L}$")
        ax.set_title(title, fontsize=8, pad=4)

        # 插图：坍塌窗口放大（从触发前一点到 rho_c 穿越）
        i_trig = int(np.argmax(res.Sh <= d["sh1"]))
        x0 = x[i_trig] - 0.10
        axi = ax.inset_axes([0.09, 0.13, 0.44, 0.46])
        axi.plot(x, res.Sh, color=BLUE, lw=1.0, zorder=3)
        for y in (d["sh1"], d["shE"], d["rho_c"]):
            axi.axhline(y, color=MUTED, lw=0.5, ls=":", zorder=1)
        axi.set_xlim(x0, float(x.max()) * 1.005)
        axi.set_ylim(0.7 * d["rho_c"], 1.6 * d["sh1"])
        axi.set_yscale("log")
        axi.yaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.2]))
        axi.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
        axi.yaxis.set_minor_formatter(NullFormatter())
        axi.tick_params(labelsize=6, pad=1.5)
        axi.set_xticks([round(x0 + 0.02, 2), round(x0 + 0.08, 2)])
        for s in axi.spines.values():
            s.set_linewidth(0.5)
        # 只保留标示放大区域的方框；连接线会横穿整幅图，去掉
        indicator = ax.indicate_inset_zoom(axi, edgecolor=MUTED, lw=0.5, alpha=0.6)
        for ln in indicator.connectors:
            ln.set_visible(False)

    axes[0].set_ylabel(r"hyperedge density  $S_h$")
    axes[0].legend(loc="upper right", borderaxespad=0.5)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig4_trajectories")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+",
                    default=["tail_diag_m2_kmin3.csv",
                             "tail_diag_m2_kmin3_N20000.csv",
                             "tail_diag_m2_kmin3_N50000_quenched.csv",
                             "tail_diag_m2_kmin3_N100000_quenched.csv"])
    ap.add_argument("--band", type=float, default=10.0,
                    help="fig2 参考带半宽（%%）")
    ap.add_argument("--max_N", type=float, default=None,
                    help="只画到该 N（例如 50000：Case I 全部落在 ±10% 内）")
    ap.add_argument("--lam_a", type=float, default=0.583,
                    help="lambda(N)=a+b/ln N 的 a（论文采用值）")
    ap.add_argument("--lam_b", type=float, default=1.808,
                    help="lambda(N)=a+b/ln N 的 b（论文采用值）")
    ap.add_argument("--traj_N", type=int, default=2000)
    ap.add_argument("--traj_seed", type=int, default=1)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--m", type=int, default=2)
    args = ap.parse_args()

    set_style()
    D = collect(args.csv, args.lam_a, args.lam_b, args.max_N)
    fig_validation(D)
    fig_error(D, band=args.band)
    fig_lambda(D, args.lam_a, args.lam_b)
    fig_trajectories(args.traj_N, args.traj_seed, args.k_min, args.m)


if __name__ == "__main__":
    main()
