"""从 tail_diag CSV 复算论文两分支的检验表（λ 与 Case I 误差）。

此前 v13/v14 备忘里的数字是临时算出来的，没有留下可复现的脚本。本脚本把那套口径
固化下来，任何一份 `run_tail_diag.py` 产出的 CSV 都能直接复算，也能一次读多份
（新增大 N 时不必重跑旧 N）。

口径（与 papers/main.tex 一致）：
  - 仿真坍塌时刻取 `t_rhoc_sim`（S_h 首次跌破 rho_c），不是巨分量口径 T_c^sim；
  - Case II：T_c^th = t^1_L（零参数单边上界），lambda = t_rhoc_sim / t^1_L；
    校准式 lambda(N) = a + b/ln N（饱和式，8 点加权 LSQ）；
  - Case I：T_c^th = t^1_L + (t^E - t^1)_MF（尾段置零），误差 = (th - sim)/sim。

用法：
  .venv/Scripts/python.exe experiments/analyze_tail_diag.py
  .venv/Scripts/python.exe experiments/analyze_tail_diag.py --csv tail_diag_m2_kmin3.csv tail_diag_m2_kmin3_N20000.csv
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

FLOAT_COLS = ("C", "k0", "k2", "b_mean", "t_sh1_sim", "t_shE_sim", "t_rhoc_sim",
              "t1_mf", "t1_laplace_ih", "t1_laplace_emp", "casc_mf", "tail_mf")


def load_rows(names: list[str]) -> list[dict]:
    """读入若干 CSV，数值列转 float / int；t_*_sim 可能为空（未穿越）→ nan。"""
    rows = []
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(RESULTS, name)
        with open(path, newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                r = dict(raw)
                r["N"] = int(r["N"])
                r["seed"] = int(r["seed"])
                for c in FLOAT_COLS:
                    r[c] = float(r[c]) if r.get(c) not in (None, "") else math.nan
                rows.append(r)
    return rows


def group_by_config(rows: list[dict]) -> dict:
    """按 (N, regime) 聚合；同一 N 下两个 C 分属两个 regime，不会碰撞。"""
    out: dict = {}
    for r in rows:
        out.setdefault((r["N"], r["regime"]), []).append(r)
    return out


def sem(a: np.ndarray) -> float:
    return float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else math.nan


def report_case_II(groups: dict, lam_a: float, lam_b: float) -> None:
    keys = sorted(k for k in groups if k[1] == "case_II")
    if not keys:
        return
    print("\n=== Case II: T_c^th = t^1_L (zero-parameter one-sided bound) ===")
    print(f"{'N':>7} {'seeds':>6} {'lambda':>8} {'sem':>7} {'min':>7} {'max':>7} "
          f"{'lam<1':>7} {'lam(N)':>8} {'calib err':>10} {'bound err':>10}")
    for k in keys:
        rs = groups[k]
        lam = np.array([r["t_rhoc_sim"] / r["t1_laplace_ih"] for r in rs])
        lam_pred = theory.lambda_calibration(k[0], a=lam_a, b=lam_b)
        calib_err = lam_pred / lam.mean() - 1.0
        bound_err = 1.0 / lam.mean() - 1.0
        print(f"{k[0]:>7} {len(lam):>6} {lam.mean():>8.4f} {sem(lam):>7.4f} "
              f"{lam.min():>7.4f} {lam.max():>7.4f} "
              f"{int((lam < 1).sum()):>4}/{len(lam):<2} {lam_pred:>8.4f} "
              f"{calib_err:>+9.1%} {bound_err:>+9.1%}")


def report_case_I(groups: dict) -> None:
    keys = sorted(k for k in groups if k[1] == "case_I")
    if not keys:
        return
    print("\n=== Case I: T_c^th = t^1_L + (t^E - t^1)_MF (tail zeroed) ===")
    print(f"{'N':>7} {'seeds':>6} {'err mean':>9} {'sem':>7} {'err min':>8} "
          f"{'err max':>8} {'trig/t1_L':>10} {'casc sim/MF':>12} "
          f"{'tail/T_c':>9} {'tail evts':>10} {'S_h^E/rho_c':>12}")
    for k in keys:
        rs = groups[k]
        th = np.array([r["t1_laplace_ih"] + r["casc_mf"] for r in rs])
        sim = np.array([r["t_rhoc_sim"] for r in rs])
        err = th / sim - 1.0
        trig = np.array([r["t_sh1_sim"] / r["t1_laplace_ih"] for r in rs])
        # 三段实测分解：初始段 [1, S_h^1] / 级联段 [S_h^E, S_h^1] / 尾段 [rho_c, S_h^E)
        # 理论把尾段置零，故尾段占比是"negligible tail"假设的直接体检指标。
        casc_sim = np.array([r["t_shE_sim"] - r["t_sh1_sim"] for r in rs])
        casc_mf = np.array([r["casc_mf"] for r in rs])
        tail_share = np.array([(r["t_rhoc_sim"] - r["t_shE_sim"]) / r["t_rhoc_sim"]
                               for r in rs])
        n_tail = np.array([float(r["n_events_tail"]) for r in rs])
        k0 = np.array([r["k0"] for r in rs]).mean()
        k2 = np.array([r["k2"] for r in rs]).mean()
        band = theory.S_h_E(k0) / theory.rho_c(k0, k2, 2)
        print(f"{k[0]:>7} {len(err):>6} {err.mean():>+8.1%} {sem(err):>7.1%} "
              f"{err.min():>+7.1%} {err.max():>+7.1%} {trig.mean():>10.3f} "
              f"{(casc_sim / casc_mf).mean():>12.3f} {tail_share.mean():>8.1%} "
              f"{n_tail.mean():>10.1f} {band:>12.2f}")


def report_lambda_fits(groups: dict, lam_a: float, lam_b: float) -> None:
    """在全部 N 上比较 lambda(N) 的候选函数形式（sem 加权最小二乘）。

    findings ⑲ 在 N<=2e4 时判定"对数式与饱和式不可辨识"；把 N 推到 1e5 后，
    对数式在外推方向已被数据甩开（chi^2 14.7 vs 6.1，同为 2 参数），论文遂改用
    饱和式 a+b/ln N（findings ㉓㉕）。对数式两行留作被弃用形式的对照。
    """
    from scipy.optimize import curve_fit

    keys = sorted(k for k in groups if k[1] == "case_II")
    if len(keys) < 4:
        return
    N = np.array([k[0] for k in keys], float)
    lam = np.array([np.mean([r["t_rhoc_sim"] / r["t1_laplace_ih"] for r in groups[k]])
                    for k in keys])
    err = np.array([sem(np.array([r["t_rhoc_sim"] / r["t1_laplace_ih"]
                                  for r in groups[k]])) for k in keys])

    models = {
        "paper  a + b/ln N  (fixed)": (lambda n: lam_a + lam_b / np.log(n),
                                       None, 0),
        "refit  a + b/ln N": (lambda n, a, b: a + b / np.log(n), (0.6, 1.8), 2),
        "dropped  1-g*ln(N/N0)  (refit)": (lambda n, g, n0: 1.0 - g * np.log(n / n0),
                                           (0.03, 10.0), 2),
        "refit  linf+(1-linf)(N/N0)^-alpha":
            (lambda n, li, n0, al: li + (1 - li) * (n / n0) ** (-al),
             (0.7, 10.0, 0.3), 3),
    }
    print("\n=== lambda(N) functional form (weighted LSQ over all N) ===")
    print(f"{'model':>34} {'chi2':>8} {'dof':>4} {'max |dev|':>10} "
          f"{'lambda(1e6)':>12}  params")
    for name, (fn, p0, npar) in models.items():
        if p0 is None:
            pred, par = fn(N), ()
        else:
            par, _ = curve_fit(fn, N, lam, p0=p0, sigma=err, absolute_sigma=True,
                               maxfev=200_000)
            pred = fn(N, *par)
        chi2 = float((((lam - pred) / err) ** 2).sum())
        far = fn(1e6) if p0 is None else fn(1e6, *par)
        ps = ", ".join(f"{v:.4g}" for v in par) if len(par) else "-"
        print(f"{name:>34} {chi2:>8.1f} {len(N) - npar:>4} "
              f"{np.abs(pred / lam - 1).max():>9.1%} {far:>12.3f}  {ps}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=["tail_diag_m2_kmin3.csv"],
                    help="results/ 下的一或多份 tail_diag CSV")
    ap.add_argument("--lam_a", type=float, default=0.583,
                    help="lambda(N)=a+b/ln N 的 a（论文采用值）")
    ap.add_argument("--lam_b", type=float, default=1.808,
                    help="lambda(N)=a+b/ln N 的 b（论文采用值）")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    groups = group_by_config(rows)
    print(f"loaded {len(rows)} rows from {len(args.csv)} file(s); "
          f"N = {sorted({k[0] for k in groups})}")
    report_case_II(groups, args.lam_a, args.lam_b)
    report_case_I(groups)
    report_lambda_fits(groups, args.lam_a, args.lam_b)


if __name__ == "__main__":
    main()
