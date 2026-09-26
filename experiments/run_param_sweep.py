"""T_c 对五个参数的依赖：C、m、<k>_0、N、<b>（main.tex L342 的待填内容）。

每个 sweep 只动一个参数，其余固定在参考点 N=2000, m=2, k_min=3, b~U(0,1)；
容量默认取跨 C* 的两个值（0.75C*/1.5C*）以同时给出 Case I 与 Case II 两支，
`C` sweep 例外——它就是要连续扫过 C* 看 regime 交界。

仿真用 fastsim 淬火时钟引擎（离散模型的精确仿真，见 src/fastsim.py），
仿真口径为 t_rhoc（S_h 首次跌破 rho_c），与 run_tail_diag / 论文检验表一致。
理论列同时给零参数式与 Case II 的 lambda(N) 校准式。

每个 sweep 内，同一 seed 复用同一张超图（C / <b> sweep 连 b 也复用），
所以曲线的形状不被网络实现的涨落污染。

用法：
  .venv/Scripts/python.exe experiments/run_param_sweep.py --sweeps C m k0 N b --seeds 10
  .venv/Scripts/python.exe experiments/run_param_sweep.py --sweeps C --seeds 20
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import theory  # noqa: E402
from src.fastsim import (  # noqa: E402
    build_incidence_lists, edge_rates, sample_quenched_uniforms, simulate_quenched,
)
from src.hypergraph import (  # noqa: E402
    build_incidence, generate_powerlaw_hypergraph, hyperdegree_moments,
    sample_vulnerability,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# 参考点（除被扫的那个参数外全部固定于此）
REF = dict(N=2000, m=2, k_min=3, gamma=2.5, b_dist="uniform", b_param=1.0)


class Network:
    """一次生成、多次复用的超图 + 脆弱度（同 seed 下 C / <b> 扫描共用）。"""

    def __init__(self, N: int, m: int, k_min: int, gamma: float,
                 b_dist: str, b_param: float, seed: int):
        rng = np.random.default_rng(seed)
        self.edge_nodes, _ = generate_powerlaw_hypergraph(
            N, m, gamma=gamma, k_min=k_min, rng=rng)
        self.N, self.m = N, m
        self.E0 = len(self.edge_nodes)
        self.node_edges = build_incidence_lists(self.edge_nodes, N)
        self.k0, self.k2 = hyperdegree_moments(build_incidence(self.edge_nodes, N))
        self.b = sample_vulnerability(N, b_dist, rng, b_param)
        self.b_dist, self.b_param = b_dist, b_param
        self.b_mean = float(self.b.mean())
        self.u = sample_quenched_uniforms(self.E0, rng)
        self.rho_c = theory.rho_c(self.k0, self.k2, m)
        self.shE = theory.S_h_E(self.k0)
        self.C_star = theory.C_threshold(self.k0, m)

    def run(self, C: float, b_scale: float = 1.0) -> dict:
        """在容量 C 上跑一条轨迹；b_scale 用于 <b> sweep（同一 b 整体缩放）。"""
        sh1 = theory.S_h_1star(self.k0, self.m, C)
        if not (0.0 < sh1 < 1.0):
            return {}                                   # 退化容量区，跳过
        b_param = self.b_param * b_scale
        edge_p = edge_rates(self.edge_nodes, self.b * b_scale, self.N, self.m)
        res = simulate_quenched(self.edge_nodes, self.node_edges, edge_p, C,
                                sh1, self.shE, self.rho_c, self.u)
        b_mean = self.b_mean * b_scale
        rgm = theory.regime(self.k0, self.m, C)
        th0 = theory.collapse_time_Tc(self.N, self.m, self.k0, self.k2, b_mean, C,
                                      self.b_dist, b_param)
        th_cal = th0
        if rgm == "case_II":
            th_cal = th0 * theory.lambda_calibration(self.N)
        return dict(
            N=self.N, m=self.m, E0=self.E0, k0=self.k0, k2=self.k2,
            b_dist=self.b_dist, b_param=b_param, b_mean=b_mean,
            C=C, C_star=self.C_star, C_over_Cstar=C / self.C_star, regime=rgm,
            rho_c=self.rho_c, S_h_E=self.shE, S_h_1=sh1,
            Tc_sim=res["t_rhoc_sim"], Tc_th=th0, Tc_th_calib=th_cal,
            t_sh1_sim=res["t_sh1_sim"], t_shE_sim=res["t_shE_sim"],
        )


def _rows(sweep: str, values, make_net, run_args, seeds: int) -> list[dict]:
    """通用骨架：对每个参数值 × 每个 seed 跑一次，打上 sweep/xvalue 标签。"""
    rows = []
    for val in values:
        for sd in range(seeds):
            net = make_net(val, sd)
            for C, b_scale in run_args(net, val):
                r = net.run(C, b_scale)
                if not r:
                    continue
                rows.append(dict(sweep=sweep, x=float(val), x_ctrl=float(val),
                             seed=sd, **r))
    return rows


def sweep_C(seeds: int, n_points: int = 15) -> list[dict]:
    """连续扫 C/C*：Case II -> Case I 的 regime 交界（预期在 C=C* 处出现拐点）。"""
    rows = []
    for sd in range(seeds):
        net = Network(REF["N"], REF["m"], REF["k_min"], REF["gamma"],
                      REF["b_dist"], REF["b_param"], sd)
        # 下界取退化容量区之上一点：S_h^1<1 ⟺ C > (k0-(m-1)/m)/(k0-1)
        m, k0 = net.m, net.k0
        C_lo = 1.05 * (k0 - (m - 1) / m) / (k0 - 1)
        for i, C in enumerate(np.geomspace(C_lo, 3.0 * net.C_star, n_points)):
            r = net.run(float(C))
            if r:
                rows.append(dict(sweep="C", x=r["C_over_Cstar"], x_ctrl=float(i),
                                 seed=sd, **r))
    return rows


def sweep_m(seeds: int, ms=(2, 3, 4, 5)) -> list[dict]:
    """扫超边阶数 m（T_c ~ N^m，预期每加一阶跳几个数量级）。"""
    return _rows("m", ms,
                 lambda m, sd: Network(REF["N"], m, REF["k_min"], REF["gamma"],
                                       REF["b_dist"], REF["b_param"], sd),
                 lambda net, _: [(0.75 * net.C_star, 1.0), (1.5 * net.C_star, 1.0)],
                 seeds)


def sweep_k0(seeds: int, k_mins=(1, 2, 3, 4, 5, 6, 8)) -> list[dict]:
    """经 k_min 扫 <k>_0（gamma 固定 2.5）；x 记为实测 <k>_0。"""
    rows = []
    for k_min in k_mins:
        for sd in range(seeds):
            net = Network(REF["N"], REF["m"], k_min, REF["gamma"],
                          REF["b_dist"], REF["b_param"], sd)
            if not theory.has_initial_gc(net.k0, net.k2, net.m):
                continue                                # 初始就无巨分量，跳过
            for C in (0.75 * net.C_star, 1.5 * net.C_star):
                r = net.run(float(C))
                if r:
                    rows.append(dict(sweep="k0", x=net.k0, x_ctrl=float(k_min),
                                     seed=sd, k_min=k_min, **r))
    return rows


def sweep_N(seeds: int, Ns=(500, 1000, 2000, 5000, 10000, 20000)) -> list[dict]:
    """扫 N（预期 T_c ∝ N^m；拟合斜率见 analyze 输出）。"""
    return _rows("N", Ns,
                 lambda N, sd: Network(N, REF["m"], REF["k_min"], REF["gamma"],
                                       REF["b_dist"], REF["b_param"], sd),
                 lambda net, _: [(0.75 * net.C_star, 1.0), (1.5 * net.C_star, 1.0)],
                 seeds)


def sweep_b(seeds: int, scales=(0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0)) -> list[dict]:
    """扫 <b>（b~U(0,beta)，<b>=beta/2）；p_e 同比例缩放 => 预期 T_c 精确 ∝ 1/<b>。"""
    rows = []
    for sd in range(seeds):
        net = Network(REF["N"], REF["m"], REF["k_min"], REF["gamma"],
                      REF["b_dist"], REF["b_param"], sd)
        for beta in scales:
            for C in (0.75 * net.C_star, 1.5 * net.C_star):
                r = net.run(float(C), b_scale=beta)
                if r:
                    rows.append(dict(sweep="b", x=r["b_mean"], x_ctrl=beta,
                                     seed=sd, **r))
    return rows


SWEEPS = {"C": sweep_C, "m": sweep_m, "k0": sweep_k0, "N": sweep_N, "b": sweep_b}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", default=list(SWEEPS),
                    choices=list(SWEEPS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="param_sweep_m2_kmin3.csv")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for name in args.sweeps:
        t0 = time.perf_counter()
        rows = SWEEPS[name](args.seeds)
        all_rows.extend(rows)
        print(f"sweep {name:>2}: {len(rows):>5} rows in {time.perf_counter()-t0:6.1f}s")

    fields: list[str] = []
    for r in all_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path = os.path.join(RESULTS, args.out)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(all_rows)
    print(f"saved: {path}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
