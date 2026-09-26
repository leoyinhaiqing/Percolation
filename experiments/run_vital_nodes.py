"""关键节点识别：动态 ΔT_c 排序 vs 静态中心性（Track B）。

协议照 main.tex \\S numerical-simulation：固定超边阶数 m，先给所有节点一个基线
脆弱度，再逐个把某节点的 b_i 提到 1、其余不动，测 T_c 的下降量
  ΔT_c(i) = T_c(baseline) - T_c(b_i -> 1)   （越大越关键）

基线有两种（`--b_dist`）：
  constant  全体 b_i≡b0（0.2/0.4）—— main.tex 写的协议。此时把 b_i 提到 1
            等于"把恰好 k_i 条关联边的速率提高同一个量"，ΔT_c 到一阶就是超度的
            单调函数，因此动态排序**结构上不可能超过超度**（实测重合度 88–94%）。
  uniform   b~U(0,1) 的异质基线。此时 ΔT_c 依赖 k_i·(1−b_i)，静态中心性看不到
            b 那一半信息，动态测量才有可能带来结构量给不出的东西。
`degree_x_vuln` = k_i(1−b_i) 是对应的零成本静态对照（异质基线下 ΔT_c 的一阶近似），
用来分清"动态测量的收益"与"仅仅是把 b 也算进去"。

三个让它跑得动的做法：
1. **共同随机数（CRN）**：淬火时钟 u_e 在 baseline 与所有 N 次扰动之间完全固定
   （src/fastsim.py）。b_i 只改变含 i 的 k_i 条边的 p_e，其余边的时钟一字不动，
   所以配对差 ΔT_c 几乎没有采样噪声——否则单节点扰动（~k_i/E 的相对速率变化）
   会被 seed 间涨落彻底淹没。
2. **两段筛选**：第一段用少量时钟扫全部节点，取前 screen_frac 进第二段用多时钟精算。
3. **增量速率**：扰动只需 `edge_p[incident] += coef*(1-b0)`，不重算全图；
   逐节点扫描本身完全独立，用进程池并行（各 worker 由同一 seed 重建 Setup，
   时钟逐位相同，故并行只改墙钟时间、不改任何结果）。

另三条对照基线为 src/centrality.py 的超度 / 介数 / PageRank（团展开）。
评估用两条曲线：
  - 静态：移除 top-k 节点（连带其全部超边）后的巨分量大小；
  - 动态：把 top-k 节点的 b 一起提到 1 后的 T_c 相对下降。
后者与本文的动力学口径一致，前者是文献里通用的结构鲁棒性口径，两条都给。

`--greedy`（默认开）另算集合感知的贪心 ΔT_c 与贪心覆盖对照。**实测贪心 ΔT_c 反而最差**：
单步边际信号只有 T_c 的 ~0.2%，在可负担的时钟数下逐步 argmin 会被 winner's curse
主导（选中的是噪声最有利的候选，不是最好的节点），误差逐步累积。默认仍算出来是
为了把这条负面证据留在数据里。

用法：
  .venv/Scripts/python.exe experiments/run_vital_nodes.py --N 1000 --b0 0.2
  .venv/Scripts/python.exe experiments/run_vital_nodes.py --N 1000 --b_dist uniform --no-greedy
  .venv/Scripts/python.exe experiments/run_vital_nodes.py --dataset house-bills --m 2
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import theory  # noqa: E402
from src.centrality import static_centralities  # noqa: E402
from src.fastsim import (  # noqa: E402
    build_incidence_lists, giant_component_fraction_fast, sample_quenched_uniforms,
    simulate_quenched,
)
from src.fastsim import edge_rates  # noqa: E402
from src.hypergraph import (  # noqa: E402
    build_incidence, generate_powerlaw_hypergraph, hyperdegree_moments,
    sample_vulnerability,
)
from src.realdata import load_uniform_sub_hypergraph  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

METHODS = ("dTc", "hyperdegree", "betweenness", "pagerank", "degree_x_vuln")
GREEDY_METHODS = ("dTc_greedy", "coverage_greedy")


class Setup:
    """固定的超图 + 常数脆弱度基线 + 一组淬火时钟。"""

    def __init__(self, edge_nodes: np.ndarray, N: int, m: int, b0: float,
                 C_frac: float, n_clocks: int, seed: int,
                 b_dist: str = "constant"):
        self.edge_nodes, self.N, self.m, self.b0 = edge_nodes, N, m, b0
        self.b_dist = b_dist
        self.E0 = len(edge_nodes)
        self.node_edges = build_incidence_lists(edge_nodes, N)
        self.k0, self.k2 = hyperdegree_moments(build_incidence(edge_nodes, N))
        self.rho_c = theory.rho_c(self.k0, self.k2, m)
        self.shE = theory.S_h_E(self.k0)
        self.C_star = theory.C_threshold(self.k0, m)
        self.C = C_frac * self.C_star
        self.sh1 = theory.S_h_1star(self.k0, m, self.C)
        self.regime = theory.regime(self.k0, m, self.C)

        self.coef = math.factorial(m) / N ** m
        rng = np.random.default_rng(seed)
        # b_dist="constant" 是 main.tex 的协议（全体 b_i≡b0）；"uniform" 是异质基线
        # b~U(0,1)，此时 ΔT_c 依赖 k_i·(1−b_i)，静态中心性看不到 b 那一半信息。
        self.b = (np.full(N, b0) if b_dist == "constant"
                  else sample_vulnerability(N, "uniform", rng))
        self.edge_p = edge_rates(edge_nodes, self.b, N, m)
        self.clocks = [sample_quenched_uniforms(self.E0, rng) for _ in range(n_clocks)]

    def Tc(self, edge_p: np.ndarray, clocks) -> list[float]:
        """给定速率，在每个时钟上跑一次，返回 T_c 列表（未穿越记 nan）。"""
        out = []
        for u in clocks:
            r = simulate_quenched(self.edge_nodes, self.node_edges, edge_p, self.C,
                                  self.sh1, self.shE, self.rho_c, u)
            out.append(float(r["t_rhoc_sim"]) if r["t_rhoc_sim"] is not None
                       else math.nan)
        return out

    def perturbed_rates(self, nodes) -> np.ndarray:
        """把给定节点的 b 提到 1：只在其关联边上加 coef*(1-b_i)。"""
        ep = self.edge_p.copy()
        for i in nodes:
            idx = self.node_edges[i]
            if idx:
                ep[idx] += self.coef * (1.0 - self.b[i])
        return ep


# --------------------------------------------------------------------------
# 并行：逐节点扫描是完全独立的，唯一的共享状态是 Setup（各 worker 用同一 seed
# 重建，时钟因而逐位相同——并行不改变任何结果，只改变墙钟时间）。
# --------------------------------------------------------------------------

_W: dict = {}


def _init_worker(spec: tuple) -> None:
    _W["S"] = Setup(*spec)
    _W["base"] = {}


def _base_for(R: int) -> np.ndarray:
    S = _W["S"]
    if R not in _W["base"]:
        _W["base"][R] = np.array(S.Tc(S.edge_p, S.clocks[:R]), float)
    return _W["base"][R]


def _dTc_one(job: tuple) -> float:
    i, R = job
    S = _W["S"]
    pert = np.array(S.Tc(S.perturbed_rates([i]), S.clocks[:R]), float)
    return float(np.nanmean(_base_for(R) - pert))


def _Tc_topk(job: tuple) -> float:
    top, R = job
    S = _W["S"]
    if not top:
        return float(np.nanmean(_base_for(R)))
    return float(np.nanmean(S.Tc(S.perturbed_rates(list(top)), S.clocks[:R])))


def delta_Tc_scan(S: Setup, candidates, R: int, pool=None) -> np.ndarray:
    """对候选节点逐个测 ΔT_c（配对到同一组时钟）。返回与 candidates 等长的数组。"""
    jobs = [(int(i), R) for i in candidates]
    if pool is not None:
        return np.array(pool.map(_dTc_one, jobs, chunksize=8), float)
    base = np.array(S.Tc(S.edge_p, S.clocks[:R]), float)
    out = np.empty(len(jobs), float)
    for j, (i, _) in enumerate(jobs):
        pert = np.array(S.Tc(S.perturbed_rates([i]), S.clocks[:R]), float)
        out[j] = float(np.nanmean(base - pert))
    return out


def greedy_dTc(S: Setup, candidates, K: int, R: int, pool=None) -> list[int]:
    """贪心动态排序：每步挑"加进来后 T_c 掉得最多"的节点（边际 ΔT_c 最大）。

    与逐节点 ΔT_c 排序的区别在**集合感知**：单节点分数最高的 K 个节点关联的超边
    高度重叠，取前 K 会浪费预算；贪心每一步都按"在已选集合之上还能多推进多少"来选。
    """
    selected: list[int] = []
    remaining = [int(i) for i in candidates]
    while len(selected) < K and remaining:
        jobs = [(tuple(selected + [i]), R) for i in remaining]
        Tcs = (pool.map(_Tc_topk, jobs, chunksize=2) if pool is not None
               else [_Tc_topk_serial(S, j) for j in jobs])
        selected.append(remaining.pop(int(np.argmin(Tcs))))
    return selected


def _Tc_topk_serial(S: Setup, job: tuple) -> float:
    top, R = job
    if not top:
        return float(np.nanmean(S.Tc(S.edge_p, S.clocks[:R])))
    return float(np.nanmean(S.Tc(S.perturbed_rates(list(top)), S.clocks[:R])))


def greedy_coverage(node_edges, candidates, K: int) -> list[int]:
    """贪心静态对照：每步挑"还能新覆盖最多条超边"的节点（max-coverage 贪心）。

    这是必需的公平对照——否则"贪心动态 vs 非贪心静态"分不清优势来自动力学
    还是仅仅来自集合感知。
    """
    covered: set[int] = set()
    remaining = [int(i) for i in candidates]
    selected: list[int] = []
    while len(selected) < K and remaining:
        gains = [len(set(node_edges[i]) - covered) for i in remaining]
        j = int(np.argmax(gains))
        i = remaining.pop(j)
        selected.append(i)
        covered |= set(node_edges[i])
    return selected


def rank_desc(score: np.ndarray) -> np.ndarray:
    """降序排名（0 = 最关键）。"""
    order = np.argsort(-score, kind="stable")
    r = np.empty(len(score), np.int64)
    r[order] = np.arange(len(score))
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关（并列名次用平均秩）。"""
    def ranks(x):
        o = np.argsort(x, kind="stable")
        rk = np.empty(len(x), float)
        rk[o] = np.arange(len(x), dtype=float)
        # 并列取平均秩
        xs = x[o]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                rk[o[i:j + 1]] = np.arange(i, j + 1).mean()
            i = j + 1
        return rk
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den else math.nan


def evaluation_curves(S: Setup, orders: dict, ks, R: int, pool=None) -> list[dict]:
    """对每个方法、每个 k，算 (静态巨分量, 动态 T_c 比值)。orders[name] 为有序节点表。"""
    gc0 = giant_component_fraction_fast(S.edge_nodes, None, S.N)
    tops, meta = [], []
    for name, order in orders.items():
        for k in ks:
            tops.append(tuple(int(v) for v in order[:k]))
            meta.append((name, k))

    jobs = [((), R)] + [(t, R) for t in tops]
    if pool is not None:
        Tcs = pool.map(_Tc_topk, jobs, chunksize=1)
    else:
        base = np.array(S.Tc(S.edge_p, S.clocks[:R]), float)
        Tcs = [float(np.nanmean(base))] + [
            float(np.nanmean(S.Tc(S.perturbed_rates(list(t)), S.clocks[:R])))
            if t else float(np.nanmean(base)) for t in tops]
    base_Tc, Tcs = Tcs[0], Tcs[1:]

    rows = []
    for (name, k), top, Tc_k in zip(meta, tops, Tcs):
        keep = np.ones(S.N, bool)
        keep[list(top)] = False
        # 节点被移除 => 其所有关联超边随之消失（超边是不可分的整体）
        alive = keep[S.edge_nodes].all(axis=1)
        gc = giant_component_fraction_fast(S.edge_nodes, alive, S.N, keep)
        rows.append(dict(method=name, k=k, gc=gc, gc_rel=gc / gc0,
                         Tc=Tc_k, Tc_rel=Tc_k / base_Tc))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="真实数据集名（如 house-bills）；缺省则生成幂律超图")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=2.5)
    ap.add_argument("--b0", type=float, default=0.2, help="基线常数脆弱度（0.2/0.4）")
    ap.add_argument("--b_dist", choices=("constant", "uniform"), default="constant",
                    help="基线脆弱度分布：constant=main.tex 协议；uniform=异质基线 U(0,1)")
    ap.add_argument("--C_fracs", type=float, nargs="+", default=[1.5, 0.75],
                    help="容量取 C*的倍数：>=1 为 Case I，<1 为 Case II")
    ap.add_argument("--clocks_screen", type=int, default=20)
    ap.add_argument("--clocks_final", type=int, default=200)
    ap.add_argument("--clocks_eval", type=int, default=200)
    ap.add_argument("--screen_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="并行进程数（0=自动取 cpu_count-2；1=串行）")
    ap.add_argument("--greedy", action="store_true", default=True,
                    help="额外算集合感知的贪心动态排序与贪心覆盖对照")
    ap.add_argument("--no-greedy", dest="greedy", action="store_false")
    ap.add_argument("--clocks_greedy", type=int, default=20,
                    help="贪心每步用的时钟数（步数多，单步可少些）")
    ap.add_argument("--n_k", type=int, default=12, help="评估曲线的 k 取点数")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.dataset:
        edge_nodes, N, meta = load_uniform_sub_hypergraph(args.dataset, args.m)
        tag = args.tag or f"{args.dataset}_m{args.m}"
        print(f"dataset {args.dataset}: N={N} |E_m|={len(edge_nodes)} "
              f"({meta['n_edges_raw']} raw hyperedges, sizes "
              f"{meta['size_min']}-{meta['size_max']})")
    else:
        rng = np.random.default_rng(args.seed)
        edge_nodes, _ = generate_powerlaw_hypergraph(
            args.N, args.m, gamma=args.gamma, k_min=args.k_min, rng=rng)
        N = args.N
        tag = args.tag or f"synthetic_N{N}_m{args.m}"

    cent = static_centralities(edge_nodes, N)
    node_rows: list[dict] = []
    curve_rows: list[dict] = []

    n_workers = (args.workers if args.workers > 0
                 else max(1, (os.cpu_count() or 2) - 2))

    for C_frac in args.C_fracs:
        spec = (edge_nodes, N, args.m, args.b0, C_frac, args.clocks_final,
                args.seed + 1, args.b_dist)
        S = Setup(*spec)
        print(f"\n--- C={S.C:.3f} ({C_frac}C*, {S.regime}), <k>0={S.k0:.2f}, "
              f"rho_c={S.rho_c:.4f}, S_h^E/rho_c={S.shE / S.rho_c:.2f}, "
              f"b={args.b_dist}({args.b0 if args.b_dist == 'constant' else '0,1'}), "
              f"workers={n_workers} ---", flush=True)

        pool = (mp.Pool(n_workers, initializer=_init_worker, initargs=(spec,))
                if n_workers > 1 else None)
        # 只有关联到超边的节点才可能有 ΔT_c；孤立点恒为 0
        active = [i for i in range(N) if S.node_edges[i]]
        t0 = time.perf_counter()
        s1 = delta_Tc_scan(S, active, args.clocks_screen, pool)
        print(f"stage 1: {len(active)} nodes x {args.clocks_screen} clocks "
              f"in {time.perf_counter()-t0:.1f}s", flush=True)

        # 入围名单 = 第一段 ΔT_c 前 screen_frac ∪ 超度前 screen_frac。
        # 单取前者会因第一段时钟少而漏掉真正的高位节点（rho(ΔT_c^{R=20}, 收敛值)
        # 只有 ~0.63，见 run_vital_diag.py）；并上超度前列可保证结构性枢纽不被漏掉，
        # 同时仍允许"度不高但动态上关键"的节点通过第一段自行入围。
        n_keep = max(20, int(round(args.screen_frac * len(active))))
        cand = {active[j] for j in np.argsort(-s1)[:n_keep]}
        cand |= {int(i) for i in np.argsort(-cent["hyperdegree"])[:n_keep]
                 if S.node_edges[int(i)]}
        short = sorted(cand)
        t0 = time.perf_counter()
        s2 = delta_Tc_scan(S, short, args.clocks_final, pool)
        print(f"stage 2: {len(short)} nodes x {args.clocks_final} clocks "
              f"in {time.perf_counter()-t0:.1f}s", flush=True)

        # 最终分数：入围节点用第二段精算值，其余保留第一段（量级已远小）
        dTc = np.full(N, -np.inf)
        for j, i in enumerate(active):
            dTc[i] = s1[j]
        for j, i in enumerate(short):
            dTc[i] = s2[j]
        dTc[np.isneginf(dTc)] = 0.0                 # 孤立点

        # 静态混合对照 k_i(1-b_i)：把"结构"和"脆弱度"相乘，零仿真成本。
        # 异质基线下它是 ΔT_c 的一阶近似，正好检验动态测量有没有超出一阶的信息。
        cent_here = dict(cent)
        cent_here["degree_x_vuln"] = cent["hyperdegree"] * (1.0 - S.b)
        scores = {"dTc": dTc, **cent_here}
        for name in METHODS:
            if name == "dTc":
                continue
            print(f"  spearman(dTc, {name:>11}) = "
                  f"{spearman(dTc, scores[name]):+.3f}")
        top10 = np.argsort(-dTc)[:10]
        print(f"  top-10 by dTc: {top10.tolist()}")
        print(f"  their hyperdegrees: {cent['hyperdegree'][top10].astype(int).tolist()}")

        ks = sorted({int(round(x)) for x in
                     np.linspace(0, max(10, int(0.10 * N)), args.n_k)})
        K = max(ks)
        orders = {name: [int(v) for v in np.argsort(-sc, kind="stable")]
                  for name, sc in scores.items()}

        if args.greedy:
            # 贪心候选池：入围名单已含 ΔT_c 与超度两条前列，够覆盖 top-K
            t0 = time.perf_counter()
            orders["dTc_greedy"] = greedy_dTc(S, short, K, args.clocks_greedy, pool)
            print(f"greedy dTc: K={K} over {len(short)} candidates x "
                  f"{args.clocks_greedy} clocks in {time.perf_counter()-t0:.1f}s",
                  flush=True)
            orders["coverage_greedy"] = greedy_coverage(S.node_edges, active, K)

        t0 = time.perf_counter()
        for r in evaluation_curves(S, orders, ks, args.clocks_eval, pool):
            curve_rows.append(dict(tag=tag, C_frac=C_frac, regime=S.regime, **r))
        print(f"evaluation curves: {len(ks)} k-values in "
              f"{time.perf_counter()-t0:.1f}s", flush=True)
        if pool is not None:
            pool.close()
            pool.join()

        # 记录每个节点的分数用了多少个时钟：入围节点是第二段的精算值，其余是第一段的
        # 粗估——散点图必须区分这两种精度，否则粗估的噪声会被读成"动态排序很散"。
        n_clocks_of = {int(i): args.clocks_screen for i in active}
        n_clocks_of.update({int(i): args.clocks_final for i in short})
        for i in range(N):
            node_rows.append(dict(
                tag=tag, C_frac=C_frac, regime=S.regime, node=i,
                n_clocks=n_clocks_of.get(i, 0),
                b=float(S.b[i]),
                dTc=dTc[i], hyperdegree=cent["hyperdegree"][i],
                betweenness=cent["betweenness"][i], pagerank=cent["pagerank"][i],
                degree_x_vuln=cent_here["degree_x_vuln"][i],
                **{f"rank_{n}": 0 for n in METHODS}))
        block = node_rows[-N:]
        for name in METHODS:
            rk = rank_desc(scores[name])
            for i in range(N):
                block[i][f"rank_{name}"] = int(rk[i])

    for rows, suffix in ((node_rows, "nodes"), (curve_rows, "curves")):
        path = os.path.join(RESULTS, f"vital_{tag}_{suffix}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"saved: {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
