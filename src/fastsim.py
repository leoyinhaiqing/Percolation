"""Quenched-clock 仿真引擎：离散时间模型的精确仿真 + 稀疏级联。

服务两个当前瓶颈：Track A 的大 N（N>=5e4）与 Track B 的逐节点扰动（N 次重跑）。
与 `dynamics.simulate` 的事件驱动 Gillespie 循环相比有三处改动：

1. **淬火时钟（quenched clocks）**：每条边一次性抽 tau_e ~ Geom(p_e)（逆变换，
   tau_e = ceil(ln u_e / ln(1-p_e))），排序后按序处理、跳过已死的边。
   "每步各边独立以 p_e 自发坍塌"等价于"首次自发坍塌时刻独立同几何分布"，
   故这是**离散时间模型的精确仿真**——比 `dynamics.simulate`「每事件恰一条边
   自发坍塌」的稀有事件近似更严格（后者误差量化见 findings ⑩），且同一步的
   同时坍塌（tau 相同）被正确合并成一个事件处理。
2. **稀疏级联**：重分配只在被触及的边上累加（dict），不再每波分配 O(E) 零数组；
   配合增量存活计数，单事件代价从 O(E) 降到 O(局部邻域)。
3. **共同随机数（CRN）**：u_e 由调用方给定并在各扰动间固定。改变某个 b_i 只改变
   含 i 的 k_i 条边的 p_e（tau_e 随 p_e 单调变小），其余边的时钟一字不动——
   Track B 逐节点测 ΔT_c 的配对差因此几乎无采样噪声。

热循环用 Python list（load/alive/关联表）而非 numpy 标量索引：E~1e5 规模下
list 元素访问明显快于 ndarray 标量访问，且级联本身是不可向量化的图遍历。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# 结构预处理（一次性；Track B 在固定 H 上重复调用时复用）
# --------------------------------------------------------------------------

def build_incidence_lists(edge_nodes: np.ndarray, N: int) -> List[List[int]]:
    """节点 -> 所含超边索引列表（升序）。等价 hypergraph.build_incidence 的 list 版。"""
    size = edge_nodes.shape[1]
    flat = edge_nodes.ravel()
    order = np.argsort(flat, kind="stable")
    nodes_sorted = flat[order]
    edges_sorted = (order // size).astype(np.int64)
    bounds = np.searchsorted(nodes_sorted, np.arange(N + 1))
    return [edges_sorted[bounds[v]:bounds[v + 1]].tolist() for v in range(N)]


def edge_rates(edge_nodes: np.ndarray, b: np.ndarray, N: int, m: int) -> np.ndarray:
    """单步自发坍塌概率 p_e = m!/N^m * sum_{i in e} b_i（main.tex Eq. P_T(e)）。"""
    return math.factorial(m) / N ** m * b[edge_nodes].sum(axis=1)


def quenched_clocks(edge_p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """首次自发坍塌步 tau_e ~ Geom(p_e)，由淬火均匀数 u_e in (0,1] 逆变换给出。

    P(tau > t) = (1-p)^t  =>  tau = ceil(ln u / ln(1-p))，支撑 {1,2,...}。
    p_e 增大 => tau_e 单调不增，这正是 Track B 需要的 CRN 单调性。
    """
    lam = np.log1p(-edge_p)                      # < 0
    if np.any(lam >= 0.0):
        raise ValueError("edge_p 含 <=0 或 >=1 的分量：tau 无定义")
    tau = np.ceil(np.log(u) / lam)
    np.maximum(tau, 1.0, out=tau)                # u==1 的边界
    return tau.astype(np.int64)


def sample_quenched_uniforms(E: int, rng: np.random.Generator) -> np.ndarray:
    """抽 E 个 (0,1] 上的淬火均匀数（避开 0，log 才有定义）。"""
    return 1.0 - rng.random(E)


# --------------------------------------------------------------------------
# 稀疏级联（等价 cascade.stage2_redistribute_cascade，去掉每波 O(E) 分配）
# --------------------------------------------------------------------------

def cascade_sparse(edge_nodes: List[Tuple[int, ...]],
                   node_edges: List[List[int]],
                   load: List[float],
                   alive: List[bool],
                   C: float,
                   seeds: Sequence[int]) -> Tuple[List[int], float]:
    """波同步负载重分配级联（就地改 load/alive），只在被触及的边上累加。

    与 `cascade.stage2_redistribute_cascade` 逐条对应：一波内先把本波所有坍塌边的
    负载按共享节点数 |e∩f| 分给存活邻边，加完再判过载，过载者构成下一波。
    返回 (本事件全部坍塌边[种子+级联], 因无存活邻边而丢失的负载)。
    """
    collapsed: List[int] = []
    lost = 0.0

    frontier = [int(e) for e in seeds]
    for e in frontier:
        alive[e] = False
    collapsed.extend(frontier)

    while frontier:
        extra: Dict[int, float] = {}
        for e in frontier:
            shared: Dict[int, int] = {}
            for v in edge_nodes[e]:
                for f in node_edges[v]:
                    if alive[f]:
                        shared[f] = shared.get(f, 0) + 1
            if not shared:
                lost += load[e]              # 无存活邻边 -> 负载丢失
                continue
            denom = 0
            for s in shared.values():
                denom += s
            le = load[e] / denom
            for f, s in shared.items():
                extra[f] = extra.get(f, 0.0) + le * s

        new: List[int] = []
        for f, dl in extra.items():
            lf = load[f] + dl
            load[f] = lf
            if lf > C:                       # extra 里的边在本波收集时均存活
                alive[f] = False
                new.append(f)
        new.sort()                           # 与 np.nonzero 的升序一致，便于比对
        collapsed.extend(new)
        frontier = new

    return collapsed, lost


# --------------------------------------------------------------------------
# 主循环
# --------------------------------------------------------------------------

class QuenchedRun(dict):
    """仿真输出（dict 子类，便于直接写 CSV 行）。"""

    __getattr__ = dict.__getitem__


def simulate_quenched(edge_nodes: np.ndarray,
                      node_edges: List[List[int]],
                      edge_p: np.ndarray,
                      C: float,
                      sh1: float,
                      shE: float,
                      rho_c: float,
                      u: np.ndarray,
                      L0: float = 1.0,
                      collect_extras: bool = False,
                      stop_at_rho_c: bool = True) -> QuenchedRun:
    """跑一条完整轨迹，返回三段阈值穿越时刻与分区级联统计。

    阈值口径与 `run_tail_diag.run_one` 一致：
      t_sh1  —— S_h 首次 <= S_h^1（级联触发）
      t_shE  —— S_h 首次 <= S_h^E（级联区终点，B=0）
      t_rhoc —— S_h 首次 <  rho_c（密度渗流穿越，即理论 T_c 的仿真对照量）

    `u` 为淬火均匀数（长度 E），调用方负责其跨扰动的一致性（CRN）。
    """
    E0 = len(edge_nodes)
    if len(u) != E0 or len(edge_p) != E0:
        raise ValueError("u / edge_p 长度必须等于超边数 E0")

    tau = quenched_clocks(edge_p, u)
    order = np.argsort(tau, kind="stable")
    tau_sorted = tau[order].tolist()
    order_list = order.tolist()

    en_list: List[Tuple[int, ...]] = [tuple(int(v) for v in row) for row in edge_nodes]
    load: List[float] = [L0] * E0
    alive: List[bool] = [True] * E0
    alive_count = E0

    t_sh1: Optional[int] = None
    t_shE: Optional[int] = None
    t_rhoc: Optional[int] = None
    lost_total = 0.0
    extras: Dict[str, List[int]] = {"init": [], "casc": [], "tail": []}

    i, n = 0, E0
    while i < n:
        t = tau_sorted[i]
        batch: List[int] = []
        j = i
        while j < n and tau_sorted[j] == t:
            e = order_list[j]
            if alive[e]:
                batch.append(e)
            j += 1
        i = j
        if not batch:
            continue

        sh_before = alive_count / E0
        collapsed, lost = cascade_sparse(en_list, node_edges, load, alive, C, batch)
        lost_total += lost
        alive_count -= len(collapsed)
        sh = alive_count / E0

        if collect_extras:
            region = ("init" if sh_before > sh1
                      else "casc" if sh_before >= shE else "tail")
            extras[region].append(len(collapsed) - len(batch))

        if t_sh1 is None and sh <= sh1:
            t_sh1 = t
        if t_shE is None and sh <= shE:
            t_shE = t
        if t_rhoc is None and sh < rho_c:
            t_rhoc = t
            if stop_at_rho_c:
                break
        if alive_count == 0:
            break

    return QuenchedRun(
        t_sh1_sim=t_sh1, t_shE_sim=t_shE, t_rhoc_sim=t_rhoc,
        n_events_init=len(extras["init"]), n_events_casc=len(extras["casc"]),
        n_events_tail=len(extras["tail"]),
        extras=extras, lost_load=lost_total,
        Sh_final=alive_count / E0, alive=alive,
    )


# --------------------------------------------------------------------------
# 静态巨分量（top-k 移除鲁棒性曲线用；不进热循环）
# --------------------------------------------------------------------------

def giant_component_fraction_fast(edge_nodes: np.ndarray,
                                  alive_mask: Optional[np.ndarray],
                                  N: int,
                                  node_mask: Optional[np.ndarray] = None) -> float:
    """最大连通分量节点数 / N（并查集）。

    alive_mask=None 表示全部超边存活；node_mask 给出被保留的节点（被移除的节点
    既不计入分量、也不作为连接媒介）。分母固定为 N，便于不同 k 之间横向比较。
    """
    parent = list(range(N))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    keep = None if node_mask is None else node_mask.tolist()
    for ei in range(len(edge_nodes)):
        if alive_mask is not None and not alive_mask[ei]:
            continue
        nodes = [int(v) for v in edge_nodes[ei]]
        if keep is not None:
            nodes = [v for v in nodes if keep[v]]
            if len(nodes) < 2:
                continue
        ra = find(nodes[0])
        for v in nodes[1:]:
            rb = find(v)
            if ra != rb:
                parent[rb] = ra
                ra = find(ra)

    sizes: Dict[int, int] = {}
    for v in range(N):
        if keep is not None and not keep[v]:
            continue
        r = find(v)
        sizes[r] = sizes.get(r, 0) + 1
    return (max(sizes.values()) / N) if sizes else 0.0
