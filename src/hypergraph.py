"""超图生成、关联结构、超度矩、巨分量。"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def sample_vulnerability(N: int, dist: str, rng: np.random.Generator,
                         b_param: float = 1.0) -> np.ndarray:
    """抽取节点脆弱度 b_i。

      "uniform"  b ~ U(0, b_param)   -> <b> = b_param/2（b_param=1 即论文默认）
      "constant" b ≡ b_param         -> <b> = b_param（main.tex L344 的
                                        real-data 协议：全体先设成 0.2 或 0.4）
    b_param 只改变整体尺度时 p_e 同比例缩放，故 T_c ∝ 1/<b> 是精确关系。
    """
    if not (0.0 < b_param <= 1.0):
        raise ValueError(f"b_param={b_param} 须在 (0,1]（b_i ∈ [0,1]）")
    if dist == "uniform":
        return b_param * rng.random(N)
    if dist == "constant":
        return np.full(N, float(b_param))
    raise ValueError(f"unsupported b_dist: {dist!r}")


def sample_powerlaw_hyperdegrees(N: int, gamma: float, k_min: int, k_max: int,
                                 size: int, rng: np.random.Generator) -> np.ndarray:
    """从截断幂律 P(k) ∝ k^{-gamma}（k∈[k_min,k_max]）抽 N 个超度，

    并微调使 Σk_i 能被 size=m+1 整除（握手恒等式 Σk_i=(m+1)|E|，配置模型必需）。
    返回长度 N 的 int 数组。
    """
    ks = np.arange(k_min, k_max + 1)
    weights = ks.astype(float) ** (-gamma)
    weights /= weights.sum()
    deg = rng.choice(ks, size=N, p=weights).astype(np.int64)
    # 把总度数调成 size 的倍数：优先给未达上限的节点 +1，必要时给高于下限的 -1
    while int(deg.sum()) % size != 0:
        incrementable = np.where(deg < k_max)[0]
        if len(incrementable) > 0:
            deg[int(rng.choice(incrementable))] += 1
        else:
            decrementable = np.where(deg > k_min)[0]
            if len(decrementable) == 0:
                raise ValueError("无法调整总度数为 (m+1) 的倍数（度分布过窄）")
            deg[int(rng.choice(decrementable))] -= 1
    return deg


def generate_powerlaw_hypergraph(N: int, m: int, gamma: float = 2.5,
                                 k_min: int = 1, k_max: Optional[int] = None,
                                 rng: Optional[np.random.Generator] = None,
                                 max_rounds: int = 20) -> Tuple[np.ndarray, float]:
    """配置模型生成幂律超度的 m 阶超图（本项目主生成器）。

    抽超度序列 → 建 stub → 随机分组成 (m+1)-超边，拒绝含重复节点的组与重复超边，
    未成边的 stub 收集起来重洗多轮尽量用尽。
    返回 (edge_nodes 形状 (E,m+1) 升序去重, discarded_fraction 未成边 stub 占比)。
    """
    if rng is None:
        rng = np.random.default_rng()
    size = m + 1
    if N < size:
        raise ValueError(f"N={N} < m+1={size}")
    if k_max is None:
        k_max = max(k_min, int(N ** (1.0 / (gamma - 1.0))))  # 自然截断 ~N^{1/(γ-1)}
    k_max = min(k_max, N - 1)

    deg = sample_powerlaw_hyperdegrees(N, gamma, k_min, k_max, size, rng)
    total_stubs = int(deg.sum())
    stubs = np.repeat(np.arange(N, dtype=np.int64), deg)

    edges: set = set()
    for _ in range(max_rounds):
        if len(stubs) < size:
            break
        rng.shuffle(stubs)
        n_groups = len(stubs) // size
        groups = stubs[: n_groups * size].reshape(n_groups, size)
        leftover: List[int] = [int(x) for x in stubs[n_groups * size:]]
        for g in groups:
            key = tuple(sorted(int(x) for x in g))
            if len(set(key)) == size and key not in edges:
                edges.add(key)
            else:  # 组内有重复节点或超边重复 → stub 回收下一轮
                leftover.extend(int(x) for x in g)
        stubs = np.array(leftover, dtype=np.int64)
        if len(leftover) == 0:
            break

    if not edges:
        raise RuntimeError("配置模型未能生成任何超边；调大 N 或调整度分布")
    edge_nodes = np.array(sorted(edges), dtype=np.int64)
    placed_stubs = len(edge_nodes) * size
    discarded_fraction = 1.0 - placed_stubs / total_stubs
    return edge_nodes, discarded_fraction


def generate_er_hypergraph(N: int, m: int, k0: float,
                           rng: np.random.Generator) -> np.ndarray:
    """生成 ER / Poisson 型 m-均匀超图（对照基线，非主生成器）。

    目标超边数 E = round(N*k0/(m+1))，均匀随机抽不重复的 (m+1)-节点子集。
    返回 edge_nodes：形状 (E_actual, m+1) 的 int 数组，每行升序。
    """
    size = m + 1
    E_target = int(round(N * k0 / size))
    if E_target < 1:
        raise ValueError(f"E_target={E_target} < 1；增大 N 或 k0")
    if N < size:
        raise ValueError(f"N={N} < m+1={size}")

    edges = set()
    max_attempts = E_target * 100 + 1000
    attempts = 0
    while len(edges) < E_target and attempts < max_attempts:
        nodes = tuple(sorted(int(x) for x in rng.choice(N, size=size, replace=False)))
        edges.add(nodes)
        attempts += 1
    if len(edges) < E_target:
        raise RuntimeError(
            f"只生成了 {len(edges)}/{E_target} 条超边（碰撞过多）"
        )
    edge_nodes = np.array(sorted(edges), dtype=np.int64)
    return edge_nodes


def build_incidence(edge_nodes: np.ndarray, N: int) -> List[set]:
    """构建节点→所含超边索引集 的关联表。"""
    node_edges: List[set] = [set() for _ in range(N)]
    for ei in range(len(edge_nodes)):
        for v in edge_nodes[ei]:
            node_edges[int(v)].add(ei)
    return node_edges


def hyperdegree_array(node_edges: List[set]) -> np.ndarray:
    """每个节点的超度 k_i = 所含超边数。"""
    return np.array([len(s) for s in node_edges], dtype=np.int64)


def hyperdegree_moments(node_edges: List[set]) -> tuple[float, float]:
    """返回 (<k>, <k^2>)。"""
    k = hyperdegree_array(node_edges).astype(float)
    return float(k.mean()), float((k ** 2).mean())


def giant_component_fraction(edge_nodes: np.ndarray, alive: np.ndarray,
                             N: int) -> float:
    """用并查集计算最大连通分量节点数 / N（孤立点各自成 1 分量）。"""
    parent = list(range(N))
    rank = [0] * N

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # 路径压缩
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for ei in range(len(edge_nodes)):
        if not alive[ei]:
            continue
        nodes = edge_nodes[ei]
        first = int(nodes[0])
        for v in nodes[1:]:
            union(first, int(v))

    sizes: dict[int, int] = {}
    for i in range(N):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values()) / N


def hyperedge_density(alive: np.ndarray, E0: int) -> float:
    """超边密度 S_h = 存活超边数 / 初始超边数。"""
    return float(alive.sum()) / E0
