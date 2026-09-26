"""Stage 1 随机坍塌 + Stage 2 负载重分配级联（波同步）。"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def stage1_random_collapse(edge_p: np.ndarray, alive: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """对存活超边按 P(e)=edge_p[e] 伯努利采样，返回初始坍塌超边索引数组。"""
    r = rng.random(len(edge_p))
    mask = alive & (r < edge_p)
    return np.nonzero(mask)[0]


def stage2_redistribute_cascade(
    edge_nodes: np.ndarray,
    node_edges: List[set],
    load: np.ndarray,
    alive: np.ndarray,
    C: float,
    D: np.ndarray,
) -> Tuple[List[int], float]:
    """波同步负载重分配级联（就地修改 load 与 alive）。

    对应 main.tex Stage 2：
      - 把当前坍塌边的负载按共享节点数 |e∩f| 比例分给**存活**邻边；
      - 一波内先把所有贡献加完，再判过载；过载者构成下一波；
      - 直到无新坍塌。

    返回 (本步全部坍塌超边索引列表[初始+级联], 因无存活邻边而丢失的负载总量)。
    """
    E = len(edge_nodes)
    collapsed: List[int] = []
    lost = 0.0

    frontier = [int(e) for e in D]
    for e in frontier:
        alive[e] = False  # 初始坍塌边移除
    collapsed.extend(frontier)

    while frontier:
        extra = np.zeros(E)
        for e in frontier:
            # 收集 e 的存活邻边及共享节点数 |e∩f|
            shared: dict[int, int] = {}
            for v in edge_nodes[e]:
                for f in node_edges[int(v)]:
                    if alive[f]:
                        shared[f] = shared.get(f, 0) + 1
            if not shared:
                lost += float(load[e])  # 无存活邻边 → 负载丢失
                continue
            denom = sum(shared.values())
            le = float(load[e])
            for f, s in shared.items():
                extra[f] += le * s / denom

        if extra.any():
            load += extra  # 仅存活边收到（extra 对非存活边恒为 0）

        new = np.nonzero(alive & (load > C))[0]
        for f in new:
            alive[int(f)] = False
        new_list = [int(f) for f in new]
        collapsed.extend(new_list)
        frontier = new_list

    return collapsed, lost
