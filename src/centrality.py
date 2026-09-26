"""静态中心性基线（超度 / 介数 / PageRank），作为动态 ΔT_c 排序的对照。

超图上没有唯一的"介数""PageRank"定义，本项目统一用**团展开（clique expansion）**：
把每条 (m+1)-超边展成其成员节点间的完全图，边权 w_ij = 同时含 i,j 的超边数。
理由：(1) 与 vital-node 文献里最常见的做法一致，可比性好；(2) 保留了超边内部
"共同参与"的强度信息；(3) 超度基线本身就是超图原生量，三条基线覆盖了
"局部度 / 全局路径 / 谱" 三类不同机制。

介数用未加权团展开上的 Brandes 算法（最短路计数以跳数计）；PageRank 用加权
团展开上的幂迭代。两者都在库内实现，不引入新依赖。
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np


def clique_expansion(edge_nodes: np.ndarray, N: int) -> Tuple[List[List[int]],
                                                              List[List[float]]]:
    """团展开：返回 (adj, w)，adj[i] 为邻居列表，w[i][j] 为对应边权（共超边数）。"""
    acc: List[Dict[int, float]] = [dict() for _ in range(N)]
    for row in edge_nodes:
        nodes = [int(v) for v in row]
        for a in range(len(nodes)):
            for c in range(a + 1, len(nodes)):
                i, j = nodes[a], nodes[c]
                acc[i][j] = acc[i].get(j, 0.0) + 1.0
                acc[j][i] = acc[j].get(i, 0.0) + 1.0
    adj, w = [], []
    for i in range(N):
        nb = sorted(acc[i])
        adj.append(nb)
        w.append([acc[i][j] for j in nb])
    return adj, w


def hyperdegree_centrality(edge_nodes: np.ndarray, N: int) -> np.ndarray:
    """超度 k_i = 含节点 i 的超边数（超图原生基线）。"""
    k = np.zeros(N, dtype=float)
    for row in edge_nodes:
        for v in row:
            k[int(v)] += 1.0
    return k


def betweenness_centrality(adj: List[List[int]], N: int,
                           normalized: bool = True) -> np.ndarray:
    """未加权团展开上的介数中心性（Brandes 1999，无权最短路）。"""
    BC = np.zeros(N, dtype=float)
    for s in range(N):
        if not adj[s]:
            continue
        stack: List[int] = []
        preds: List[List[int]] = [[] for _ in range(N)]
        sigma = [0.0] * N
        dist = [-1] * N
        sigma[s], dist[s] = 1.0, 0
        Q = deque([s])
        while Q:
            v = Q.popleft()
            stack.append(v)
            dv, sv = dist[v], sigma[v]
            for x in adj[v]:
                if dist[x] < 0:
                    dist[x] = dv + 1
                    Q.append(x)
                if dist[x] == dv + 1:
                    sigma[x] += sv
                    preds[x].append(v)
        delta = [0.0] * N
        while stack:
            x = stack.pop()
            coeff = (1.0 + delta[x]) / sigma[x]
            for v in preds[x]:
                delta[v] += sigma[v] * coeff
            if x != s:
                BC[x] += delta[x]
    BC *= 0.5                                        # 无向图每条最短路被数两次
    if normalized and N > 2:
        BC *= 2.0 / ((N - 1) * (N - 2))
    return BC


def pagerank_centrality(adj: List[List[int]], w: List[List[float]], N: int,
                        alpha: float = 0.85, tol: float = 1e-10,
                        max_iter: int = 500) -> np.ndarray:
    """加权团展开上的 PageRank（幂迭代；悬挂点质量均摊回全体）。"""
    out = np.array([sum(wi) for wi in w], dtype=float)
    dangling = out == 0.0
    x = np.full(N, 1.0 / N)
    for _ in range(max_iter):
        nxt = np.zeros(N)
        contrib = np.where(dangling, 0.0, x / np.where(dangling, 1.0, out))
        for i in range(N):
            ci = contrib[i]
            if ci == 0.0:
                continue
            nb, wi = adj[i], w[i]
            for t in range(len(nb)):
                nxt[nb[t]] += ci * wi[t]
        nxt = alpha * (nxt + x[dangling].sum() / N) + (1.0 - alpha) / N
        if np.abs(nxt - x).sum() < tol:
            return nxt
        x = nxt
    return x


def static_centralities(edge_nodes: np.ndarray, N: int,
                        alpha: float = 0.85) -> Dict[str, np.ndarray]:
    """一次性算三条静态基线，返回 {name: score array}。"""
    adj, w = clique_expansion(edge_nodes, N)
    return {
        "hyperdegree": hyperdegree_centrality(edge_nodes, N),
        "betweenness": betweenness_centrality(adj, N),
        "pagerank": pagerank_centrality(adj, w, N, alpha=alpha),
    }
