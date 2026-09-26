"""静态中心性基线的正确性测试（对可解析算出的小图核对）。

m=1（每条"超边"两个节点）时团展开就是原图本身，故可以直接用普通图的
教科书答案检验 Brandes 与 PageRank 的实现。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.centrality import (
    betweenness_centrality,
    clique_expansion,
    hyperdegree_centrality,
    pagerank_centrality,
    static_centralities,
)


def _path_graph(n: int) -> np.ndarray:
    return np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)


def test_clique_expansion_weights_count_shared_hyperedges():
    # 两条 3-节点超边共享 {0,1}
    edge_nodes = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    adj, w = clique_expansion(edge_nodes, 4)
    assert adj[0] == [1, 2, 3]
    assert dict(zip(adj[0], w[0])) == {1: 2.0, 2: 1.0, 3: 1.0}
    assert dict(zip(adj[2], w[2])) == {0: 1.0, 1: 1.0}


def test_hyperdegree_counts_incident_hyperedges():
    edge_nodes = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3]], dtype=np.int64)
    k = hyperdegree_centrality(edge_nodes, 4)
    assert k.tolist() == [3.0, 2.0, 2.0, 2.0]


def test_betweenness_matches_path_graph_closed_form():
    """P5 的归一化介数：端点 0，次端点 0.5，中点 2/3。"""
    N = 5
    adj, _ = clique_expansion(_path_graph(N), N)
    bc = betweenness_centrality(adj, N)
    assert bc[0] == pytest.approx(0.0)
    assert bc[4] == pytest.approx(0.0)
    assert bc[1] == pytest.approx(0.5)
    assert bc[3] == pytest.approx(0.5)
    assert bc[2] == pytest.approx(2.0 / 3.0)


def test_betweenness_star_hub_is_one():
    """星图中心的归一化介数恰为 1（所有叶子对都必须经过它）。"""
    N = 6
    edges = np.array([[0, i] for i in range(1, N)], dtype=np.int64)
    adj, _ = clique_expansion(edges, N)
    bc = betweenness_centrality(adj, N)
    assert bc[0] == pytest.approx(1.0)
    assert np.allclose(bc[1:], 0.0)


def test_pagerank_uniform_on_cycle_and_sums_to_one():
    N = 7
    edges = np.array([[i, (i + 1) % N] for i in range(N)], dtype=np.int64)
    adj, w = clique_expansion(edges, N)
    pr = pagerank_centrality(adj, w, N)
    assert pr.sum() == pytest.approx(1.0)
    assert np.allclose(pr, 1.0 / N, atol=1e-9)


def test_pagerank_ranks_hub_first_and_handles_isolated_node():
    N = 7                                     # 节点 6 孤立（悬挂点路径）
    edges = np.array([[0, i] for i in range(1, 6)], dtype=np.int64)
    adj, w = clique_expansion(edges, N)
    pr = pagerank_centrality(adj, w, N)
    assert pr.sum() == pytest.approx(1.0)
    assert int(np.argmax(pr)) == 0
    assert pr[6] > 0.0                        # 孤立点仍分到 teleport 质量


def test_static_centralities_bundle_shapes():
    rng = np.random.default_rng(0)
    from src.hypergraph import generate_powerlaw_hypergraph
    N = 200
    edge_nodes, _ = generate_powerlaw_hypergraph(N, 2, gamma=2.5, k_min=3, rng=rng)
    cent = static_centralities(edge_nodes, N)
    assert set(cent) == {"hyperdegree", "betweenness", "pagerank"}
    for name, v in cent.items():
        assert v.shape == (N,)
        assert np.all(np.isfinite(v)), name
    assert cent["pagerank"].sum() == pytest.approx(1.0)
