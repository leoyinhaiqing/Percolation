import numpy as np
import pytest

from src import hypergraph as hg


def test_generation_edge_count_and_handshake():
    rng = np.random.default_rng(0)
    N, m, k0 = 500, 2, 0.6
    edge_nodes = hg.generate_er_hypergraph(N, m, k0, rng)
    E = len(edge_nodes)
    assert E == round(N * k0 / (m + 1))
    # 每行 m+1 个不同节点、升序
    assert edge_nodes.shape[1] == m + 1
    for row in edge_nodes:
        assert len(set(row.tolist())) == m + 1
        assert list(row) == sorted(row.tolist())
    # 握手恒等式 sum k_i = (m+1)*E
    node_edges = hg.build_incidence(edge_nodes, N)
    k = hg.hyperdegree_array(node_edges)
    assert int(k.sum()) == (m + 1) * E


def test_poisson_moments_approx():
    rng = np.random.default_rng(1)
    N, m, k0 = 20000, 2, 0.6
    edge_nodes = hg.generate_er_hypergraph(N, m, k0, rng)
    node_edges = hg.build_incidence(edge_nodes, N)
    k_mean, k2_mean = hg.hyperdegree_moments(node_edges)
    assert k_mean == pytest.approx(k0, abs=0.03)
    # Poisson: <k^2> ≈ <k> + <k>^2
    assert k2_mean == pytest.approx(k_mean + k_mean**2, abs=0.05)


def test_giant_component_toy():
    # 两个分量：{0,1,2,3,4}(5) 与 {5,6,7}(3)，N=8 -> GC=5/8
    edge_nodes = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7]], dtype=np.int64)
    alive = np.ones(3, dtype=bool)
    assert hg.giant_component_fraction(edge_nodes, alive, 8) == pytest.approx(5 / 8)
    # 移除第二条边后 {0,1,2}(3),{3,4}? 3,4 变孤立 -> 最大仍是 {0,1,2}=3, {5,6,7}=3 -> 3/8
    alive2 = np.array([True, False, True])
    assert hg.giant_component_fraction(edge_nodes, alive2, 8) == pytest.approx(3 / 8)


def test_hyperedge_density():
    alive = np.array([True, True, False, False])
    assert hg.hyperedge_density(alive, 4) == pytest.approx(0.5)


def test_powerlaw_degree_sum_divisible():
    rng = np.random.default_rng(3)
    N, m = 2000, 2
    deg = hg.sample_powerlaw_hyperdegrees(N, 2.5, 1, 40, m + 1, rng)
    assert len(deg) == N
    assert int(deg.sum()) % (m + 1) == 0          # 握手可整除
    assert deg.min() >= 1 and deg.max() <= 40


def test_powerlaw_generation_handshake_and_shape():
    rng = np.random.default_rng(4)
    N, m = 2000, 2
    edge_nodes, discarded = hg.generate_powerlaw_hypergraph(
        N, m, gamma=2.5, k_min=1, rng=rng)
    E = len(edge_nodes)
    assert E > 0
    assert 0.0 <= discarded < 1.0
    assert edge_nodes.shape[1] == m + 1
    for row in edge_nodes:                          # 每条超边 m+1 个不同节点、升序
        assert len(set(row.tolist())) == m + 1
        assert list(row) == sorted(row.tolist())
    # 无重复超边
    assert len({tuple(r.tolist()) for r in edge_nodes}) == E
    # 握手恒等式 Σk_i = (m+1)*E（对实际建成的超图）
    node_edges = hg.build_incidence(edge_nodes, N)
    k = hg.hyperdegree_array(node_edges)
    assert int(k.sum()) == (m + 1) * E


def test_powerlaw_heavy_tail_and_gc():
    rng = np.random.default_rng(5)
    N, m = 3000, 2
    edge_nodes, _ = hg.generate_powerlaw_hypergraph(
        N, m, gamma=2.5, k_min=1, rng=rng)
    node_edges = hg.build_incidence(edge_nodes, N)
    k = hg.hyperdegree_array(node_edges)
    k_mean, k2_mean = hg.hyperdegree_moments(node_edges)
    # 重尾：<k^2> 明显重于同均值 Poisson 的 <k>+<k>^2
    assert k2_mean > k_mean + k_mean**2
    # 存在高度节点（重尾表征）
    assert int(k.max()) >= 3 * max(1.0, k_mean)
    # 幂律异质度 → 初始有巨分量
    alive = np.ones(len(edge_nodes), dtype=bool)
    assert hg.giant_component_fraction(edge_nodes, alive, N) > 0.1
