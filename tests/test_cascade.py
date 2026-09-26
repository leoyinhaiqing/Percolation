import numpy as np
import pytest

from src import hypergraph as hg
from src.cascade import stage1_random_collapse, stage2_redistribute_cascade


def _chain():
    # 链式：e0={0,1,2}, e1={2,3,4}, e2={4,5,6}; e0-e1 共享 2, e1-e2 共享 4
    edge_nodes = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.int64)
    node_edges = hg.build_incidence(edge_nodes, 7)
    return edge_nodes, node_edges


def test_stage1_deterministic():
    edge_p = np.array([1.0, 0.0, 1.0])
    alive = np.ones(3, dtype=bool)
    rng = np.random.default_rng(0)
    D = stage1_random_collapse(edge_p, alive, rng)
    assert set(D.tolist()) == {0, 2}
    # edge_p=0 永不坍塌
    D0 = stage1_random_collapse(np.zeros(3), alive, rng)
    assert len(D0) == 0


def test_stage2_no_cascade():
    edge_nodes, node_edges = _chain()
    load = np.array([1.0, 1.0, 1.0])
    alive = np.ones(3, dtype=bool)
    C = 3.0
    collapsed, lost = stage2_redistribute_cascade(
        edge_nodes, node_edges, load, alive, C, np.array([0])
    )
    # e0 坍塌，全部负载(1)给 e1 -> e1=2<=3 不级联
    assert set(collapsed) == {0}
    assert list(alive) == [False, True, True]
    assert load[1] == pytest.approx(2.0)
    assert lost == pytest.approx(0.0)


def test_stage2_full_cascade():
    edge_nodes, node_edges = _chain()
    load = np.array([3.0, 1.0, 1.0])
    alive = np.ones(3, dtype=bool)
    C = 3.0
    collapsed, lost = stage2_redistribute_cascade(
        edge_nodes, node_edges, load, alive, C, np.array([0])
    )
    # e0(3)->e1: 1+3=4>3 坍塌; e1(4)->e2: 1+4=5>3 坍塌 -> 全坍
    assert set(collapsed) == {0, 1, 2}
    assert not alive.any()


def test_stage2_load_lost_when_no_neighbor():
    # 孤立超边坍塌，无邻边 -> 负载丢失
    edge_nodes = np.array([[0, 1, 2]], dtype=np.int64)
    node_edges = hg.build_incidence(edge_nodes, 3)
    load = np.array([5.0])
    alive = np.ones(1, dtype=bool)
    collapsed, lost = stage2_redistribute_cascade(
        edge_nodes, node_edges, load, alive, 3.0, np.array([0])
    )
    assert set(collapsed) == {0}
    assert lost == pytest.approx(5.0)


def test_stage2_load_monotonic_for_survivors():
    # 分配比例：星形，中心节点连多条边
    # e0={0,1,2}, e1={2,3,4}, e2={2,5,6}; e0 与 e1、e2 各共享节点 2
    edge_nodes = np.array([[0, 1, 2], [2, 3, 4], [2, 5, 6]], dtype=np.int64)
    node_edges = hg.build_incidence(edge_nodes, 7)
    load = np.array([2.0, 1.0, 1.0])
    alive = np.ones(3, dtype=bool)
    before = load.copy()
    collapsed, lost = stage2_redistribute_cascade(
        edge_nodes, node_edges, load, alive, 3.0, np.array([0])
    )
    # e0 坍塌，负载2 平分给 e1,e2 各 1 -> 各 2 <=3 不级联
    assert set(collapsed) == {0}
    assert load[1] == pytest.approx(2.0)
    assert load[2] == pytest.approx(2.0)
    # 存活边负载不减
    for f in (1, 2):
        assert load[f] >= before[f]


def test_stage2_share_two_nodes_gets_double_weight():
    # m=2 超边 3 节点：e0={0,1,2} 与 e1={0,1,3} 共享 2 个节点（|e∩f|=2）
    # 与 e2={2,4,5} 共享 1 个节点。比例 2:1 → e0 负载 6 分 4 给 e1、2 给 e2。
    edge_nodes = np.array([[0, 1, 2], [0, 1, 3], [2, 4, 5]], dtype=np.int64)
    node_edges = hg.build_incidence(edge_nodes, 6)
    load = np.array([6.0, 1.0, 1.0])
    alive = np.ones(3, dtype=bool)
    collapsed, lost = stage2_redistribute_cascade(
        edge_nodes, node_edges, load, alive, 10.0, np.array([0])
    )
    assert set(collapsed) == {0}
    assert load[1] == pytest.approx(5.0)   # 1 + 6*2/3
    assert load[2] == pytest.approx(3.0)   # 1 + 6*1/3
    assert lost == pytest.approx(0.0)


def test_stage2_wave_sync_judges_after_full_contribution():
    # 波同步语义：同波两条坍塌边对同一条邻边 f 的贡献先加和再判过载。
    # e0,e1 各给 e2 加 1.5；f 初始 1 → 1+1.5+1.5=4 > C=3 → 波末坍塌。
    # 若逐条判（1+1.5=2.5<3）则不会坍塌——此测试锁定"先加后判"。
    edge_nodes = np.array([[0, 1, 2], [2, 3, 4], [2, 5, 6]], dtype=np.int64)
    node_edges = hg.build_incidence(edge_nodes, 7)
    load = np.array([1.5, 1.5, 1.0])
    alive = np.ones(3, dtype=bool)
    collapsed, lost = stage2_redistribute_cascade(
        edge_nodes, node_edges, load, alive, 3.0, np.array([0, 1])
    )
    assert set(collapsed) == {0, 1, 2}
    assert not alive.any()


def test_stage2_multi_wave_chain():
    # 4 边链多波传播：e0 坍塌 → e1 过载 → e2 过载 → e3 过载
    edge_nodes = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]], dtype=np.int64)
    node_edges = hg.build_incidence(edge_nodes, 9)
    load = np.array([3.0, 1.0, 1.0, 1.0])
    alive = np.ones(4, dtype=bool)
    collapsed, lost = stage2_redistribute_cascade(
        edge_nodes, node_edges, load, alive, 3.0, np.array([0])
    )
    assert set(collapsed) == {0, 1, 2, 3}
    assert not alive.any()


def test_stage2_no_cascade_at_huge_capacity():
    # C → ∞ 退化：只有初始坍塌边被移除，无级联
    edge_nodes, node_edges = _chain()
    load = np.array([100.0, 1.0, 1.0])
    alive = np.ones(3, dtype=bool)
    C = 1e9
    collapsed, lost = stage2_redistribute_cascade(
        edge_nodes, node_edges, load, alive, C, np.array([0])
    )
    assert set(collapsed) == {0}
    assert load[1] == pytest.approx(101.0)   # 负载照常重分配，只是不过载
    assert list(alive) == [False, True, True]


def test_stage2_overload_strictly_greater():
    # main.tex L81 判据是 L_f' > C（严格大于）：恰好等于 C 不坍塌
    edge_nodes, node_edges = _chain()
    load = np.array([2.0, 1.0, 1.0])
    alive = np.ones(3, dtype=bool)
    collapsed, lost = stage2_redistribute_cascade(
        edge_nodes, node_edges, load, alive, 3.0, np.array([0])
    )
    # e0(2) -> e1: 1+2=3 恰好等于 C → 不坍塌
    assert set(collapsed) == {0}
    assert load[1] == pytest.approx(3.0)
    assert alive[1] and alive[2]
