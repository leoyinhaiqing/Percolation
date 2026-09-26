"""真实数据集读取与 m-均匀子超图抽取的测试（house-bills 存在时才跑）。"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src import theory
from src.hypergraph import build_incidence, hyperdegree_moments
from src.realdata import load_hyperedges, load_uniform_sub_hypergraph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_HB = os.path.exists(os.path.join(ROOT, "house-bills.zip"))
skip_hb = pytest.mark.skipif(not HAS_HB, reason="house-bills.zip 不在仓库根目录")


@skip_hb
def test_load_hyperedges_shape_and_indexing():
    edges, N, meta = load_hyperedges("house-bills")
    assert meta["dataset"] == "house-bills"
    assert N == 1494                                  # node-names/labels 各 1494 行
    assert len(edges) == meta["n_edges_raw"]
    assert len(set(edges)) == len(edges)              # 已去重
    flat = [v for e in edges for v in e]
    assert min(flat) == 0 and max(flat) <= N - 1      # 已转成 0-based
    for e in edges[:200]:
        assert list(e) == sorted(set(e))              # 每条内部去重且升序


@skip_hb
@pytest.mark.parametrize("m", [2, 3])
def test_uniform_sub_hypergraph_is_exactly_m_uniform(m):
    edge_nodes, N, meta = load_uniform_sub_hypergraph("house-bills", m)
    assert edge_nodes.ndim == 2 and edge_nodes.shape[1] == m + 1
    assert meta["m"] == m and meta["n_edges_m"] == len(edge_nodes)
    assert len(np.unique(edge_nodes, axis=0)) == len(edge_nodes)
    for row in edge_nodes:
        assert len(set(row.tolist())) == m + 1        # 无重复节点


@skip_hb
def test_house_bills_m2_lands_in_the_solvable_regime():
    """m=2 子超图必须满足理论的有效性条件，否则这个应用无从谈起。"""
    edge_nodes, N, _ = load_uniform_sub_hypergraph("house-bills", 2)
    k0, k2 = hyperdegree_moments(build_incidence(edge_nodes, N))
    rc = theory.rho_c(k0, k2, 2)
    assert theory.has_initial_gc(k0, k2, 2)           # 初始有巨分量
    assert rc < theory.S_h_E(k0)                      # rho_c < S_h^E：可解条件
    C_star = theory.C_threshold(k0, 2)
    assert theory.regime(k0, 2, 0.75 * C_star) == "case_II"
    assert theory.regime(k0, 2, 1.5 * C_star) == "case_I"
    for frac in (0.75, 1.5):
        assert 0.0 < theory.S_h_1star(k0, 2, frac * C_star) < 1.0


def test_missing_dataset_raises():
    with pytest.raises(FileNotFoundError):
        load_hyperedges("definitely-not-a-dataset")
