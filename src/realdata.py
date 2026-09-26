"""真实高阶网络数据集的读取与 m-均匀子超图抽取。

数据放在仓库根目录的 `<name>.zip`（Benson 的 node-labeled hypergraph 格式：
`hyperedges-<name>.txt` 每行一条超边、逗号分隔的 1-based 节点编号；
`node-names-` / `node-labels-` / `label-names-` 为可选元数据）。

模型是 (m+1)-均匀的，而真实数据的超边尺寸从 2 到几百都有，因此按 main.tex
\\S numerical-simulation 的规定「for a fixed hyperedge order m, we extract all
m-order hyperedges from a given real higher-order network H」抽取尺寸恰为 m+1 的
子超图。不做跨尺寸的统一化：p_e = m!/N^m·Σb 的 N^{-m} 因子是 m 的函数，把不同
尺寸混在一起会让小超边的坍塌率高出若干个数量级，那已经是另一个模型了。

节点编号一律重映射到 0..N-1（N = 原始节点总数，保留孤立点以免改变 <k>_0 口径）。
"""
from __future__ import annotations

import os
import zipfile
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_member(z: zipfile.ZipFile, name: str, dataset: str) -> str | None:
    for cand in (f"{dataset}/{name}-{dataset}.txt", f"{name}-{dataset}.txt"):
        if cand in z.namelist():
            return z.read(cand).decode("utf-8", "replace")
    return None


def load_hyperedges(dataset: str) -> Tuple[List[Tuple[int, ...]], int, Dict]:
    """读全部超边（去重、每条内部去重并升序），返回 (edges, N, meta)。"""
    path = os.path.join(ROOT, f"{dataset}.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到数据集压缩包：{path}")
    with zipfile.ZipFile(path) as z:
        raw = _read_member(z, "hyperedges", dataset)
        if raw is None:
            raise ValueError(f"{dataset}.zip 内没有 hyperedges 文件")
        names = _read_member(z, "node-names", dataset)
        labels = _read_member(z, "node-labels", dataset)
        label_names = _read_member(z, "label-names", dataset)

    edges = sorted({tuple(sorted({int(x) - 1 for x in line.split(",")}))
                    for line in raw.splitlines() if line.strip()})
    n_from_edges = max(max(e) for e in edges) + 1
    node_names = names.splitlines() if names else []
    node_labels = [int(x) for x in labels.split()] if labels else []
    N = max(n_from_edges, len(node_names), len(node_labels))

    sizes = [len(e) for e in edges]
    meta = dict(
        dataset=dataset, n_edges_raw=len(edges), N=N,
        size_min=min(sizes), size_max=max(sizes),
        size_mean=float(np.mean(sizes)),
        node_names=node_names, node_labels=node_labels,
        label_names=label_names.splitlines() if label_names else [],
    )
    return edges, N, meta


def load_uniform_sub_hypergraph(dataset: str, m: int) -> Tuple[np.ndarray, int, Dict]:
    """抽取尺寸恰为 m+1 的子超图，返回 (edge_nodes[(E,m+1)], N, meta)。"""
    edges, N, meta = load_hyperedges(dataset)
    sub = [e for e in edges if len(e) == m + 1]
    if not sub:
        raise ValueError(f"{dataset} 中没有 {m + 1}-节点超边（m={m}）")
    edge_nodes = np.array(sub, dtype=np.int64)
    meta = dict(meta, m=m, n_edges_m=len(sub),
                n_nodes_touched=len(np.unique(edge_nodes)))
    return edge_nodes, N, meta
