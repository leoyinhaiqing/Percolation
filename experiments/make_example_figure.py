"""
Illustrative example for the vital-node score k_i(1-b_i)  (revised, clearer).

A small m=2 hypergraph (10 nodes, 5 hyperedges of 3 nodes each).
Node 1 is the highest-degree node (k_1=3) but is already very vulnerable
(b_1=0.9), so its linear response k_1(1-b_1)=0.3 is small. Node 2 has a
smaller degree (k_2=2) but a much lower vulnerability (b_2=0.2), giving the
largest score k_2(1-b_2)=1.6.

Two layouts, same data and same topology:

  --layout wide    原始版本：跨栏（figure*），figsize 12.8x7.2，十个节点全部带
                   标签框，图例一行七项。
  --layout column  单栏版本（默认）：figsize 3.375x2.3，字号按 COL1 实尺寸给，
                   `\\includegraphics[width=\\columnwidth]` 时 1:1 不缩放。
                   只给四个取值互不相同的节点（1/2/3/7）画标签框——其余六个
                   节点的 k=1, b=0.5, k(1-b)=0.50 完全相同，逐个标注只是把
                   同一组数字重复六遍，在单栏尺寸下会把图挤到读不了；这六个
                   节点的取值写进图注即可。图例只留五条超边。

用法：
  .venv/Scripts/python.exe experiments/make_example_figure.py
  .venv/Scripts/python.exe experiments/make_example_figure.py --layout wide

Output: figures/fig7_vital_example.pdf (+ .png)
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch, Circle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
COL1 = 3.375                      # revtex 单栏宽度（英寸），与 make_figures.py 一致

# ----------------------------------------------------------------------
# Topology
# ----------------------------------------------------------------------
nodes = list(range(1, 11))
hyperedges = [(1, 2, 3), (1, 2, 4), (1, 3, 5), (6, 7, 8), (7, 9, 10)]

# ----------------------------------------------------------------------
# Vulnerability, hyperdegree, and score
# ----------------------------------------------------------------------
b = {1: 0.9, 2: 0.2, 3: 0.5, 4: 0.5, 5: 0.5,
     6: 0.5, 7: 0.6, 8: 0.5, 9: 0.5, 10: 0.5}
k = {1: 3, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 2, 8: 1, 9: 1, 10: 1}
score = {i: round(k[i] * (1 - b[i]), 2) for i in nodes}

# 取值互不相同的四个节点；其余六个都是 k=1, b=0.5, k(1-b)=0.50
DISTINCT = [1, 2, 3, 7]

# distinct colors for the five hyperedges (Tableau)
hyper_colors = ['#4e79a7', '#f28e2b', '#59a14f', '#b07aa1', '#e15759']
hyper_dark = ['#2f4b6e', '#9c5a12', '#35622e', '#6b4a6a', '#8c3233']

# ----------------------------------------------------------------------
# 两套尺寸参数：wide 是原始跨栏版，column 是单栏版
# ----------------------------------------------------------------------
STYLE = {
    "wide": dict(
        figsize=(12.8, 7.2), xlim=(-2.7, 3.1), ylim=(-1.9, 2.1), ellipse=1.45,
        r0=0.10, rk=0.035, ring=1.6, node_fs=13, label_fs=11, call_fs=12,
        lw_edge=1.8, lw_node=1.8, lw_ring=3.0, lw_arrow=1.6,
        legend_fs=9.5, legend_ncol=7, legend_y=-0.11, legend_rings=True,
        labels=nodes, pad=0.30,
        call_xy=((1.35, 1.85), (-2.4, 1.85)),
    ),
    "column": dict(
        figsize=(COL1, COL1 * 0.63), xlim=(-2.55, 2.95), ylim=(-1.52, 2.00),
        ellipse=1.45,
        r0=0.115, rk=0.030, ring=1.55, node_fs=6.0, label_fs=5.2, call_fs=5.4,
        lw_edge=0.9, lw_node=0.9, lw_ring=1.5, lw_arrow=0.8,
        legend_fs=5.0, legend_ncol=5, legend_y=-0.085, legend_rings=False,
        labels=DISTINCT, pad=0.22,
        call_xy=((1.15, 1.82), (-2.35, 1.82)),
    ),
}

# per-node label anchors (hand-placed to avoid overlap)
offsets = {
    1: (0.00, 0.42, 'center'),
    2: (0.52, 0.12, 'left'),
    3: (0.58, 0.00, 'left'),
    4: (0.58, -0.04, 'left'),
    5: (0.52, -0.12, 'left'),
    6: (0.00, -0.42, 'center'),
    7: (-0.52, -0.12, 'right'),
    8: (-0.58, -0.04, 'right'),
    9: (-0.58, 0.00, 'right'),
    10: (-0.52, 0.12, 'right'),
}


def draw(layout: str) -> None:
    S = STYLE[layout]

    # ellipse layout (wider horizontally so labels have room), node 1 at top
    angles = np.linspace(90.0, 90.0 - 360.0, len(nodes), endpoint=False)
    pos = {i: (S["ellipse"] * np.cos(np.radians(a)), np.sin(np.radians(a)))
           for i, a in zip(nodes, angles)}

    fig, ax = plt.subplots(figsize=S["figsize"])
    ax.set_aspect('equal')
    ax.set_xlim(*S["xlim"])
    ax.set_ylim(*S["ylim"])
    ax.axis('off')

    # hyperedges: distinct saturated fill + thick edge
    for (he, fc, ec) in zip(hyperedges, hyper_colors, hyper_dark):
        pts = [pos[i] for i in he]
        cx, cy = np.mean(pts, axis=0)
        pts = sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        ax.add_patch(Polygon(pts, closed=True, alpha=0.18, facecolor=fc,
                             edgecolor=ec, lw=S["lw_edge"], zorder=1))

    # nodes: white fill, dark border, radius proportional to score
    node_radius = {i: S["r0"] + S["rk"] * score[i] for i in nodes}
    for i in nodes:
        x, y = pos[i]
        ax.add_patch(Circle((x, y), node_radius[i], facecolor='white',
                            edgecolor='#333333', linewidth=S["lw_node"],
                            zorder=3))
        ax.text(x, y, str(i), ha='center', va='center', fontsize=S["node_fs"],
                color='#111111', fontweight='bold', zorder=4)

    # highlight rings: consistent scaling relative to node radius
    x2, y2 = pos[2]
    ax.add_patch(Circle((x2, y2), node_radius[2] * S["ring"], fill=False,
                        edgecolor='#1d9e75', lw=S["lw_ring"], zorder=3.5))
    x1, y1 = pos[1]
    ax.add_patch(Circle((x1, y1), node_radius[1] * S["ring"], fill=False,
                        edgecolor='#d85a30', lw=S["lw_ring"], zorder=3.5))

    # per-node labels on an opaque white background
    for i in S["labels"]:
        x, y = pos[i]
        dx, dy, ha = offsets[i]
        txt = f"$k$={k[i]},  $b$={b[i]}\n$k(1-b)$={score[i]:.2f}"
        ax.text(x + dx, y + dy, txt, ha=ha, va='center',
                fontsize=S["label_fs"], color='#111111',
                bbox=dict(boxstyle=f'round,pad={S["pad"]}', fc='white',
                          ec='#999999', lw=0.6, alpha=0.95))

    # 单栏版只画四个取值互不相同的节点的标签框，其余六个的公共取值在这里说明，
    # 信息不丢，但不用把同一组数字重复六遍。
    if len(S["labels"]) < len(nodes):
        same = sorted(set(nodes) - set(S["labels"]))
        ax.text(S["xlim"][0] + 0.05, S["ylim"][0] + 0.12,
                "nodes " + ", ".join(str(i) for i in same)
                + ":  $k$=1,  $b$=0.5,  $k(1-b)$=0.50",
                ha='left', va='bottom', fontsize=S["label_fs"],
                color='#444444')

    # callouts for the two highlighted nodes
    ax.annotate('largest $k_i(1-b_i)$',
                xy=(x2 + 0.20, y2 + 0.20), xytext=S["call_xy"][0],
                fontsize=S["call_fs"], color='#0f6e56', fontweight='bold',
                ha='left',
                arrowprops=dict(arrowstyle='->', color='#0f6e56',
                                lw=S["lw_arrow"],
                                connectionstyle='arc3,rad=-0.1'))
    ax.annotate('largest $k_i$, smallest $k_i(1-b_i)$',
                xy=(x1 - 0.15, y1 + 0.15), xytext=S["call_xy"][1],
                fontsize=S["call_fs"], color='#993c1d', fontweight='bold',
                ha='left',
                arrowprops=dict(arrowstyle='->', color='#993c1d',
                                lw=S["lw_arrow"],
                                connectionstyle='arc3,rad=0.1'))

    # legend for hyperedges (and, in the wide layout, the highlight rings)
    handles = [Patch(facecolor=fc, edgecolor=ec, lw=S["lw_edge"],
                     label=f"$e_{{{he[0]},{he[1]},{he[2]}}}$")
               for (he, fc, ec) in zip(hyperedges, hyper_colors, hyper_dark)]
    if S["legend_rings"]:
        handles += [
            Patch(facecolor='none', edgecolor='#1d9e75', lw=S["lw_ring"],
                  label='largest $k_i(1-b_i)$ (node 2)'),
            Patch(facecolor='none', edgecolor='#d85a30', lw=S["lw_ring"],
                  label='largest $k_i$, smallest $k_i(1-b_i)$ (node 1)'),
        ]
    ax.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(0.5, S["legend_y"]), ncol=S["legend_ncol"],
              frameon=False, fontsize=S["legend_fs"], columnspacing=0.9,
              handlelength=1.4, handletextpad=0.4)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(os.path.join(FIGDIR, f"fig7_vital_example.{ext}"),
                    bbox_inches='tight', **kw)
    plt.close(fig)
    print(f"saved figures/fig7_vital_example.pdf + .png  (layout={layout})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", choices=("column", "wide"), default="column")
    args = ap.parse_args()
    draw(args.layout)

    print()
    print('node  : ' + '  '.join(f'{i:>2}' for i in nodes))
    print('k     : ' + '  '.join(f'{k[i]:>2}' for i in nodes))
    print('b     : ' + '  '.join(f'{b[i]:>2}' for i in nodes))
    print('score : ' + '  '.join(f'{score[i]:>4.2f}' for i in nodes))
