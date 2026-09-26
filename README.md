# Cascading failures on temporal higher-order networks

Reference implementation, mean-field theory and vital-node identification code for

> Zihao Song, Haiqing Yin and Xiao-Dong Zhang,
> *Dynamic topology and percolation criticality in higher-order
> activity–vulnerability driven networks.*

The model puts a dynamic load-redistribution cascade on a temporal hypergraph:
hyperedges fail spontaneously with a probability set by the vulnerability of
their constituent nodes, and the load released by a failed hyperedge triggers
overload-induced failures in the hyperedges that absorb it. This repository
contains the event-driven simulator, the closed-form mean-field predictions for
the collapse time `T_c`, the vital-node ranking experiments, and the data and
scripts that produce every figure in the paper.

## Install

Python 3.11 with the pinned dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
```

Run everything from the repository root, so that `src` is importable.

```bash
python -m pytest tests/ -q
```

`tests/test_realdata.py` skips itself unless the real dataset is present (see
*Real data* below); the rest of the suite is self-contained.

## Layout

| Path | Contents |
|---|---|
| `src/config.py` | `Config` dataclass: `N`, `m`, `gamma`, `k_min`, `C`, `b_dist`, `eps`, `seed`, … |
| `src/hypergraph.py` | power-law configuration-model hypergraphs, incidence tables, moments, giant component |
| `src/cascade.py` | Stage-1 spontaneous failure and Stage-2 synchronous load redistribution |
| `src/dynamics.py` | event-driven main loop with Gillespie jumps, records `S(t)`, `S_h(t)`, `L(t)`, `T_c` |
| `src/fastsim.py` | quenched-clock engine: exact simulation of the discrete model with sparse cascades and common random numbers — required for the large sizes |
| `src/theory.py` | mean-field formulas: `rho_c`, `B(t)`, `S_h^1`/`S_h^E`, the Laplace-corrected trigger time `t^1_L`, and the two `T_c` branches |
| `src/centrality.py` | static baselines on the clique expansion: hyperdegree, betweenness, PageRank |
| `src/realdata.py` | reader for Benson-format node-labelled hypergraphs, plus `m`-uniform sub-hypergraph extraction |
| `experiments/` | driver scripts (below) |
| `results/` | the measured data behind the figures and tables |
| `figures/` | the figures as they appear in the paper, `.pdf` for the manuscript and `.png` for preview |

## Reproducing the figures

`results/` already holds the measured data, so the figure scripts run in
seconds without repeating the simulations:

```bash
python experiments/make_figures.py          # Fig. 1-4
python experiments/make_param_figures.py    # Fig. 5
python experiments/make_vital_figures.py --tag house-bills_m2_hetb   # Fig. 6
python experiments/make_example_figure.py   # Fig. 7
```

| Figure | Script | Data |
|---|---|---|
| 1 `T_c` simulation vs theory | `make_figures.py` | `results/tail_diag_m2_kmin3*.csv` |
| 2 relative error and its decomposition | `make_figures.py` | same |
| 3 calibration factor `lambda(N)` | `make_figures.py` | same |
| 4 density trajectories | `make_figures.py` | simulated on the fly (`N=2000`, `seed=1`) |
| 5 parameter dependence | `make_param_figures.py` | `results/param_sweep_m2_kmin3.csv` |
| 6 vital nodes on House bills | `make_vital_figures.py` | `results/vital_house-bills_m2_hetb_*.csv` |
| 7 worked example of the score | `make_example_figure.py` | self-contained |

The schematic in the paper's Fig. 8 was drawn separately and is not produced by
this repository.

## Regenerating the data

The eight sizes `N = 500 … 10^5` at two capacities, 15 realizations each, which
back Figs. 1-3 and the error tables. The largest sizes need the quenched-clock
engine.

```bash
python experiments/run_tail_diag.py --Ns 500 1000 2000 5000 10000 --seeds 15
python experiments/run_tail_diag.py --Ns 20000 --seeds 15 \
    --out tail_diag_m2_kmin3_N20000.csv
python experiments/run_tail_diag.py --Ns 100000 --seeds 15 --engine quenched \
    --out tail_diag_m2_kmin3_N100000_quenched.csv

python experiments/analyze_tail_diag.py --csv tail_diag_m2_kmin3.csv \
    tail_diag_m2_kmin3_N20000.csv tail_diag_m2_kmin3_N50000_quenched.csv \
    tail_diag_m2_kmin3_N100000_quenched.csv
```

The five-parameter sweep (`C`, `m`, `<k>_0`, `N`, `<b>`) behind Fig. 5, the
`N x C` sweep, and the vital-node runs:

```bash
python experiments/run_param_sweep.py --sweeps C m k0 N b --seeds 10
python experiments/run_sweep.py --k_min 3 --repeat 10 --Ns 500 1000 2000
python experiments/run_vital_nodes.py --N 1000 --b_dist uniform --no-greedy \
    --tag synthetic_N1000_m2_hetb
```

Consistency checks:

```bash
python experiments/validate_fastsim.py --Ns 500 1000 2000 --seeds 15
python experiments/compare_event_vs_stepwise.py
python experiments/run_step1.py --N 1000 --m 2 --seed 1
```

Every run takes an explicit `--seed` and records it, so results are
reproducible.

## Real data

Fig. 6 uses the `m=2` sub-hypergraph of the U.S. House of Representatives bill
co-sponsorship hypergraph (`N = 1494`, 4369 hyperedges) from

> A. R. Benson, R. Abebe, M. T. Schaub, A. Jadbabaie and J. Kleinberg,
> *Simplicial closure and higher-order link prediction*,
> PNAS **115**, E11221 (2018).

The dataset is not redistributed here. Download `house-bills` from
<https://www.cs.cornell.edu/~arb/data/> and place the archive at the repository
root as `house-bills.zip`; `src/realdata.py` reads it from there. Then

```bash
python experiments/run_vital_nodes.py --dataset house-bills --m 2 \
    --b_dist uniform --no-greedy --tag house-bills_m2_hetb
```

## Notes on the source

Docstrings and inline comments are written in Chinese. Two kinds of marker in
them point outside this repository: `main.tex L###` refers to a line of the
manuscript, and `findings <circled number>` refers to an entry in the authors'
internal experiment log. Neither file is distributed here; the comments are
self-contained without them.

## License

MIT, see [LICENSE](LICENSE).
