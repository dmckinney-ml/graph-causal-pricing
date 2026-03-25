# Graph Causal Pricing

![Research Status](https://img.shields.io/badge/status-complete-brightgreen)
![Dataset](https://img.shields.io/badge/dataset-Dunnhumby%20Complete%20Journey-green)
![Task](https://img.shields.io/badge/task-retail%20promotion%20spillovers-orange)
![Models](https://img.shields.io/badge/models-LinearDML%20%7C%20CausalForestDML%20%7C%20LightGBM-purple)
![Preprint](https://img.shields.io/badge/preprint-not%20peer%20reviewed-yellow)

Graph-aware causal estimation for retail promotions and pricing spillovers.

**Preprint. Not yet peer reviewed.** | [Read the paper (PDF)](paper/main.pdf)

This repository studies how a promotion on one product changes demand not only for that product, but also for nearby substitutes and complements. The core idea is to build a product graph, convert neighbor promotions into exposure variables, and estimate direct and spillover effects with Double Machine Learning (DML).

The project is built around the Dunnhumby Complete Journey retail dataset and focuses on a graph-aware DML pipeline, baseline causal estimators, ablations, and a semi-synthetic validation study.

Because the graph-aware DML pipeline is memory-intensive, the main training runs were executed on a Google Cloud VM: `n4-highmem-16` (16 vCPUs, 128 GB memory).

## What the project does

- Builds substitute and complement product graphs from retail transactions
- Computes one-hop exposure mappings for neighboring promotions
- Estimates direct, substitute-spillover, and complement-spillover effects
- Benchmarks graph-aware DML against no-graph baselines
- Runs ablations on edge weighting, graph channels, and embeddings
- Evaluates recovery of known effects in a semi-synthetic simulation study

## Repository layout

- `configs/causal_config.yaml`: experiment configuration, paths, split ranges, estimands, and DML settings
- `src/data/`: panel construction and temporal splits
- `src/graph/`: graph construction utilities for co-purchase, substitutes, complements, and spillover features
- `src/causal/`: exposure mapping, graph-aware DML, and interpretation helpers
- `src/models/`: baseline estimators
- `src/evaluation/`: metrics and semi-synthetic data-generating process
- `experiments/graph_models/`: main graph-aware DML and simulation entrypoints
- `experiments/baseline_models/`: baseline benchmark script
- `experiments/ablations/`: ablation runner and plotting script
- `notebooks/`: exploratory and development notebooks
- `paper/`: manuscript source

## Environment setup

This project uses Poetry and targets Python 3.13.

For reproducibility, note that the full experiment pipeline was run on a high-memory GCP instance. Smaller local machines may need to rely on the built-in subsampling paths in the experiment scripts.

```bash
poetry install
poetry shell
```

If you prefer one-off commands instead of opening a Poetry shell:

```bash
poetry run python experiments/graph_models/run_graph_dml.py
```

## Data and expected inputs

The configuration expects processed artifacts under `data/processed/`, including:

- `panel.parquet`
- `A_sub.npz`
- `A_comp.npz`
- `embeddings.npy`
- `pid_idx.npy`

Raw source files live under `data/raw/`. Both raw and processed data directories are ignored by git.

## Main experiment workflow

Run the graph-aware model:

```bash
poetry run python experiments/graph_models/run_graph_dml.py
```

Run baseline estimators:

```bash
poetry run python experiments/baseline_models/run_baselines.py
```

Run ablations:

```bash
poetry run python experiments/ablations/run_ablations.py
```

Plot ablation results:

```bash
poetry run python experiments/ablations/plot_ablations.py
```

Resume only the semi-synthetic study after the main graph outputs already exist:

```bash
poetry run python experiments/graph_models/run_simulation_only.py
```

## Outputs

Generated tables are written to `results/tables/` and figures to `results/figures/`. Common outputs include:

- `graph_model_results.csv`
- `effect_comparison.csv`
- `cate_by_department.csv`
- `baseline_results.csv`
- `ablation_results.csv`
- `synthetic_evaluation.csv`

These result directories are ignored by git so local experiment runs do not pollute version control.

## Current modeling choices

- Panel grain: `(PRODUCT_ID, STORE_ID, WEEK_NO)`
- Primary treatment: `promo_any`
- Secondary treatment: `discount_rate`
- Primary outcome: `log(1 + units_sold)`
- Exposure mapping: one-hop weighted neighbor treatment share
- Nuisance models: LightGBM
- Final estimator: linear DML with two-way cluster-robust standard errors
- Cross-fitting: 5 folds grouped by `PRODUCT_ID`
- Main training runs: subsampled to 6M rows for memory safety

## Current empirical takeaway

The main result is that product-relationship structure improves estimation of downstream treatment effects. In this project, the graph matters in two ways:

- it defines the spillover exposures for substitutes and complements
- its embedding representation provides rich product-level controls that materially shift treatment-effect estimates in ablations

In the current manuscript, the graph-aware model finds a positive direct promotion effect, a statistically significant negative substitute spillover, and a weaker but directionally positive complement effect that becomes clearer under continuous discount depth.

## Limitations

- **One-hop neighborhood aggregation.** Exposure variables summarize only immediate graph neighbors. Multi-hop propagation is not modeled.
- **Binary and continuous treatments are estimated separately.** The main `promo_any` model and the `discount_rate` model are fit independently.
- **Static product graphs.** Product relationships are estimated once from transaction history and do not vary over time.
- **Observational identification risk.** The design relies on conditional ignorability given observed product, store, time, and embedding-based controls.
- **Store-week confounding remains a key threat.** Coordinated store-week promotions on related products may induce correlated assignment and outcomes not fully captured by observed controls.
- **Direct-effect uncertainty is imperfectly calibrated.** The semi-synthetic study shows undercoverage for the direct-effect confidence intervals.

## Paper

The manuscript source is in `paper/main.tex`, with references in `paper/refs.bib`.
