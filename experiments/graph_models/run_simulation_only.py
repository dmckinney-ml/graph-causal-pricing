"""
Run only the semi-synthetic simulation study, skipping the initial DML fits.
Use this to resume after the initial graph_model_results/effect_comparison/
cate_by_department CSVs have already been written.

Usage:
    cd graph-causal-pricing
    nohup poetry run python -m experiments.graph_models.run_simulation_only \
      > results/run_simulation_only.log 2>&1 &
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from src.data.splits import create_temporal_split
from src.causal.graph_dml import run_graph_dml
from src.evaluation.synthetic_dgp import SyntheticDGPConfig, run_simulation_study
from src.evaluation.metrics import full_synthetic_evaluation


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "causal_config.yaml").read_text())
    seed = cfg["seed"]

    # Load data
    panel = pd.read_parquet(ROOT / cfg["data"]["panel_path"])
    A_sub = sp.load_npz(str(ROOT / cfg["data"]["a_sub_path"]))
    A_comp = sp.load_npz(str(ROOT / cfg["data"]["a_comp_path"]))
    pid_idx = np.load(ROOT / cfg["data"]["pid_idx_path"])
    embeddings = np.load(ROOT / cfg["data"]["embeddings_path"])

    # Temporal split
    splits = create_temporal_split(
        panel,
        train_weeks=tuple(cfg["split"]["train_weeks"]),
        val_weeks=tuple(cfg["split"]["val_weeks"]),
        test_weeks=tuple(cfg["split"]["test_weeks"]),
    )
    train_panel = splits["train"]

    # Subsample — must match the same seed/max_rows as the original run
    max_rows = 6_000_000
    if len(train_panel) > max_rows:
        train_panel = train_panel.sample(n=max_rows, random_state=seed)
        print(f"Subsampled to {max_rows:,} rows (from {splits['train'].shape[0]:,})")

    # Free the full panel immediately — we only need train_panel from here
    del panel, splits
    gc.collect()

    print(f"Running simulation study on {len(train_panel):,} observations")

    out_dir = ROOT / cfg["output"]["tables_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    dgp_cfg = SyntheticDGPConfig(
        true_direct_effect=cfg["synthetic"]["true_direct_effect"],
        true_sub_spillover=cfg["synthetic"]["true_sub_spillover"],
        true_comp_spillover=cfg["synthetic"]["true_comp_spillover"],
        noise_std=cfg["synthetic"]["noise_std"],
        confounding_strength=cfg["synthetic"]["confounding_strength"],
        baseline_demand=cfg["synthetic"].get("baseline_demand", 1.5),
        log_space=cfg["dml"]["log_transform"],
        seed=seed,
    )

    def estimator_fn(sim_panel):
        sim_result = run_graph_dml(
            panel=sim_panel,
            A_sub=A_sub, sub_pid_idx=pid_idx,
            A_comp=A_comp, comp_pid_idx=pid_idx,
            outcome_col="y_synthetic",
            treatment_col=cfg["treatment"]["primary"],
            weight_type="edge_weight",
            n_folds=cfg["dml"]["n_folds"],
            embeddings=embeddings, emb_pid_idx=pid_idx,
            confidence_level=cfg["dml"]["confidence_level"],
            seed=seed,
            log_transform=cfg["dml"]["log_transform"],
            min_demand_filter=cfg["dml"]["min_demand_filter"],
            include_discount_rate=cfg["dml"].get("include_discount_rate", False),
        )
        return {
            "direct":        (sim_result.direct_ate,       *sim_result.direct_ci),
            "sub_spillover": (sim_result.sub_spillover_ate, *sim_result.sub_spillover_ci),
            "comp_spillover":(sim_result.comp_spillover_ate,*sim_result.comp_spillover_ci),
        }

    print(f"\nRunning semi-synthetic simulation study "
          f"({cfg['synthetic']['n_simulations']} sims)...")

    sim_results = run_simulation_study(
        panel=train_panel,
        A_sub=A_sub, sub_pid_idx=pid_idx,
        A_comp=A_comp, comp_pid_idx=pid_idx,
        estimator_fn=estimator_fn,
        config=dgp_cfg,
        n_simulations=cfg["synthetic"]["n_simulations"],
    )

    eval_summary = full_synthetic_evaluation(sim_results)
    eval_summary.to_csv(out_dir / "synthetic_evaluation.csv", index=False)

    print("\n— Semi-synthetic evaluation —")
    print(eval_summary.to_string(index=False))
    print(f"\nResults saved to {out_dir / 'synthetic_evaluation.csv'}")


if __name__ == "__main__":
    main()
