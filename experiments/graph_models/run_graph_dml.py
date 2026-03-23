"""
Run the graph-aware exposure-mapping DML and save results.

Usage:
    cd graph-causal-pricing
    python experiments/graph_models/run_graph_dml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from src.data.splits import create_temporal_split
from src.causal.graph_dml import run_graph_dml, graph_result_to_dataframe
from src.causal.interpretability import cate_by_group, compare_effects
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

    print(f"Running graph-aware DML on {len(train_panel):,} observations")

    # Run with edge weights (the main model)
    result = run_graph_dml(
        panel=train_panel,
        A_sub=A_sub,
        sub_pid_idx=pid_idx,
        A_comp=A_comp,
        comp_pid_idx=pid_idx,
        outcome_col=cfg["outcome"]["primary"],
        treatment_col=cfg["treatment"]["primary"],
        weight_type="edge_weight",
        n_folds=cfg["dml"]["n_folds"],
        embeddings=embeddings,
        emb_pid_idx=pid_idx,
        confidence_level=cfg["dml"]["confidence_level"],
        seed=seed,
        log_transform=cfg["dml"]["log_transform"],
        min_demand_filter=cfg["dml"]["min_demand_filter"],
        include_discount_rate=cfg["dml"].get("include_discount_rate", False),
    )

    # Also run with binary weights for comparison
    result_binary = run_graph_dml(
        panel=train_panel,
        A_sub=A_sub,
        sub_pid_idx=pid_idx,
        A_comp=A_comp,
        comp_pid_idx=pid_idx,
        outcome_col=cfg["outcome"]["primary"],
        treatment_col=cfg["treatment"]["primary"],
        weight_type="binary",
        n_folds=cfg["dml"]["n_folds"],
        embeddings=embeddings,
        emb_pid_idx=pid_idx,
        confidence_level=cfg["dml"]["confidence_level"],
        seed=seed,
        log_transform=cfg["dml"]["log_transform"],
        min_demand_filter=cfg["dml"]["min_demand_filter"],
        include_discount_rate=cfg["dml"].get("include_discount_rate", False),
    )

    # Save results
    out_dir = ROOT / cfg["output"]["tables_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df_weighted = graph_result_to_dataframe(result)
    df_binary = graph_result_to_dataframe(result_binary)
    df_all = pd.concat([df_weighted, df_binary], ignore_index=True)
    df_all.to_csv(out_dir / "graph_model_results.csv", index=False)

    # Effect comparison
    effects = compare_effects(result)
    effects.to_csv(out_dir / "effect_comparison.csv", index=False)

    # CATE by department
    cate_dept = cate_by_group(train_panel, result, group_col="DEPARTMENT")
    cate_dept.to_csv(out_dir / "cate_by_department.csv")

    # Semi-synthetic simulation study
    print("\nRunning semi-synthetic simulation study "
          f"({cfg['synthetic']['n_simulations']} sims on full training panel)...")

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

    print(f"\nResults saved to {out_dir}")
    print("\n— Effect comparison (edge-weighted) —")
    print(effects.to_string(index=False))
    print("\n— Graph model results —")
    print(df_all.to_string(index=False))


if __name__ == "__main__":
    main()
