"""
Run all baseline causal estimators and save results.

Usage:
    cd graph-causal-pricing
    python experiments/baseline_models/run_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data.splits import create_temporal_split
from src.models.baselines import run_all_baselines, results_to_dataframe


def main():
    # Load config
    cfg = yaml.safe_load((ROOT / "configs" / "causal_config.yaml").read_text())
    seed = cfg["seed"]
    n_folds = cfg["dml"]["n_folds"]
    confidence_level = cfg["dml"]["confidence_level"]

    # Load data
    panel = pd.read_parquet(ROOT / cfg["data"]["panel_path"])
    embeddings = np.load(ROOT / cfg["data"]["embeddings_path"])
    pid_idx = np.load(ROOT / cfg["data"]["pid_idx_path"])

    # Temporal split — train on train set, evaluate on test
    splits = create_temporal_split(
        panel,
        train_weeks=tuple(cfg["split"]["train_weeks"]),
        val_weeks=tuple(cfg["split"]["val_weeks"]),
        test_weeks=tuple(cfg["split"]["test_weeks"]),
    )
    train_panel = splits["train"]

    # Apply log transform to match graph DML outcome scale
    if cfg["dml"]["log_transform"]:
        train_panel = train_panel.copy()
        train_panel[cfg["outcome"]["primary"]] = np.log1p(
            train_panel[cfg["outcome"]["primary"]]
        )
        print("Applied log(1+Y) transform to outcome variable")

    print(f"Training baselines on {len(train_panel):,} observations "
          f"(weeks {cfg['split']['train_weeks']})")

    # Run all baselines
    results = run_all_baselines(
        panel=train_panel,
        outcome_col=cfg["outcome"]["primary"],
        treatment_col=cfg["treatment"]["primary"],
        n_folds=n_folds,
        embeddings=embeddings,
        emb_pid_idx=pid_idx,
        confidence_level=confidence_level,
        seed=seed,
        include_discount_rate=cfg["dml"].get("include_discount_rate", False),
    )

    # Save results
    out_dir = ROOT / cfg["output"]["tables_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = results_to_dataframe(results)
    df.to_csv(out_dir / "baseline_results.csv", index=False)
    print(f"\nResults saved to {out_dir / 'baseline_results.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
