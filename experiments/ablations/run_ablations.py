"""
Ablation experiments to quantify the contribution of each modeling choice.

Ablations:
    1. No graph features at all       (≈ dml_no_graph baseline)
    2. Substitutes only               (drop complement exposure)
    3. Complements only               (drop substitute exposure)
    4. Binary vs edge-weighted        (remove PPMI/cosine weights)
    5. No embeddings                  (graph-aware DML without Node2Vec)

Usage:
    cd graph-causal-pricing
    python experiments/ablations/run_ablations.py
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


def _zero_matrix(A: sp.spmatrix) -> sp.csr_matrix:
    """Return a zero sparse matrix with the same shape as A."""
    return sp.csr_matrix(A.shape)


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "causal_config.yaml").read_text())
    seed = cfg["seed"]

    panel = pd.read_parquet(ROOT / cfg["data"]["panel_path"])
    A_sub = sp.load_npz(str(ROOT / cfg["data"]["a_sub_path"]))
    A_comp = sp.load_npz(str(ROOT / cfg["data"]["a_comp_path"]))
    pid_idx = np.load(ROOT / cfg["data"]["pid_idx_path"])
    embeddings = np.load(ROOT / cfg["data"]["embeddings_path"])

    splits = create_temporal_split(
        panel,
        train_weeks=tuple(cfg["split"]["train_weeks"]),
        val_weeks=tuple(cfg["split"]["val_weeks"]),
        test_weeks=tuple(cfg["split"]["test_weeks"]),
    )
    train = splits["train"]

    max_rows = 6_000_000
    if len(train) > max_rows:
        train = train.sample(n=max_rows, random_state=seed)
        print(f"Subsampled to {max_rows:,} rows (from {len(splits['train']):,})")

    n_folds = cfg["dml"]["n_folds"]
    cl = cfg["dml"]["confidence_level"]

    common = dict(
        outcome_col=cfg["outcome"]["primary"],
        treatment_col=cfg["treatment"]["primary"],
        n_folds=n_folds,
        confidence_level=cl,
        seed=seed,
        log_transform=cfg["dml"]["log_transform"],
        min_demand_filter=cfg["dml"]["min_demand_filter"],
        include_discount_rate=cfg["dml"].get("include_discount_rate", False),
    )

    results: list[pd.DataFrame] = []

    # ── 1. Full model (edge-weighted, with embeddings) ────────────────────────
    print("1/5 Full model (edge-weighted + embeddings) …")
    r = run_graph_dml(
        panel=train, A_sub=A_sub, sub_pid_idx=pid_idx,
        A_comp=A_comp, comp_pid_idx=pid_idx,
        weight_type="edge_weight", embeddings=embeddings, emb_pid_idx=pid_idx,
        **common,
    )
    df = graph_result_to_dataframe(r)
    df["ablation"] = "full_model"
    results.append(df)

    # ── 2. Substitutes only (zero out complement adjacency) ──────────────────
    print("2/5 Substitutes only …")
    r = run_graph_dml(
        panel=train, A_sub=A_sub, sub_pid_idx=pid_idx,
        A_comp=_zero_matrix(A_comp), comp_pid_idx=pid_idx,
        weight_type="edge_weight", embeddings=embeddings, emb_pid_idx=pid_idx,
        **common,
    )
    df = graph_result_to_dataframe(r)
    df["ablation"] = "sub_only"
    results.append(df)

    # ── 3. Complements only (zero out substitute adjacency) ──────────────────
    print("3/5 Complements only …")
    r = run_graph_dml(
        panel=train, A_sub=_zero_matrix(A_sub), sub_pid_idx=pid_idx,
        A_comp=A_comp, comp_pid_idx=pid_idx,
        weight_type="edge_weight", embeddings=embeddings, emb_pid_idx=pid_idx,
        **common,
    )
    df = graph_result_to_dataframe(r)
    df["ablation"] = "comp_only"
    results.append(df)

    # ── 4. Binary weights (no PPMI/cosine) ───────────────────────────────────
    print("4/5 Binary weights …")
    r = run_graph_dml(
        panel=train, A_sub=A_sub, sub_pid_idx=pid_idx,
        A_comp=A_comp, comp_pid_idx=pid_idx,
        weight_type="binary", embeddings=embeddings, emb_pid_idx=pid_idx,
        **common,
    )
    df = graph_result_to_dataframe(r)
    df["ablation"] = "binary_weights"
    results.append(df)

    # ── 5. No embeddings ─────────────────────────────────────────────────────
    print("5/5 No embeddings …")
    r = run_graph_dml(
        panel=train, A_sub=A_sub, sub_pid_idx=pid_idx,
        A_comp=A_comp, comp_pid_idx=pid_idx,
        weight_type="edge_weight", embeddings=None, emb_pid_idx=None,
        **common,
    )
    df = graph_result_to_dataframe(r)
    df["ablation"] = "no_embeddings"
    results.append(df)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = ROOT / cfg["output"]["tables_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(out_dir / "ablation_results.csv", index=False)

    print(f"\nAblation results saved to {out_dir / 'ablation_results.csv'}")
    print(all_results.to_string(index=False))


if __name__ == "__main__":
    main()
