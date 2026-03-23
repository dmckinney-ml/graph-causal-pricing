"""
Interpretability helpers for causal effect decomposition.

After running the graph-aware DML, use these functions to:
  - Decompose CATEs by department / commodity
  - Identify top spillover-affected products
  - Compare substitute vs complement effects
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.graph_dml import GraphCausalResult


def cate_by_group(
    panel: pd.DataFrame,
    result: GraphCausalResult,
    group_col: str = "DEPARTMENT",
) -> pd.DataFrame:
    """
    Decompose conditional average treatment effects by a categorical group.

    Parameters
    ----------
    panel      : the panel used during estimation (same row order)
    result     : GraphCausalResult with .cate of shape (n_obs, 3)
    group_col  : column in panel to group by

    Returns
    -------
    DataFrame with mean CATE and std for each group × effect type.
    """
    if result.cate is None:
        raise ValueError("GraphCausalResult has no CATE (run with return_cate=True)")

    cate = result.cate  # (n_obs, 3)

    # If demand filter was applied, align panel to the filtered rows
    filtered_idx = result.details.get("filtered_index")
    if filtered_idx is not None and len(filtered_idx) != len(panel):
        panel = panel.loc[filtered_idx]

    df = pd.DataFrame({
        group_col: panel[group_col].values,
        "direct_cate": cate[:, 0],
        "sub_spillover_cate": cate[:, 1],
        "comp_spillover_cate": cate[:, 2],
    })

    summary = df.groupby(group_col).agg(
        direct_mean=("direct_cate", "mean"),
        direct_std=("direct_cate", "std"),
        sub_spill_mean=("sub_spillover_cate", "mean"),
        sub_spill_std=("sub_spillover_cate", "std"),
        comp_spill_mean=("comp_spillover_cate", "mean"),
        comp_spill_std=("comp_spillover_cate", "std"),
        n_obs=("direct_cate", "count"),
    ).sort_values("sub_spill_mean", ascending=True)  # most cannibalized first

    return summary


def top_spillover_products(
    panel: pd.DataFrame,
    result: GraphCausalResult,
    effect_type: str = "sub_spillover",
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Identify products with the largest (most extreme) spillover effects.

    Parameters
    ----------
    effect_type : "sub_spillover" or "comp_spillover"
    top_k       : number of products to return

    Returns
    -------
    DataFrame with PRODUCT_ID and mean CATE, sorted by magnitude.
    """
    col_idx = {"direct": 0, "sub_spillover": 1, "comp_spillover": 2}[effect_type]
    cate_vals = result.cate[:, col_idx]

    # If demand filter was applied, align panel to the filtered rows
    filtered_idx = result.details.get("filtered_index")
    if filtered_idx is not None and len(filtered_idx) != len(panel):
        panel = panel.loc[filtered_idx]

    df = pd.DataFrame({
        "PRODUCT_ID": panel["PRODUCT_ID"].values,
        "cate": cate_vals,
    })

    product_cate = df.groupby("PRODUCT_ID")["cate"].agg(["mean", "std", "count"])
    product_cate.columns = ["cate_mean", "cate_std", "n_obs"]
    product_cate = product_cate.sort_values("cate_mean", key=abs, ascending=False)

    return product_cate.head(top_k).reset_index()


def compare_effects(
    result: GraphCausalResult,
) -> pd.DataFrame:
    """
    Side-by-side comparison of direct, substitute, and complement effects.
    """
    return pd.DataFrame([
        {
            "effect": "Direct (own promo)",
            "ATE": result.direct_ate,
            "CI_lower": result.direct_ci[0],
            "CI_upper": result.direct_ci[1],
            "interpretation": "Units gained from own promotion",
        },
        {
            "effect": "Substitute spillover",
            "ATE": result.sub_spillover_ate,
            "CI_lower": result.sub_spillover_ci[0],
            "CI_upper": result.sub_spillover_ci[1],
            "interpretation": "Cannibalization: demand loss when substitutes promoted",
        },
        {
            "effect": "Complement spillover",
            "ATE": result.comp_spillover_ate,
            "CI_lower": result.comp_spillover_ci[0],
            "CI_upper": result.comp_spillover_ci[1],
            "interpretation": "Attachment lift: demand gain when complements promoted",
        },
    ])
