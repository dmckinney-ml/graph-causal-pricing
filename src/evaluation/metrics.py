"""
Evaluation metrics for causal estimation.

Supports two evaluation modes:
  1. Semi-synthetic — known ground truth → bias, RMSE, CI coverage
  2. Real-data refutation — placebo tests, covariate balance diagnostics
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Semi-synthetic metrics (require known true effects)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bias(sim_results: pd.DataFrame) -> pd.DataFrame:
    """
    Mean bias: E[estimate − true_value] for each effect type.

    Parameters
    ----------
    sim_results : DataFrame from run_simulation_study with columns
                  [sim_id, effect, true_value, estimate, ci_lower, ci_upper]
    """
    sim_results = sim_results.copy()
    sim_results["bias"] = sim_results["estimate"] - sim_results["true_value"]

    return (
        sim_results.groupby("effect")
        .agg(
            mean_bias=("bias", "mean"),
            std_bias=("bias", "std"),
            n_sims=("bias", "count"),
        )
        .reset_index()
    )


def compute_rmse(sim_results: pd.DataFrame) -> pd.DataFrame:
    """Root mean squared error for each effect type."""
    sim_results = sim_results.copy()
    sim_results["sq_error"] = (sim_results["estimate"] - sim_results["true_value"]) ** 2

    agg = sim_results.groupby("effect")["sq_error"].mean().reset_index()
    agg.columns = ["effect", "mse"]
    agg["rmse"] = np.sqrt(agg["mse"])
    return agg[["effect", "rmse"]]


def compute_ci_coverage(sim_results: pd.DataFrame) -> pd.DataFrame:
    """
    Fraction of simulations where the true value falls within the CI.
    Target: 95% for a 95% CI.
    """
    sim_results = sim_results.copy()
    sim_results["covers"] = (
        (sim_results["ci_lower"] <= sim_results["true_value"])
        & (sim_results["true_value"] <= sim_results["ci_upper"])
    ).astype(int)

    return (
        sim_results.groupby("effect")
        .agg(
            coverage=("covers", "mean"),
            n_sims=("covers", "count"),
        )
        .reset_index()
    )


def full_synthetic_evaluation(sim_results: pd.DataFrame) -> pd.DataFrame:
    """Combine bias, RMSE, and coverage into one summary table."""
    bias = compute_bias(sim_results)
    rmse = compute_rmse(sim_results)
    coverage = compute_ci_coverage(sim_results)

    summary = bias.merge(rmse, on="effect").merge(coverage, on="effect")
    # After merging, n_sims from bias becomes n_sims_x and from coverage n_sims_y
    if "n_sims_x" in summary.columns:
        summary = summary.rename(columns={"n_sims_x": "n_sims"}).drop(columns=["n_sims_y"])
    summary = summary[["effect", "mean_bias", "std_bias", "rmse", "coverage", "n_sims"]]
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Real-data refutation tests
# ═══════════════════════════════════════════════════════════════════════════════

def placebo_test(
    panel: pd.DataFrame,
    estimator_fn,
    n_permutations: int = 20,
    treatment_col: str = "promo_any",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Placebo test: shuffle treatment labels → all effects should be ≈ 0.

    Parameters
    ----------
    estimator_fn : callable(panel) → dict mapping effect names to
                   (estimate, ci_lower, ci_upper)
    n_permutations : number of random shuffles

    Returns
    -------
    DataFrame with placebo estimates for each permutation × effect.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for perm_id in range(n_permutations):
        panel_perm = panel.copy()
        panel_perm[treatment_col] = rng.permutation(panel_perm[treatment_col].values)

        estimates = estimator_fn(panel_perm)
        for effect_name, (est, ci_lo, ci_hi) in estimates.items():
            rows.append({
                "perm_id": perm_id,
                "effect": effect_name,
                "estimate": est,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            })

    return pd.DataFrame(rows)


def covariate_balance(
    panel: pd.DataFrame,
    treatment_col: str = "promo_any",
    covariate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute standardized mean differences (SMD) for treatment/control balance.

    SMD = (mean_treated − mean_control) / pooled_std
    Rule of thumb: |SMD| < 0.1 is acceptable.
    """
    if covariate_cols is None:
        covariate_cols = ["discount_rate", "WEEK_NO"]

    treated = panel[panel[treatment_col] == 1]
    control = panel[panel[treatment_col] == 0]

    rows = []
    for col in covariate_cols:
        if col not in panel.columns:
            continue
        vals = panel[col]
        if vals.dtype == "object" or vals.dtype.name == "category":
            continue

        t_mean = treated[col].mean()
        c_mean = control[col].mean()
        t_std = treated[col].std()
        c_std = control[col].std()
        pooled_std = np.sqrt((t_std ** 2 + c_std ** 2) / 2)
        smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0.0

        rows.append({
            "covariate": col,
            "treated_mean": t_mean,
            "control_mean": c_mean,
            "smd": smd,
            "acceptable": abs(smd) < 0.1,
        })

    return pd.DataFrame(rows)


def directional_check(
    result,
) -> pd.DataFrame:
    """
    Verify effects have expected signs:
        direct > 0, sub_spillover < 0, comp_spillover > 0

    Parameters
    ----------
    result : GraphCausalResult or dict-like with direct_ate, sub_spillover_ate,
             comp_spillover_ate
    """
    checks = [
        {
            "effect": "direct",
            "estimate": result.direct_ate,
            "expected_sign": "positive",
            "passes": result.direct_ate > 0,
        },
        {
            "effect": "sub_spillover",
            "estimate": result.sub_spillover_ate,
            "expected_sign": "negative",
            "passes": result.sub_spillover_ate < 0,
        },
        {
            "effect": "comp_spillover",
            "estimate": result.comp_spillover_ate,
            "expected_sign": "positive",
            "passes": result.comp_spillover_ate > 0,
        },
    ]
    return pd.DataFrame(checks)
