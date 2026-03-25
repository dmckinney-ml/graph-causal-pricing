"""
Semi-synthetic data generating process (DGP) for causal evaluation.

Uses the REAL graph structure, REAL covariates, and REAL treatment patterns
from the panel, but replaces the outcome with a simulated Y that has KNOWN
direct and spillover effects. This allows measuring bias, RMSE, and CI
coverage of each estimator against ground truth.

Two modes:

1. Level-space DGP (log_space=False, the legacy default):
    Y_i = β₀ + β₁·T_i + β₂·e_sub_i + β₃·e_comp_i + X_i·γ + ε_i

2. Log-space DGP (log_space=True, for use with log_transform=True):
    log(1+Y_i) = β₀ + β₁·T_i + β₂·e_sub_i + β₃·e_comp_i + X_i·γ + ε_i
    ⇒ Y_i = exp(β₀ + ...) − 1

   In this mode the stated β values are the TRUE effects in the scale
   the estimator targets, so coverage/bias metrics are scale-consistent.

where:
    T_i        = real promo_any (or re-assigned with confounding)
    e_sub_i    = weighted substitute exposure (from real graph)
    e_comp_i   = weighted complement exposure (from real graph)
    X_i        = real covariates (STORE_ID, WEEK_NO, embeddings)
    γ          = covariate effects (estimated from data or set manually)
    β₁, β₂, β₃ = known true effects (configurable)
    ε_i        ~ N(0, σ²)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from src.causal.exposure import add_exposures


@dataclass
class SyntheticDGPConfig:
    """Configuration for the semi-synthetic DGP."""
    true_direct_effect: float = 0.10      # β₁ (log-scale when log_space=True)
    true_sub_spillover: float = -0.05     # β₂ (negative = cannibalization)
    true_comp_spillover: float = 0.03     # β₃ (positive = lift)
    noise_std: float = 0.5                # σ
    confounding_strength: float = 0.5     # how correlated T is with X
    baseline_demand: float = 1.5          # β₀
    log_space: bool = False               # if True, linear model is in log(1+Y) space
    seed: int = 42


def generate_synthetic_outcome(
    panel: pd.DataFrame,
    A_sub: sp.spmatrix,
    sub_pid_idx: np.ndarray,
    A_comp: sp.spmatrix,
    comp_pid_idx: np.ndarray,
    config: SyntheticDGPConfig | None = None,
    reassign_treatment: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Generate a synthetic outcome Y using the real graph and covariates.

    Parameters
    ----------
    panel               : real panel with treatments and covariates
    A_sub, sub_pid_idx  : substitute adjacency and index
    A_comp, comp_pid_idx: complement adjacency and index
    config              : DGP parameters
    reassign_treatment  : if True, re-assign treatment with confounding
                          (correlated with store/week effects)

    Returns
    -------
    panel_synth : panel with 'y_synthetic' replacing units_sold
    true_effects : dict with the true β values used
    """
    if config is None:
        config = SyntheticDGPConfig()

    rng = np.random.default_rng(config.seed)
    n = len(panel)
    panel_synth = panel.copy()

    # Optionally reassign treatment with confounding
    if reassign_treatment:
        # Treatment probability depends on store/week (confounding)
        store_effect = pd.Categorical(panel["STORE_ID"]).codes.astype(float)
        store_effect = (store_effect - store_effect.mean()) / (store_effect.std() + 1e-8)
        week_effect = (panel["WEEK_NO"].values - panel["WEEK_NO"].mean()) / (panel["WEEK_NO"].std() + 1e-8)

        logit = config.confounding_strength * (store_effect + week_effect)
        prob = 1 / (1 + np.exp(-logit))
        panel_synth["promo_any"] = rng.binomial(1, prob).astype(int)

    # Compute weighted exposures from the real graph
    panel_synth = add_exposures(
        panel_synth, A_sub, sub_pid_idx, A_comp, comp_pid_idx,
        treatment_col="promo_any",
        weight_type="edge_weight",
    )

    T = panel_synth["promo_any"].values.astype(float)
    e_sub = panel_synth["sub_exposure_weighted"].values.astype(float)
    e_comp = panel_synth["comp_exposure_weighted"].values.astype(float)

    # Covariate effect: store and week contribute to baseline demand
    store_codes = pd.Categorical(panel_synth["STORE_ID"]).codes.astype(float)
    week_vals = panel_synth["WEEK_NO"].values.astype(float)

    # Normalize to unit scale
    x_store = (store_codes - store_codes.mean()) / (store_codes.std() + 1e-8)
    x_week = (week_vals - week_vals.mean()) / (week_vals.std() + 1e-8)

    # Generate outcome
    noise = rng.normal(0, config.noise_std, size=n)
    linear = (
        config.baseline_demand
        + config.true_direct_effect * T
        + config.true_sub_spillover * e_sub
        + config.true_comp_spillover * e_comp
        + 0.5 * x_store   # covariate effects
        + 0.3 * x_week
        + noise
    )

    if config.log_space:
        # linear is log(1+Y); recover Y in level space
        y = np.expm1(linear)  # exp(linear) - 1
    else:
        y = linear

    # Floor at 0 (demand can't be negative)
    y = np.maximum(y, 0.0)

    panel_synth["y_synthetic"] = y

    true_effects = {
        "direct": config.true_direct_effect,
        "sub_spillover": config.true_sub_spillover,
        "comp_spillover": config.true_comp_spillover,
    }

    return panel_synth, true_effects


def run_simulation_study(
    panel: pd.DataFrame,
    A_sub: sp.spmatrix,
    sub_pid_idx: np.ndarray,
    A_comp: sp.spmatrix,
    comp_pid_idx: np.ndarray,
    estimator_fn,
    config: SyntheticDGPConfig | None = None,
    n_simulations: int = 50,
) -> pd.DataFrame:
    """
    Run multiple DGP draws and evaluate an estimator.

    Parameters
    ----------
    estimator_fn : callable(panel) → dict with keys "direct", "sub_spillover",
                   "comp_spillover", each mapping to (estimate, ci_lower, ci_upper)
    n_simulations : number of independent DGP draws

    Returns
    -------
    DataFrame with columns: sim_id, effect, true_value, estimate, ci_lower, ci_upper
    """
    if config is None:
        config = SyntheticDGPConfig()

    def _run_one(sim_id: int) -> list[dict]:
        sim_config = SyntheticDGPConfig(
            true_direct_effect=config.true_direct_effect,
            true_sub_spillover=config.true_sub_spillover,
            true_comp_spillover=config.true_comp_spillover,
            noise_std=config.noise_std,
            confounding_strength=config.confounding_strength,
            baseline_demand=config.baseline_demand,
            log_space=config.log_space,
            seed=config.seed + sim_id,
        )

        panel_synth, true_effects = generate_synthetic_outcome(
            panel, A_sub, sub_pid_idx, A_comp, comp_pid_idx,
            config=sim_config,
            reassign_treatment=True,
        )

        estimates = estimator_fn(panel_synth)

        return [
            {
                "sim_id": sim_id,
                "effect": effect_name,
                "true_value": true_val,
                "estimate": estimates[effect_name][0],
                "ci_lower": estimates[effect_name][1],
                "ci_upper": estimates[effect_name][2],
            }
            for effect_name, true_val in true_effects.items()
        ]

    # n_jobs=-1 uses all cores; each sim is an independent process.
    # LightGBM also parallelises internally — if contention is observed,
    # lower n_jobs (e.g. n_jobs=4) and let LightGBM use the remaining cores.
    batches = Parallel(n_jobs=2, verbose=5)(
        delayed(_run_one)(sim_id) for sim_id in range(n_simulations)
    )
    return pd.DataFrame([row for batch in batches for row in batch])
