"""
Baseline causal estimators (no graph-aware exposure mapping).

B1: Naive OLS — units_sold ~ promo_any + confounders
B2: CausalForestDML — heterogeneous TE, no graph
B3: LinearDML — partially linear, no graph
B4: LinearDML + unweighted spillover features as confounders
"""

from __future__ import annotations

import gc
import resource
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import LinearDML, CausalForestDML
from lightgbm import LGBMRegressor


def _log_memory(label: str = "") -> None:
    """Print current RSS in GB."""
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns KB
    import sys
    if sys.platform == "darwin":
        rss_gb = rss_bytes / 1e9
    else:
        rss_gb = rss_bytes / 1e6
    print(f"  [mem] {label} RSS={rss_gb:.1f} GB")


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class CausalResult:
    """Standard container for causal estimation results."""
    model_name: str
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    cate: np.ndarray | None = None      # per-observation, if available
    details: dict | None = None


# ── Feature preparation ──────────────────────────────────────────────────────

_CATEGORICAL_COLS = ["DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND"]
_SPILLOVER_COLS = [
    "n_treated_substitutes", "frac_treated_substitutes", "avg_units_substitutes",
    "n_treated_complements", "frac_treated_complements", "avg_units_complements",
]


def _prepare_features(
    panel: pd.DataFrame,
    include_spillover: bool = False,
    embeddings: np.ndarray | None = None,
    emb_pid_idx: np.ndarray | None = None,
    include_discount_rate: bool = False,
) -> pd.DataFrame:
    """
    Build the covariate matrix X from the panel.

    Encodes categoricals, adds store/week dummies, optionally appends
    spillover features and Node2Vec embeddings.
    """
    X = pd.DataFrame(index=panel.index)

    # Store and week fixed effects (as integers for tree models)
    X["STORE_ID"] = panel["STORE_ID"].values
    X["WEEK_NO"] = panel["WEEK_NO"].values

    # Categorical features — label-encode for LightGBM
    for col in _CATEGORICAL_COLS:
        if col in panel.columns:
            X[col] = panel[col].astype("category").cat.codes

    # Discount rate as a covariate (pre-treatment price info)
    if include_discount_rate and "discount_rate" in panel.columns:
        X["discount_rate"] = panel["discount_rate"].values

    # Unweighted spillover features (only for B4)
    if include_spillover:
        for col in _SPILLOVER_COLS:
            if col in panel.columns:
                X[col] = panel[col].values

    # Node2Vec embeddings as product-level features (vectorised lookup)
    if embeddings is not None and emb_pid_idx is not None:
        emb_dim = embeddings.shape[1]
        # Build a dense int→row-index array for O(1) vectorised lookup
        pid_vals = panel["PRODUCT_ID"].values.astype(int)
        emb_pids = emb_pid_idx.astype(int)
        max_pid = max(pid_vals.max(), emb_pids.max()) + 1
        pid_to_row = np.full(max_pid, -1, dtype=np.intp)
        pid_to_row[emb_pids] = np.arange(len(emb_pids), dtype=np.intp)
        row_indices = pid_to_row[pid_vals]           # -1 where pid not in index
        emb_matrix = np.zeros((len(panel), emb_dim), dtype=np.float32)
        mask = row_indices >= 0
        emb_matrix[mask] = embeddings[row_indices[mask]]
        emb_cols = [f"emb_{i}" for i in range(emb_dim)]
        X[emb_cols] = emb_matrix

    return X


def _get_nuisance_models():
    """Default LightGBM nuisance models for DML.

    Both Y and T models are regressors: EconML's DML estimators (LinearDML,
    CausalForestDML) residualise T internally and require a regressor regardless
    of whether the treatment is binary in the original data.
    """
    _params = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        num_leaves=31, min_child_samples=20, verbose=-1,
    )
    return LGBMRegressor(**_params), LGBMRegressor(**_params)


# ── B1: Naive OLS ────────────────────────────────────────────────────────────

def run_ols_baseline(
    panel: pd.DataFrame,
    outcome_col: str = "units_sold",
    treatment_col: str = "promo_any",
    embeddings: np.ndarray | None = None,
    emb_pid_idx: np.ndarray | None = None,
    confidence_level: float = 0.95,
    include_discount_rate: bool = False,
) -> CausalResult:
    """OLS regression: Y ~ T + X. Returns ATE = coefficient on T."""
    # Naive baseline: no embeddings — including 128-dim embeddings makes the
    # HC1 sandwich estimator O(n·k²) and extremely slow for no conceptual gain.
    X = _prepare_features(panel, include_spillover=False,
                          embeddings=None, emb_pid_idx=None,
                          include_discount_rate=include_discount_rate)
    T = panel[treatment_col].values
    Y = panel[outcome_col].values

    design = sm.add_constant(np.column_stack([T, X.values]))
    model = sm.OLS(Y, design).fit(cov_type="HC1")

    ate = model.params[1]  # coefficient on treatment (first variable after constant)
    ci = model.conf_int(alpha=1 - confidence_level)[1]

    return CausalResult(
        model_name="ols_no_graph",
        ate=float(ate),
        ate_ci_lower=float(ci[0]),
        ate_ci_upper=float(ci[1]),
        details={
            "r_squared": model.rsquared,
            "n_obs": model.nobs,
            "include_discount_rate": include_discount_rate,
        },
    )


# ── B2: CausalForestDML (no graph) ──────────────────────────────────────────

def run_causal_forest_baseline(
    panel: pd.DataFrame,
    outcome_col: str = "units_sold",
    treatment_col: str = "promo_any",
    n_folds: int = 5,
    n_estimators: int = 200,
    max_depth: int = 8,
    embeddings: np.ndarray | None = None,
    emb_pid_idx: np.ndarray | None = None,
    confidence_level: float = 0.95,
    seed: int = 42,
    include_discount_rate: bool = False,
) -> CausalResult:
    """CausalForestDML: heterogeneous treatment effects, no graph features."""
    X = _prepare_features(panel, include_spillover=False,
                          embeddings=embeddings, emb_pid_idx=emb_pid_idx,
                          include_discount_rate=include_discount_rate)
    T = panel[treatment_col].values.reshape(-1, 1)
    Y = panel[outcome_col].values

    model_y, model_t = _get_nuisance_models()
    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        cv=n_folds,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_samples=0.1,
        random_state=seed,
    )
    est.fit(Y, T, X=X)

    ate_inf = est.ate_inference(X)
    cate = est.effect(X)

    return CausalResult(
        model_name="causal_forest_no_graph",
        ate=float(ate_inf.mean_point),
        ate_ci_lower=float(ate_inf.conf_int_mean(alpha=1 - confidence_level)[0]),
        ate_ci_upper=float(ate_inf.conf_int_mean(alpha=1 - confidence_level)[1]),
        cate=cate.flatten(),
        details={"n_obs": len(Y), "include_discount_rate": include_discount_rate},
    )


# ── B3: LinearDML (no graph) ────────────────────────────────────────────────

def run_dml_baseline(
    panel: pd.DataFrame,
    outcome_col: str = "units_sold",
    treatment_col: str = "promo_any",
    n_folds: int = 5,
    embeddings: np.ndarray | None = None,
    emb_pid_idx: np.ndarray | None = None,
    confidence_level: float = 0.95,
    seed: int = 42,
    include_discount_rate: bool = False,
) -> CausalResult:
    """LinearDML: partially linear model, no graph features."""
    X = _prepare_features(panel, include_spillover=False,
                          embeddings=embeddings, emb_pid_idx=emb_pid_idx,
                          include_discount_rate=include_discount_rate)
    T = panel[treatment_col].values.reshape(-1, 1)
    Y = panel[outcome_col].values

    model_y, model_t = _get_nuisance_models()
    est = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=n_folds,
        random_state=seed,
    )
    est.fit(Y, T, X=X)

    ate_inf = est.ate_inference(X)
    cate = est.effect(X)

    return CausalResult(
        model_name="dml_no_graph",
        ate=float(ate_inf.mean_point),
        ate_ci_lower=float(ate_inf.conf_int_mean(alpha=1 - confidence_level)[0]),
        ate_ci_upper=float(ate_inf.conf_int_mean(alpha=1 - confidence_level)[1]),
        cate=cate.flatten(),
        details={"n_obs": len(Y), "include_discount_rate": include_discount_rate},
    )


# ── B4: LinearDML + unweighted spillover features ───────────────────────────

def run_dml_unweighted_spillover(
    panel: pd.DataFrame,
    outcome_col: str = "units_sold",
    treatment_col: str = "promo_any",
    n_folds: int = 5,
    embeddings: np.ndarray | None = None,
    emb_pid_idx: np.ndarray | None = None,
    confidence_level: float = 0.95,
    seed: int = 42,
    include_discount_rate: bool = False,
) -> CausalResult:
    """LinearDML with pre-computed unweighted spillover features as confounders."""
    X = _prepare_features(panel, include_spillover=True,
                          embeddings=embeddings, emb_pid_idx=emb_pid_idx,
                          include_discount_rate=include_discount_rate)
    T = panel[treatment_col].values.reshape(-1, 1)
    Y = panel[outcome_col].values

    model_y, model_t = _get_nuisance_models()
    est = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=n_folds,
        random_state=seed,
    )
    est.fit(Y, T, X=X)

    ate_inf = est.ate_inference(X)
    cate = est.effect(X)

    return CausalResult(
        model_name="dml_unweighted_spillover",
        ate=float(ate_inf.mean_point),
        ate_ci_lower=float(ate_inf.conf_int_mean(alpha=1 - confidence_level)[0]),
        ate_ci_upper=float(ate_inf.conf_int_mean(alpha=1 - confidence_level)[1]),
        cate=cate.flatten(),
        details={"n_obs": len(Y), "include_discount_rate": include_discount_rate},
    )


# ── Run all baselines ────────────────────────────────────────────────────────

def run_all_baselines(
    panel: pd.DataFrame,
    outcome_col: str = "units_sold",
    treatment_col: str = "promo_any",
    n_folds: int = 5,
    embeddings: np.ndarray | None = None,
    emb_pid_idx: np.ndarray | None = None,
    confidence_level: float = 0.95,
    seed: int = 42,
    include_discount_rate: bool = False,
    dml_max_rows: int = 10_000_000,
) -> list[CausalResult]:
    """Run all 4 baselines and return results.

    B1 (OLS) runs on the full panel (low memory — 6 features, no embeddings).
    B2, B3, B4 (DML variants) subsample to dml_max_rows to avoid OOM on large
    panels — the ATE estimates are statistically robust at 10M rows and these
    are comparison baselines, not the primary estimator.
    """
    common = dict(
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        embeddings=embeddings,
        emb_pid_idx=emb_pid_idx,
        confidence_level=confidence_level,
        include_discount_rate=include_discount_rate,
    )

    # DML subsample for B2/B3/B4
    if len(panel) > dml_max_rows:
        dml_panel = panel.sample(n=dml_max_rows, random_state=seed)
        print(f"DML baselines (B2-B4): subsampled to {dml_max_rows:,} rows "
              f"(from {len(panel):,}) for memory efficiency")
    else:
        dml_panel = panel

    results: list[CausalResult] = []

    # B1: OLS on full panel (6 features, ~4 GB peak)
    print(f"[B1] Starting OLS on {len(panel):,} rows...")
    _log_memory("B1 start")
    results.append(run_ols_baseline(panel=panel, **common))
    print(f"[B1] OLS complete. ATE={results[-1].ate:.6f}")
    _log_memory("B1 end")
    gc.collect()

    # B2: CausalForestDML on subsampled panel
    print(f"[B2] Starting CausalForestDML on {len(dml_panel):,} rows...")
    _log_memory("B2 start")
    results.append(run_causal_forest_baseline(
        panel=dml_panel, **common, n_folds=n_folds, seed=seed))
    print(f"[B2] CausalForestDML complete. ATE={results[-1].ate:.6f}")
    _log_memory("B2 end")
    gc.collect()

    # B3: LinearDML on subsampled panel
    print(f"[B3] Starting LinearDML on {len(dml_panel):,} rows...")
    _log_memory("B3 start")
    results.append(run_dml_baseline(
        panel=dml_panel, **common, n_folds=n_folds, seed=seed))
    print(f"[B3] LinearDML complete. ATE={results[-1].ate:.6f}")
    _log_memory("B3 end")
    gc.collect()

    # B4: LinearDML + spillover on subsampled panel
    print(f"[B4] Starting LinearDML+spillover on {len(dml_panel):,} rows...")
    _log_memory("B4 start")
    results.append(run_dml_unweighted_spillover(
        panel=dml_panel, **common, n_folds=n_folds, seed=seed))
    print(f"[B4] LinearDML+spillover complete. ATE={results[-1].ate:.6f}")
    _log_memory("B4 end")
    gc.collect()

    return results


def results_to_dataframe(results: list[CausalResult]) -> pd.DataFrame:
    """Convert list of CausalResult to a summary DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "model": r.model_name,
            "ATE": r.ate,
            "CI_lower": r.ate_ci_lower,
            "CI_upper": r.ate_ci_upper,
            **(r.details or {}),
        })
    return pd.DataFrame(rows)
