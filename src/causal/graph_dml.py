"""
Graph-aware causal estimation via exposure-mapping Double ML.

The key idea: instead of treating spillover as confounders (B4), we model
the bivariate treatment (T_i, e_i_sub, e_i_comp) jointly and estimate
three causal effects:

    1. Direct effect  — own promotion → own demand
    2. Sub spillover  — substitute neighbour promotions → own demand (cannibalization)
    3. Comp spillover — complement neighbour promotions → own demand (lift)

The exposure variables e_i use the actual graph edge weights (PPMI for
complements, cosine similarity for substitutes), keeping graph structure
in the estimation loop rather than discarding it after feature engineering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from econml.dml import LinearDML
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from scipy import stats as scipy_stats

from src.causal.exposure import add_exposures


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class GraphCausalResult:
    """Container for graph-aware DML results with direct + spillover effects."""
    model_name: str
    direct_ate: float
    direct_ci: tuple[float, float]
    sub_spillover_ate: float
    sub_spillover_ci: tuple[float, float]
    comp_spillover_ate: float
    comp_spillover_ci: tuple[float, float]
    cate: np.ndarray | None = None
    details: dict = field(default_factory=dict)


# ── Two-way clustered standard errors ────────────────────────────────────────

def _oneway_cluster_vcov(
    X: np.ndarray, residuals: np.ndarray, cluster_ids: np.ndarray,
) -> np.ndarray:
    """
    Compute one-way cluster-robust sandwich variance (Liang-Zeger).

    V = (X'X)^{-1} B (X'X)^{-1}
    B = Σ_g (X_g' e_g)(X_g' e_g)'

    Parameters
    ----------
    X          : (n, k) regressor matrix (residualized treatments)
    residuals  : (n,) OLS residuals from the final-stage regression
    cluster_ids: (n,) integer cluster labels
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)

    B = np.zeros((k, k))
    for g in unique_clusters:
        mask = cluster_ids == g
        Xg_eg = X[mask].T @ residuals[mask]  # (k,)
        B += np.outer(Xg_eg, Xg_eg)

    # Small-sample correction: G/(G-1) * (n-1)/(n-k)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    return correction * XtX_inv @ B @ XtX_inv


def _twoway_cluster_se(
    T_resid: np.ndarray,
    Y_resid: np.ndarray,
    cluster_ids_1: np.ndarray,
    cluster_ids_2: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-way clustered standard errors (Cameron-Gelbach-Miller, 2011).

    V_twoway = V_1 + V_2 − V_intersection

    Parameters
    ----------
    T_resid       : (n, d_t) residualized treatment matrix
    Y_resid       : (n,) residualized outcome
    cluster_ids_1 : (n,) first clustering dimension (e.g. PRODUCT_ID)
    cluster_ids_2 : (n,) second clustering dimension (e.g. STORE_ID)
    confidence_level : for CI construction

    Returns
    -------
    beta    : (d_t,) OLS coefficients from final-stage regression
    se      : (d_t,) two-way clustered standard errors
    ci_lower: (d_t,) lower confidence interval bounds
    ci_upper: (d_t,) upper confidence interval bounds
    """
    # Final-stage OLS: Y_resid = T_resid @ beta + eps
    beta = np.linalg.lstsq(T_resid, Y_resid, rcond=None)[0]
    eps = Y_resid - T_resid @ beta

    # Intersection cluster: unique (c1, c2) pairs
    cluster_inter = cluster_ids_1.astype(np.int64) * (cluster_ids_2.max() + 1) + cluster_ids_2.astype(np.int64)

    V1 = _oneway_cluster_vcov(T_resid, eps, cluster_ids_1)
    V2 = _oneway_cluster_vcov(T_resid, eps, cluster_ids_2)
    V12 = _oneway_cluster_vcov(T_resid, eps, cluster_inter)

    V_twoway = V1 + V2 - V12

    # Ensure positive-definite (eigenvalue floor)
    eigvals, eigvecs = np.linalg.eigh(V_twoway)
    eigvals = np.maximum(eigvals, 0.0)
    V_twoway = eigvecs @ np.diag(eigvals) @ eigvecs.T

    se = np.sqrt(np.diag(V_twoway))

    alpha = 1 - confidence_level
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    ci_lower = beta - z * se
    ci_upper = beta + z * se

    return beta, se, ci_lower, ci_upper


# ── Feature prep (same as baselines, without spillover columns) ──────────────

_CATEGORICAL_COLS = ["DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND"]


def _prepare_covariates(
    panel: pd.DataFrame,
    embeddings: np.ndarray | None = None,
    emb_pid_idx: np.ndarray | None = None,
    include_discount_rate: bool = False,
) -> pd.DataFrame:
    """Build covariate matrix W (confounders, not treatments)."""
    W = pd.DataFrame(index=panel.index)
    W["STORE_ID"] = panel["STORE_ID"].values
    W["WEEK_NO"] = panel["WEEK_NO"].values

    for col in _CATEGORICAL_COLS:
        if col in panel.columns:
            W[col] = panel[col].astype("category").cat.codes

    if include_discount_rate and "discount_rate" in panel.columns:
        W["discount_rate"] = panel["discount_rate"].values

    if embeddings is not None and emb_pid_idx is not None:
        emb_dim = embeddings.shape[1]
        pid_vals = panel["PRODUCT_ID"].values.astype(int)
        emb_pids = emb_pid_idx.astype(int)
        max_pid = max(pid_vals.max(), emb_pids.max()) + 1
        pid_to_row = np.full(max_pid, -1, dtype=np.intp)
        pid_to_row[emb_pids] = np.arange(len(emb_pids), dtype=np.intp)
        row_indices = pid_to_row[pid_vals]
        emb_matrix = np.zeros((len(panel), emb_dim), dtype=np.float32)
        mask = row_indices >= 0
        emb_matrix[mask] = embeddings[row_indices[mask]]
        emb_cols = [f"emb_{i}" for i in range(emb_dim)]
        W[emb_cols] = emb_matrix

    return W


def _get_nuisance_models():
    model_y = LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        num_leaves=31, min_child_samples=20, verbose=-1,
    )
    model_t = LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        num_leaves=31, min_child_samples=20, verbose=-1,
    )
    return model_y, model_t


# ── Main estimator ───────────────────────────────────────────────────────────

def run_graph_dml(
    panel: pd.DataFrame,
    A_sub: sp.spmatrix,
    sub_pid_idx: np.ndarray,
    A_comp: sp.spmatrix,
    comp_pid_idx: np.ndarray,
    outcome_col: str = "units_sold",
    treatment_col: str = "promo_any",
    weight_type: str = "edge_weight",
    n_folds: int = 5,
    embeddings: np.ndarray | None = None,
    emb_pid_idx: np.ndarray | None = None,
    confidence_level: float = 0.95,
    seed: int = 42,
    log_transform: bool = True,
    min_demand_filter: float | None = 1.0,
    include_discount_rate: bool = False,
) -> GraphCausalResult:
    """
    Run the exposure-mapping DML with bivariate treatment.

    Treatment vector: [T_i, e_sub_weighted, e_comp_weighted]
    This estimates the partial effect of each component controlling for the others.

    Parameters
    ----------
    log_transform     : if True, transform Y via log(1+Y) before estimation.
                        Addresses zero-inflated count outcomes.
    min_demand_filter : if not None, drop products whose mean outcome across
                        all store-weeks is below this threshold (removes
                        structural zeros). Set to None to disable.
    include_discount_rate : if True, include discount_rate in W. Default is
                        False for promo_any so the direct effect reflects the
                        total promotion package rather than a discount-held-fixed
                        incremental effect.
    """
    panel_work = panel.copy()

    # 0a. Filter structural zeros — products that almost never sell
    if min_demand_filter is not None:
        product_means = panel_work.groupby("PRODUCT_ID")[outcome_col].mean()
        keep_pids = product_means[product_means >= min_demand_filter].index
        n_before = len(panel_work)
        panel_work = panel_work[panel_work["PRODUCT_ID"].isin(keep_pids)].copy()
        n_after = len(panel_work)
        if n_before > n_after:
            print(f"  demand filter: kept {n_after:,}/{n_before:,} rows "
                  f"({len(keep_pids):,} products with mean {outcome_col} ≥ {min_demand_filter})")

    # 1. Compute weighted neighbourhood exposures
    panel_exp = add_exposures(
        panel_work, A_sub, sub_pid_idx, A_comp, comp_pid_idx,
        treatment_col=treatment_col,
        weight_type=weight_type,
    )

    # 2. Build treatment matrix (3 columns)
    T = np.column_stack([
        panel_exp[treatment_col].values,
        panel_exp["sub_exposure_weighted"].values,
        panel_exp["comp_exposure_weighted"].values,
    ]).astype(np.float64)

    # 3. Build covariate matrix and outcome
    W = _prepare_covariates(
        panel_exp,
        embeddings=embeddings,
        emb_pid_idx=emb_pid_idx,
        include_discount_rate=include_discount_rate,
    )
    Y = panel_exp[outcome_col].values.astype(np.float64)

    if log_transform:
        Y = np.log1p(Y)

    # 4. Cluster-aware CV: keep all obs for a product in the same fold
    product_ids = panel_exp["PRODUCT_ID"].values
    store_ids = panel_exp["STORE_ID"].values
    groups = product_ids  # GroupKFold grouping variable
    cv = GroupKFold(n_splits=n_folds)

    # 5. Fit LinearDML with multi-dimensional treatment
    model_y, model_t_base = _get_nuisance_models()
    model_t = MultiOutputRegressor(model_t_base)
    est = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=cv,
        random_state=seed,
    )
    est.fit(Y, T, X=W, groups=groups, cache_values=True)

    # 6. CATEs from EconML (used for group decomposition)
    cate = est.const_marginal_effect(W)  # shape: (n_obs, 3)

    # 7. Two-way clustered standard errors (Cameron-Gelbach-Miller)
    #    Extract cross-fitted residuals from the fitted DML estimator.
    #    residuals_ returns (Y_res, T_res, X_out, W_out) — note that row
    #    order may differ from input, but the residuals are aligned with
    #    each other, and we need the cluster IDs in the same order.
    Y_res, T_res, _, _ = est.residuals_

    # residuals_ may reorder rows; we need matching cluster IDs.
    # EconML's _OrthoLearner stores the reordered indices when groups
    # are used. Fall back to original order (safe when no subsampling).
    n_res = len(Y_res)
    if n_res == len(product_ids):
        pid_for_se = product_ids
        sid_for_se = store_ids
    else:
        # Subsample case — use original IDs (coverage is approximate)
        pid_for_se = product_ids[:n_res]
        sid_for_se = store_ids[:n_res]

    beta, se, ci_lower, ci_upper = _twoway_cluster_se(
        T_res, Y_res,
        cluster_ids_1=pid_for_se,
        cluster_ids_2=sid_for_se,
        confidence_level=confidence_level,
    )

    return GraphCausalResult(
        model_name=f"graph_dml_{weight_type}",
        direct_ate=float(beta[0]),
        direct_ci=(float(ci_lower[0]), float(ci_upper[0])),
        sub_spillover_ate=float(beta[1]),
        sub_spillover_ci=(float(ci_lower[1]), float(ci_upper[1])),
        comp_spillover_ate=float(beta[2]),
        comp_spillover_ci=(float(ci_lower[2]), float(ci_upper[2])),
        cate=cate,
        details={
            "n_obs": len(Y),
            "n_obs_pre_filter": len(panel),
            "filtered_index": panel_exp.index,
            "weight_type": weight_type,
            "log_transform": log_transform,
            "min_demand_filter": min_demand_filter,
            "include_discount_rate": include_discount_rate,
            "clustered_se": True,
            "cluster_dims": ["PRODUCT_ID", "STORE_ID"],
            "treatment_cols": [treatment_col, "sub_exposure_weighted", "comp_exposure_weighted"],
        },
    )


def graph_result_to_dataframe(result: GraphCausalResult) -> pd.DataFrame:
    """Convert a GraphCausalResult to a tidy DataFrame."""
    return pd.DataFrame([
        {
            "model": result.model_name,
            "effect": "direct",
            "ATE": result.direct_ate,
            "CI_lower": result.direct_ci[0],
            "CI_upper": result.direct_ci[1],
        },
        {
            "model": result.model_name,
            "effect": "sub_spillover",
            "ATE": result.sub_spillover_ate,
            "CI_lower": result.sub_spillover_ci[0],
            "CI_upper": result.sub_spillover_ci[1],
        },
        {
            "model": result.model_name,
            "effect": "comp_spillover",
            "ATE": result.comp_spillover_ate,
            "CI_lower": result.comp_spillover_ci[0],
            "CI_upper": result.comp_spillover_ci[1],
        },
    ])
