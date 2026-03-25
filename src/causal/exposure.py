"""
Exposure mapping: compute weighted neighborhood treatment exposure.

This is the core graph-aware component. Instead of pre-computing static
spillover features (as in spillover_features.py), exposure is computed
at estimation time using the full edge-weight information from the
adjacency matrices.

Key formula:
    e_i = Σ_{j ∈ N(i)} w_ij · T_j  /  Σ_{j ∈ N(i)} w_ij

Two modes:
    binary      — w_ij = 1 for all edges  (equivalent to frac_treated_*)
    edge_weight — w_ij from PPMI / cosine similarity  (graph-aware)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def compute_weighted_exposure(
    panel: pd.DataFrame,
    A: sp.spmatrix,
    pid_idx: np.ndarray,
    treatment_col: str = "promo_any",
    weight_type: str = "edge_weight",
    prefix: str = "exposure",
) -> pd.DataFrame:
    """
    Compute per-observation weighted neighborhood treatment exposure.

    Parameters
    ----------
    panel         : panel with STORE_ID, PRODUCT_ID, WEEK_NO, and treatment_col
    A             : (N × N) sparse adjacency matrix (weighted)
    pid_idx       : product IDs corresponding to A rows/cols
    treatment_col : column in panel to use as treatment indicator
    weight_type   : "binary" (unweighted) or "edge_weight" (use A weights)
    prefix        : column name prefix for the output exposure column

    Returns
    -------
    panel (copy) with two new columns:
        {prefix}_weighted   — weighted exposure  (Σ w·T / Σ w)
        {prefix}_sum        — sum of neighbor treatments weighted by edge weight
    """
    panel = panel.copy()

    if weight_type == "binary":
        A_use = (A != 0).astype(np.float32)
    elif weight_type == "edge_weight":
        A_use = A.astype(np.float32)
    else:
        raise ValueError(f"weight_type must be 'binary' or 'edge_weight', got {weight_type!r}")

    A_csr = A_use.tocsr()
    n_products = len(pid_idx)

    pid_to_col = {int(pid): i for i, pid in enumerate(pid_idx)}

    # Encode (STORE_ID, WEEK_NO) as unique integer
    sw_key = panel["STORE_ID"].astype(str) + "_" + panel["WEEK_NO"].astype(str)
    sw_codes, _ = pd.factorize(sw_key, sort=True)
    n_sw = int(sw_codes.max()) + 1

    # Map product IDs to adjacency indices
    col_idx = panel["PRODUCT_ID"].map(pid_to_col).fillna(-1).astype(int).values
    valid = col_idx >= 0

    treatment = panel[treatment_col].values.astype(np.float32)
    ones = np.ones(len(panel), dtype=np.float32)

    def _sw_prod_matrix(values, mask, col, n_cols):
        mat = sp.csr_matrix(
            (values[mask], (sw_codes[mask], col[mask])),
            shape=(n_sw, n_cols), dtype=np.float32,
        )
        mat.eliminate_zeros()
        return mat

    # Build store-week × product matrices
    treat_sw = _sw_prod_matrix(treatment, valid, col_idx, n_products)
    present_sw = _sw_prod_matrix(ones, valid, col_idx, n_products)

    # Sparse matmul: numerator[sw, p] = Σ_q A[p,q] · T[sw,q]
    #                denom[sw, p]     = Σ_q A[p,q] · 1[sw,q]  (present neighbors)
    # Keep sparse to avoid materialising a potentially large (n_sw × n_products) dense matrix
    num_sparse = (treat_sw @ A_csr.T).tocsr()
    den_sparse = (present_sw @ A_csr.T).tocsr()

    # Extract per-row values
    exposure_sum = np.zeros(len(panel), dtype=np.float32)
    exposure_weighted = np.zeros(len(panel), dtype=np.float32)
    denom_vals = np.zeros(len(panel), dtype=np.float32)

    rows_v = sw_codes[valid]
    cols_v = col_idx[valid]
    exposure_sum[valid] = np.asarray(num_sparse[rows_v, cols_v]).ravel()
    denom_vals[valid] = np.asarray(den_sparse[rows_v, cols_v]).ravel()

    safe_denom = np.where(denom_vals > 0, denom_vals, 1.0)
    exposure_weighted = exposure_sum / safe_denom
    exposure_weighted[denom_vals == 0] = 0.0

    panel[f"{prefix}_weighted"] = exposure_weighted
    panel[f"{prefix}_sum"] = exposure_sum

    return panel


def add_exposures(
    panel: pd.DataFrame,
    A_sub: sp.spmatrix,
    sub_pid_idx: np.ndarray,
    A_comp: sp.spmatrix,
    comp_pid_idx: np.ndarray,
    treatment_col: str = "promo_any",
    weight_type: str = "edge_weight",
) -> pd.DataFrame:
    """
    Convenience wrapper: add both substitute and complement exposures.

    Returns panel with 4 new columns:
        sub_exposure_weighted, sub_exposure_sum,
        comp_exposure_weighted, comp_exposure_sum
    """
    panel = compute_weighted_exposure(
        panel, A_sub, sub_pid_idx,
        treatment_col=treatment_col,
        weight_type=weight_type,
        prefix="sub_exposure",
    )
    panel = compute_weighted_exposure(
        panel, A_comp, comp_pid_idx,
        treatment_col=treatment_col,
        weight_type=weight_type,
        prefix="comp_exposure",
    )
    return panel
