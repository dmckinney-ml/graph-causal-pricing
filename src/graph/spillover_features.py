"""
Compute graph-based spillover features for the causal panel.

For each observation (STORE_ID, PRODUCT_ID, WEEK_NO), adds:

  Substitute neighborhood (A_sub):
    n_treated_substitutes      – count of substitute neighbors with promo_any=1
    frac_treated_substitutes   – fraction of substitute neighbors treated
    avg_units_substitutes      – mean units_sold of substitutes in same store-week

  Complement neighborhood (A_comp):
    n_treated_complements      – count of complement neighbors with promo_any=1
    frac_treated_complements   – fraction of complement neighbors treated
    avg_units_complements      – mean units_sold of complements in same store-week
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def add_spillover_features(
    panel: pd.DataFrame,
    A_sub: sp.spmatrix,
    sub_pid_idx: np.ndarray,
    A_comp: sp.spmatrix,
    comp_pid_idx: np.ndarray,
) -> pd.DataFrame:
    """
    Attach spillover features to the panel (returns modified copy).

    Uses sparse matrix multiplication to avoid per-row Python loops.
    Complexity: O(n_store_weeks × nnz(A)) instead of O(n_rows × mean_degree).

    Parameters
    ----------
    panel        : store × product × week panel (from panel_builder.build_panel)
    A_sub        : (N × N) substitute adjacency (sparse)
    sub_pid_idx  : product IDs for A_sub rows/cols
    A_comp       : (M × M) complement adjacency (sparse)
    comp_pid_idx : product IDs for A_comp rows/cols

    Returns
    -------
    panel with six new columns added
    """
    panel = panel.copy()

    # Binary adjacency — weights are used for substitute reweighting elsewhere;
    # here we need counts, so treat any non-zero edge as a single neighbor.
    A_sub_bin = (A_sub != 0).astype(np.float32)
    A_comp_bin = (A_comp != 0).astype(np.float32)

    # Product ID → column index in each adjacency matrix
    sub_pid_to_col = {int(pid): i for i, pid in enumerate(sub_pid_idx)}
    comp_pid_to_col = {int(pid): i for i, pid in enumerate(comp_pid_idx)}

    # Encode each (STORE_ID, WEEK_NO) pair as a unique integer row index
    sw_key = panel["STORE_ID"].astype(str) + "_" + panel["WEEK_NO"].astype(str)
    sw_codes, _ = pd.factorize(sw_key, sort=True)
    n_sw = int(sw_codes.max()) + 1

    # Map panel product IDs to adjacency column indices; -1 = not in graph
    sub_col = panel["PRODUCT_ID"].map(sub_pid_to_col).fillna(-1).astype(int).values
    comp_col = panel["PRODUCT_ID"].map(comp_pid_to_col).fillna(-1).astype(int).values
    sub_mask = sub_col >= 0
    comp_mask = comp_col >= 0

    promo = panel["promo_any"].values.astype(np.float32)
    units = panel["units_sold"].values.astype(np.float32)
    ones = np.ones(len(panel), dtype=np.float32)

    def _sw_prod_matrix(values: np.ndarray, mask: np.ndarray,
                        col: np.ndarray, n_cols: int) -> sp.csr_matrix:
        """Build a sparse (n_sw × n_cols) store-week × product matrix."""
        mat = sp.csr_matrix(
            (values[mask], (sw_codes[mask], col[mask])),
            shape=(n_sw, n_cols), dtype=np.float32,
        )
        mat.eliminate_zeros()
        return mat

    n_sub = len(sub_pid_idx)
    n_comp = len(comp_pid_idx)

    promo_sw_sub    = _sw_prod_matrix(promo, sub_mask,  sub_col,  n_sub)
    units_sw_sub    = _sw_prod_matrix(units, sub_mask,  sub_col,  n_sub)
    present_sw_sub  = _sw_prod_matrix(ones,  sub_mask,  sub_col,  n_sub)

    promo_sw_comp   = _sw_prod_matrix(promo, comp_mask, comp_col, n_comp)
    units_sw_comp   = _sw_prod_matrix(units, comp_mask, comp_col, n_comp)
    present_sw_comp = _sw_prod_matrix(ones,  comp_mask, comp_col, n_comp)

    # Sparse matmul: result[sw, p] = Σ_q A[p,q] · feature[sw, q]
    # Shape: (n_sw × n_prod) @ (n_prod × n_prod) → (n_sw × n_prod)
    n_treated_sub_mat  = (promo_sw_sub    @ A_sub_bin.T).toarray()
    sum_units_sub_mat  = (units_sw_sub    @ A_sub_bin.T).toarray()
    n_present_sub_mat  = (present_sw_sub  @ A_sub_bin.T).toarray()

    n_treated_comp_mat = (promo_sw_comp   @ A_comp_bin.T).toarray()
    sum_units_comp_mat = (units_sw_comp   @ A_comp_bin.T).toarray()
    n_present_comp_mat = (present_sw_comp @ A_comp_bin.T).toarray()

    # Extract the value for each panel row via integer indexing
    def _extract(mat: np.ndarray, mask: np.ndarray, col: np.ndarray) -> np.ndarray:
        out = np.zeros(len(panel), dtype=np.float32)
        out[mask] = mat[sw_codes[mask], col[mask]]
        return out

    n_treated_sub  = _extract(n_treated_sub_mat,  sub_mask,  sub_col)
    sum_units_sub  = _extract(sum_units_sub_mat,   sub_mask,  sub_col)
    n_present_sub  = _extract(n_present_sub_mat,   sub_mask,  sub_col)

    n_treated_comp = _extract(n_treated_comp_mat, comp_mask, comp_col)
    sum_units_comp = _extract(sum_units_comp_mat,  comp_mask, comp_col)
    n_present_comp = _extract(n_present_comp_mat,  comp_mask, comp_col)

    panel["n_treated_substitutes"]    = n_treated_sub.astype(int)
    panel["frac_treated_substitutes"] = np.where(
        n_present_sub > 0, n_treated_sub / n_present_sub, 0.0
    ).astype(np.float32)
    panel["avg_units_substitutes"]    = np.where(
        n_present_sub > 0, sum_units_sub / n_present_sub, 0.0
    ).astype(np.float32)

    panel["n_treated_complements"]    = n_treated_comp.astype(int)
    panel["frac_treated_complements"] = np.where(
        n_present_comp > 0, n_treated_comp / n_present_comp, 0.0
    ).astype(np.float32)
    panel["avg_units_complements"]    = np.where(
        n_present_comp > 0, sum_units_comp / n_present_comp, 0.0
    ).astype(np.float32)

    return panel
