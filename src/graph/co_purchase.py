"""
Complement graph via basket co-occurrence + PPMI.

Products that are frequently bought together in the same basket are connected
with edges weighted by Positive Pointwise Mutual Information (PPMI).

Edge weight: PPMI(A,B) = max(log[P(A,B)/(P(A)*P(B))], 0)
             normalised to [0, 1] across all retained pairs.

Minimum co-occurrence threshold filters noise (default: 10 baskets).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_complement_adjacency(
    transaction_path: str,
    min_cooccurrence: int = 10,
    product_ids: list[int] | None = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Compute the complement (co-purchase) adjacency matrix.

    Parameters
    ----------
    transaction_path  : path to transaction_data.csv
    min_cooccurrence  : minimum basket co-occurrence to retain an edge
    product_ids       : ordered list of product IDs that defines the matrix
                        row/column index.  If None, use all products present
                        in the transaction data.

    Returns
    -------
    A_comp  : (N × N) sparse CSR matrix of normalised PPMI weights in [0, 1]
    pid_idx : 1-D array of product IDs corresponding to rows/columns
    """
    tx = pd.read_csv(transaction_path, usecols=["BASKET_ID", "PRODUCT_ID"])
    tx.columns = tx.columns.str.strip()

    if product_ids is not None:
        tx = tx[tx["PRODUCT_ID"].isin(product_ids)]

    all_pids = sorted(tx["PRODUCT_ID"].unique()) if product_ids is None else list(product_ids)
    n = len(all_pids)
    pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}
    pid_idx = np.array(all_pids, dtype=np.int64)

    # Product marginal counts (number of baskets containing each product)
    product_basket_counts = tx.groupby("PRODUCT_ID")["BASKET_ID"].nunique()

    # Basket-level product lists for co-occurrence
    basket_products = tx.groupby("BASKET_ID")["PRODUCT_ID"].apply(list)
    n_baskets = basket_products.shape[0]

    # Accumulate co-occurrence counts in a dict to keep memory manageable
    cooc: dict[tuple[int, int], int] = {}
    for products in basket_products:
        # De-duplicate within basket
        unique_prods = list(set(products))
        for i_p in range(len(unique_prods)):
            for j_p in range(i_p + 1, len(unique_prods)):
                a, b = unique_prods[i_p], unique_prods[j_p]
                if a not in pid_to_idx or b not in pid_to_idx:
                    continue
                key = (min(a, b), max(a, b))
                cooc[key] = cooc.get(key, 0) + 1

    # Filter by minimum co-occurrence
    cooc = {k: v for k, v in cooc.items() if v >= min_cooccurrence}

    if not cooc:
        return sp.csr_matrix((n, n), dtype=np.float32), pid_idx

    # Build PPMI weights
    rows, cols, data = [], [], []
    for (a, b), cnt in cooc.items():
        ia, ib = pid_to_idx[a], pid_to_idx[b]
        p_a = product_basket_counts.get(a, 0) / n_baskets
        p_b = product_basket_counts.get(b, 0) / n_baskets
        p_ab = cnt / n_baskets
        if p_a > 0 and p_b > 0:
            pmi = np.log(p_ab / (p_a * p_b))
            ppmi = max(pmi, 0.0)
        else:
            ppmi = 0.0
        if ppmi > 0:
            # Symmetric
            rows += [ia, ib]
            cols += [ib, ia]
            data += [ppmi, ppmi]

    A = sp.csr_matrix(
        (np.array(data, dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )

    # Normalise weights to [0, 1]
    max_val = A.data.max() if A.nnz > 0 else 1.0
    if max_val > 0:
        A.data /= max_val

    return A, pid_idx
