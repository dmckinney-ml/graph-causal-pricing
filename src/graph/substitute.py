"""
Substitute graph via product hierarchy + embedding reweighting.

Strategy (hybrid):
  1. Coarse pass  – connect products sharing the same SUB_COMMODITY_DESC
                    (binary adjacency, capped at `max_neighbors` per node).
  2. Fine-grained – reweight each coarse edge by cosine similarity of
                    Node2Vec embeddings, so that products with similar purchase
                    patterns get higher weights even within the same sub-category.

Edge weight: w_sub(A,B) = cosine_sim(emb_A, emb_B)   [if embeddings provided]
             w_sub(A,B) = 1.0                          [fallback: no embeddings]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def build_substitute_adjacency(
    product_path: str,
    product_ids: list[int] | None = None,
    embeddings: np.ndarray | None = None,
    emb_pid_idx: np.ndarray | None = None,
    max_neighbors: int = 50,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build the substitute adjacency matrix.

    Parameters
    ----------
    product_path  : path to product.csv
    product_ids   : ordered list of product IDs (defines matrix index).
                    If None, use all products in product.csv.
    embeddings    : (M × D) float array of node embeddings (optional).
                    If provided, edges are reweighted by cosine similarity.
    emb_pid_idx   : (M,) int array mapping embedding rows → PRODUCT_IDs.
                    Required when `embeddings` is not None.
    max_neighbors : maximum edges per node (top-k by cosine sim within
                    same sub-commodity).

    Returns
    -------
    A_sub   : (N × N) sparse CSR matrix of substitute edge weights
    pid_idx : 1-D array of product IDs corresponding to rows/columns
    """
    product = pd.read_csv(product_path)
    product.columns = product.columns.str.strip()
    product = product[["PRODUCT_ID", "SUB_COMMODITY_DESC"]].dropna(
        subset=["SUB_COMMODITY_DESC"]
    )

    if product_ids is not None:
        product = product[product["PRODUCT_ID"].isin(product_ids)]

    all_pids = sorted(product["PRODUCT_ID"].unique()) if product_ids is None else list(product_ids)
    n = len(all_pids)
    pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}
    pid_idx = np.array(all_pids, dtype=np.int64)

    # Build embedding lookup (pid → normalised embedding row)
    emb_lookup: dict[int, np.ndarray] | None = None
    if embeddings is not None and emb_pid_idx is not None:
        normed = normalize(embeddings, norm="l2")
        emb_lookup = {int(emb_pid_idx[i]): normed[i] for i in range(len(emb_pid_idx))}

    rows, cols, data = [], [], []

    for _, group in product.groupby("SUB_COMMODITY_DESC", observed=True):
        pids_in_group = group["PRODUCT_ID"].values
        # Restrict to pids that are in our index
        pids_in_group = [p for p in pids_in_group if p in pid_to_idx]
        if len(pids_in_group) < 2:
            continue

        for pid_a in pids_in_group:
            ia = pid_to_idx[pid_a]
            candidates = [p for p in pids_in_group if p != pid_a]

            if emb_lookup is not None:
                # Score by cosine similarity; skip pairs with missing embeddings
                scored = []
                emb_a = emb_lookup.get(pid_a)
                for pid_b in candidates:
                    emb_b = emb_lookup.get(pid_b)
                    if emb_a is not None and emb_b is not None:
                        sim = float(np.dot(emb_a, emb_b))
                        sim = max(sim, 0.0)  # clamp negative cosine to 0
                        scored.append((pid_b, sim))
                    else:
                        scored.append((pid_b, 1.0))  # fallback weight
                # Top-k only
                scored.sort(key=lambda x: x[1], reverse=True)
                scored = scored[:max_neighbors]
            else:
                scored = [(pid_b, 1.0) for pid_b in candidates[:max_neighbors]]

            for pid_b, weight in scored:
                if weight > 0:
                    ib = pid_to_idx[pid_b]
                    rows.append(ia)
                    cols.append(ib)
                    data.append(weight)

    if not data:
        return sp.csr_matrix((n, n), dtype=np.float32), pid_idx

    A = sp.csr_matrix(
        (np.array(data, dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )
    # Symmetrise by taking max of (i,j) and (j,i)
    A = A.maximum(A.T)

    return A, pid_idx
