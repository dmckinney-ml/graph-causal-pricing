"""
Node2Vec-style product embeddings.

Biased random walks are generated on the PPMI co-occurrence graph, then
Word2Vec (gensim) is trained on the walk corpus.

Node2Vec parameters:
  p  – return parameter   (1.0 = DeepWalk / unbiased; < 1 = BFS-biased)
  q  – in-out parameter   (1.0 = DeepWalk; > 1 = DFS-biased; < 1 = BFS)

Default: p=1, q=1  → DeepWalk (unbiased baseline).
An ablation at p=0.5, q=2 gives DFS-biased walks (structural equivalence).

Performance:
  When p=1 and q=1 the next-hop distribution depends only on edge weights,
  not on the previous node.  Transition probabilities are therefore
  pre-computed once per node and walks are sampled with numpy — roughly
  10–50× faster than the per-step Python loop used by the biased path.
"""

from __future__ import annotations

import random

import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec


# ── fast unbiased (DeepWalk) path ─────────────────────────────────────────────

def _build_csr_neighbor_tables(
    A: sp.spmatrix, pid_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-compute CSR-style neighbor and probability arrays directly from the
    sparse adjacency matrix — no networkx traversal needed at walk time.

    Returns
    -------
    node_ids   : (N,) int64 — product IDs in index order
    indptr     : (N+1,) int64 — CSR row pointers into `neighbors` / `probs`
    neighbors : (nnz,) int64 — neighbor indices (into node_ids)
    probs      : (nnz,) float64 — normalised transition probabilities
    """
    A_csr = A.tocsr().astype(np.float64)
    n = len(pid_idx)
    indptr = A_csr.indptr.copy().astype(np.int64)
    neighbors = A_csr.indices.copy().astype(np.int64)

    # Normalise each row to a proper probability distribution
    probs = A_csr.data.copy()
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        row_sum = probs[s:e].sum()
        if row_sum > 0:
            probs[s:e] /= row_sum

    return pid_idx.astype(np.int64), indptr, neighbors, probs


def _generate_walks_unbiased(
    pid_idx: np.ndarray,
    indptr: np.ndarray,
    neighbors: np.ndarray,
    probs: np.ndarray,
    num_walks: int,
    walk_length: int,
    seed: int = 42,
) -> list[list[str]]:
    """Fast DeepWalk-style walks using pre-computed per-node probabilities."""
    rng = np.random.default_rng(seed)
    n = len(pid_idx)
    node_order = np.arange(n)
    walks: list[list[str]] = []

    for _ in range(num_walks):
        rng.shuffle(node_order)
        for start_idx in node_order:
            walk = [start_idx]
            cur = int(start_idx)
            for _ in range(walk_length - 1):
                s, e = indptr[cur], indptr[cur + 1]
                if s == e:          # isolated node
                    break
                nxt = int(rng.choice(neighbors[s:e], p=probs[s:e]))
                walk.append(nxt)
                cur = nxt
            walks.append([str(int(pid_idx[idx])) for idx in walk])

    return walks


# ── slow biased path (p ≠ 1 or q ≠ 1) ────────────────────────────────────────

def _build_nx_graph(A: sp.spmatrix, pid_idx: np.ndarray):
    """Build a networkx Graph from a sparse adjacency (used for biased walks)."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(pid_idx)))
    cx = A.tocoo()
    for i, j, v in zip(cx.row.tolist(), cx.col.tolist(), cx.data.tolist()):
        if i < j and v > 0:
            G.add_edge(i, j, weight=float(v))
    return G


def _biased_walk_idx(
    G, start: int, length: int, p: float, q: float
) -> list[int]:
    walk = [start]
    while len(walk) < length:
        cur = walk[-1]
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break
        if len(walk) == 1:
            weights = [G[cur][nb].get("weight", 1.0) for nb in nbrs]
        else:
            prev = walk[-2]
            weights = []
            for nb in nbrs:
                w = G[cur][nb].get("weight", 1.0)
                if nb == prev:
                    weights.append(w / p)
                elif G.has_edge(nb, prev):
                    weights.append(w)
                else:
                    weights.append(w / q)
        total = sum(weights)
        if total == 0:
            break
        probs = [ww / total for ww in weights]
        walk.append(random.choices(nbrs, weights=probs)[0])  # noqa: S311
    return walk


def _generate_walks_biased(
    A: sp.spmatrix,
    pid_idx: np.ndarray,
    num_walks: int,
    walk_length: int,
    p: float,
    q: float,
    seed: int = 42,
) -> list[list[str]]:
    random.seed(seed)
    G = _build_nx_graph(A, pid_idx)
    nodes = list(G.nodes())
    walks: list[list[str]] = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = _biased_walk_idx(G, node, walk_length, p, q)
            walks.append([str(int(pid_idx[idx])) for idx in walk])
    return walks


# ── public API ────────────────────────────────────────────────────────────────

def build_graph_from_adjacency(A: sp.spmatrix, pid_idx: np.ndarray):
    """Return a networkx Graph for the sparse adjacency (convenience helper)."""
    return _build_nx_graph(A, pid_idx)


def train_node2vec(
    A: sp.spmatrix,
    pid_idx: np.ndarray,
    dim: int = 64,
    num_walks: int = 10,
    walk_length: int = 80,
    p: float = 1.0,
    q: float = 1.0,
    workers: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train Node2Vec embeddings on a product graph.

    Parameters
    ----------
    A          : (N × N) sparse adjacency matrix
    pid_idx    : (N,) array of product IDs corresponding to A's rows/cols
    dim        : embedding dimension
    num_walks  : number of walks per node
    walk_length: steps per walk
    p, q       : Node2Vec return/in-out parameters
    workers    : Word2Vec training workers
    seed       : random seed

    Returns
    -------
    embeddings : (N × dim) float32 array  (rows aligned with pid_idx)
    pid_idx    : echoed back (same object passed in)
    """
    pid_idx = np.asarray(pid_idx, dtype=np.int64)

    if p == 1.0 and q == 1.0:
        # Fast path: pre-compute per-node probabilities once, walk with numpy
        _, indptr, neighbors_arr, probs_arr = _build_csr_neighbor_tables(A, pid_idx)
        walks = _generate_walks_unbiased(
            pid_idx, indptr, neighbors_arr, probs_arr,
            num_walks, walk_length, seed=seed,
        )
    else:
        # General biased path (slower; only needed for p≠1 or q≠1 ablations)
        walks = _generate_walks_biased(A, pid_idx, num_walks, walk_length, p, q, seed=seed)

    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=5,
        min_count=0,
        sg=1,          # skip-gram
        workers=workers,
        seed=seed,
        epochs=5,
    )

    embeddings = np.zeros((len(pid_idx), dim), dtype=np.float32)
    for i, pid in enumerate(pid_idx):
        key = str(int(pid))
        if key in model.wv:
            embeddings[i] = model.wv[key]

    return embeddings, pid_idx
