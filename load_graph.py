"""
Loads an edgelist into a SIMPLE, UNDIRECTED, UNWEIGHTED sparse adjacency matrix
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class GraphData:
    """
    A:CSR adjacency matrix (simple, undirected, unweighted)
    n:number of nodes
    m:number of undirected edges
    """
    A: sp.csr_matrix
    n: int
    m: int


def read_edgelist(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Edgelist not found: {path}")

    edges = np.loadtxt(path, dtype=np.int64)
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    if edges.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in edgelist, got shape {edges.shape} from {path}")
    return edges


def edges_to_simple_undirected_adjacency(edges: np.ndarray) -> sp.csr_matrix:
    if edges.size == 0:
        raise ValueError("Empty edge list.")

    u = edges[:, 0].astype(np.int64, copy=False)
    v = edges[:, 1].astype(np.int64, copy=False)

    #remove self loops from input
    mask = u != v
    u = u[mask]
    v = v[mask]
    if u.size == 0:
        raise ValueError("All edges were self-loops; graph would be empty.")

    n = int(max(u.max(), v.max()) + 1)

    data = np.ones(u.shape[0], dtype=np.uint8)
    A = sp.coo_matrix((data, (u, v)), shape=(n, n), dtype=np.uint8)

    #symmetrize
    A = A + A.T

    #Binarize
    A.data[:] = 1
    A.eliminate_zeros()

    # remove diagonal
    A.setdiag(0)
    A.eliminate_zeros()

    #CSR for efficient row ops
    A = A.tocsr()
    if not A.has_sorted_indices:
        A.sort_indices()

    #final enforce 0/1
    if A.nnz > 0:
        A.data[:] = 1

    return A


def load_graph(path: Path) -> GraphData:
    edges = read_edgelist(path)
    A = edges_to_simple_undirected_adjacency(edges)
    if A.diagonal().sum() != 0:
        raise AssertionError("Adjacency has self-loops after cleaning.")
    if (A != A.T).nnz != 0:
        raise AssertionError("Adjacency is not symmetric after symmetrization.")
    if A.nnz > 0 and np.any(A.data != 1):
        raise AssertionError("Adjacency is not unweighted (0/1).")
    n = A.shape[0]
    m = int(A.nnz // 2)
    return GraphData(A=A, n=n, m=m)


def giant_connected_component(A: sp.csr_matrix) -> Tuple[sp.csr_matrix, np.ndarray]:
    n_components, labels = connected_components(A, directed=False, return_labels=True)
    if n_components == 1:
        nodes = np.arange(A.shape[0], dtype=np.int64)
        return A, nodes

    counts = np.bincount(labels)
    gcc_label = int(np.argmax(counts))
    nodes = np.where(labels == gcc_label)[0].astype(np.int64)

    A_gcc = A[nodes][:, nodes].tocsr()
    A_gcc.setdiag(0)
    A_gcc.eliminate_zeros()
    if A_gcc.nnz > 0:
        A_gcc.data[:] = 1
    if not A_gcc.has_sorted_indices:
        A_gcc.sort_indices()

    return A_gcc, nodes