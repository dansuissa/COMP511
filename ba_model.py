"""
Albert–Barabási
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class BAGraph:
    A: sp.csr_matrix
    n: int
    m: int  #edge count
    m_param: int


def _build_csr_from_undirected_edges(n: int, u: np.ndarray, v: np.ndarray) -> sp.csr_matrix:
    """
    Build a simple undirected CSR adjacency from edge lists u,v (each edge appears once)
    """
    data = np.ones(u.shape[0], dtype=np.uint8)
    A = sp.coo_matrix((data, (u, v)), shape=(n, n), dtype=np.uint8)

    #symetrize
    A = A + A.T

    #binarize first
    A.data[:] = 1
    A.eliminate_zeros()

    #remove diagonal after binarization 
    A.setdiag(0)
    A.eliminate_zeros()

    A = A.tocsr()
    if not A.has_sorted_indices:
        A.sort_indices()
    if A.nnz > 0:
        A.data[:] = 1
    return A


def generate_ba_graph(n: int, m: int, seed: int = 0) -> BAGraph:
    """
    Generate an Albert–Barabási graph with n nodes and m edges per new node
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if m < 1:
        raise ValueError("m must be >= 1")
    if m >= n:
        raise ValueError("m must be < n")

    rng = np.random.default_rng(seed)

    m0 = m + 1
    if m0 > n:
        m0 = n

    u_list: List[int] = []
    v_list: List[int] = []

    #Initial clique
    for i in range(m0):
        for j in range(i + 1, m0):
            u_list.append(i)
            v_list.append(j)

    deg = np.zeros(n, dtype=np.int64)
    if m0 >= 2:
        deg[:m0] = m0 - 1

    repeated: List[int] = []
    if m0 >= 2:
        for i in range(m0):
            repeated.extend([i] * int(deg[i]))
    else:
        repeated = [0]

    for new_node in range(m0, n):
        targets: Set[int] = set()

        if len(repeated) == 0:
            repeated = list(range(new_node))

        #Sample unique targets proportional to degree
        while len(targets) < m:
            targets.add(repeated[int(rng.integers(0, len(repeated)))])

        for t in targets:
            u_list.append(new_node)
            v_list.append(int(t))
            deg[new_node] += 1
            deg[t] += 1

        repeated.extend([new_node] * m)
        for t in targets:
            repeated.append(int(t))

    u = np.asarray(u_list, dtype=np.int64)
    v = np.asarray(v_list, dtype=np.int64)

    #remove any accidental self-loops
    mask = u != v
    u = u[mask]
    v = v[mask]

    A = _build_csr_from_undirected_edges(n, u, v)

    #sanity checks
    if A.diagonal().sum() != 0:
        raise AssertionError("BA adjacency has self-loops.")
    if (A != A.T).nnz != 0:
        raise AssertionError("BA adjacency is not symmetric.")
    if A.nnz > 0 and np.any(A.data != 1):
        raise AssertionError("BA adjacency is not binary 0/1.")

    m_undirected = int(A.nnz // 2)
    return BAGraph(A=A, n=n, m=m_undirected, m_param=m)
