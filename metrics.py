"""
Implements the required graph properties (a)-(g)

Each metric is implemented as a reusable function that takes a graph as input
Plots are produced with matplotlib and can be saved by passing save_path
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from config import EPS, DEGREE_TAIL_PERCENTILE, EIGEN_TOP_K, MAX_SCATTER_POINTS

def _ensure_csr_binary_undirected(A: sp.spmatrix) -> sp.csr_matrix:
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    A = A.copy()
    A.setdiag(0)
    A.eliminate_zeros()
    if A.nnz > 0:
        A.data[:] = 1

    if not A.has_sorted_indices:
        A.sort_indices()
    if (A != A.T).nnz != 0:
        raise ValueError("Adjacency must be symmetric (undirected).")
    if A.diagonal().sum() != 0:
        raise ValueError("Adjacency must have no self-loops.")
    if A.nnz > 0 and np.any(A.data != 1):
        raise ValueError("Adjacency must be unweighted/binary (0/1).")
    return A


def degrees(A: sp.csr_matrix) -> np.ndarray:
    deg = np.asarray(A.sum(axis=1)).ravel()
    return deg.astype(np.int64, copy=False)


def undirected_edge_count(A: sp.csr_matrix) -> int:
    return int(A.nnz // 2)


def savefig(path: Path, dpi: int = 160) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _maybe_sample_indices(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    k = int(min(max(k, 0), n))
    if k == n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False).astype(np.int64, copy=False)

def basic_stats(A: sp.csr_matrix) -> Dict[str, int]:
    A = _ensure_csr_binary_undirected(A)
    n = int(A.shape[0])
    m = undirected_edge_count(A)

    n_components, labels = connected_components(A, directed=False, return_labels=True)
    counts = np.bincount(labels)
    gcc_size = int(counts.max()) if counts.size > 0 else 0

    return {
        "n": n,
        "m": m,
        "n_components": int(n_components),
        "gcc_size": gcc_size,
    }

@dataclass(frozen=True)
class DegreeFitResult:
    slope: float
    intercept: float
    kmin: int


def degree_distribution(A: sp.csr_matrix) -> Dict[str, np.ndarray]:
    A = _ensure_csr_binary_undirected(A)
    deg = degrees(A)

    max_k = int(deg.max()) if deg.size else 0
    counts = np.bincount(deg, minlength=max_k + 1)
    p = counts / max(1, deg.size)

    k_vals = np.arange(len(p), dtype=np.int64)
    return {"k": k_vals, "p": p.astype(float), "deg": deg}


def powerlaw_fit_loglog(
    k: np.ndarray,
    p: np.ndarray,
    deg: Optional[np.ndarray] = None,
    tail_percentile: int = DEGREE_TAIL_PERCENTILE,
    n_log_bins: int = 12,
) -> DegreeFitResult:
    k = np.asarray(k)
    p = np.asarray(p, dtype=float)
    if deg is not None:
        deg = np.asarray(deg, dtype=np.int64)
        deg_pos = deg[deg >= 1]
        if deg_pos.size < 10:
            return DegreeFitResult(slope=np.nan, intercept=np.nan, kmin=1)
        kmin = int(np.percentile(deg_pos, tail_percentile))
        kmin = max(1, kmin)
        kmax = int(deg_pos.max())
    else:
        mask = (k >= 1) & (p > 0)
        kk = k[mask]
        if kk.size < 10:
            return DegreeFitResult(slope=np.nan, intercept=np.nan, kmin=1)
        kmin = int(np.percentile(kk, tail_percentile))
        kmin = max(1, kmin)
        kmax = int(kk.max())

    if kmax <= kmin:
        return DegreeFitResult(slope=np.nan, intercept=np.nan, kmin=kmin)

    edges = np.logspace(np.log10(kmin), np.log10(kmax + 1e-9), n_log_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    y = np.zeros(n_log_bins, dtype=float)
    for i in range(n_log_bins):
        lo = int(np.floor(edges[i]))
        hi = int(np.floor(edges[i + 1]))
        lo = max(lo, 1)
        hi = max(hi, lo + 1)
        hi = min(hi, len(p))
        if lo < hi:
            y[i] = p[lo:hi].sum()

    mask_bin = y > 0
    x_fit = centers[mask_bin]
    y_fit = y[mask_bin]

    if x_fit.size < 5:
        mask2 = (k >= kmin) & (k >= 1) & (p > 0)
        kt = k[mask2].astype(float)
        pt = p[mask2]
        if kt.size < 5:
            return DegreeFitResult(slope=np.nan, intercept=np.nan, kmin=kmin)
        slope, intercept = np.polyfit(np.log(kt + EPS), np.log(pt + EPS), 1)
        return DegreeFitResult(slope=float(slope), intercept=float(intercept), kmin=int(kmin))

    slope, _intercept_bin = np.polyfit(np.log(x_fit + EPS), np.log(y_fit + EPS), 1)
    slope = float(slope)
    mask_raw = (k >= kmin) & (k >= 1) & (p > 0)
    kt_raw = k[mask_raw].astype(float)
    pt_raw = p[mask_raw].astype(float)

    if kt_raw.size >= 5:
        x = np.log(kt_raw + EPS)
        ylog = np.log(pt_raw + EPS)
        intercept = float(ylog.mean() - slope * x.mean())
    else:
        intercept = float(_intercept_bin)

    return DegreeFitResult(slope=slope, intercept=intercept, kmin=int(kmin))

def plot_degree_distribution(k: np.ndarray, p: np.ndarray, fit: DegreeFitResult, save_path: Optional[Path] = None, title: str = "") -> None:
    k = np.asarray(k)
    p = np.asarray(p, dtype=float)

    mask = (k >= 1) & (p > 0)
    kk = k[mask]
    pp = p[mask]

    plt.figure()
    plt.loglog(kk, pp, marker="o", linestyle="none", markersize=3)
    if np.isfinite(fit.slope):
        kline = kk[kk >= fit.kmin]
        if kline.size >= 2:
            yline = np.exp(fit.intercept + fit.slope * np.log(kline.astype(float) + EPS))
            plt.loglog(kline, yline, linestyle="-")

    plt.xlabel("Degree k")
    plt.ylabel("P(k)")
    if title:
        plt.title(title)

    if save_path is not None:
        savefig(save_path)
    else:
        plt.show()

def shortest_paths_distribution(
    A: sp.csr_matrix,
    rng: np.random.Generator,
    n_sources: int = 500,
    save_path: Optional[Path] = None,
    title: str = "",
) -> Dict[str, np.ndarray | float | int]:
    A = _ensure_csr_binary_undirected(A)
    n = A.shape[0]
    sources = _maybe_sample_indices(rng, n, n_sources)
    n_sources_used = int(sources.size)

    dist_mat = shortest_path(A, directed=False, unweighted=True, indices=sources)
    d = np.asarray(dist_mat).ravel()
    d = d[np.isfinite(d)]
    d = d[(d > 0)]

    if d.size == 0:
        distances = np.array([], dtype=np.int64)
        prob = np.array([], dtype=float)
        avg_dist = np.nan
    else:
        d_int = d.astype(np.int64, copy=False)
        max_d = int(d_int.max())
        counts = np.bincount(d_int, minlength=max_d + 1)
        counts[0] = 0
        total = counts.sum()
        distances = np.nonzero(counts)[0].astype(np.int64)
        prob = (counts[distances] / max(1, total)).astype(float)
        avg_dist = float((distances * prob).sum())

    if save_path is not None:
        plt.figure()
        if distances.size > 0:
            plt.plot(distances, prob, marker="o", linestyle="-")
        plt.xlabel("Shortest path length ℓ")
        plt.ylabel("Probability")
        if title:
            plt.title(title)
        savefig(save_path)

    return {
        "distances": distances,
        "prob": prob,
        "avg_dist": float(avg_dist),
        "n_sources_used": n_sources_used,
    }

def local_clustering_coefficients(
    A: sp.csr_matrix,
    rng: np.random.Generator,
    n_nodes_sample: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    A = _ensure_csr_binary_undirected(A)
    n = A.shape[0]
    nodes = _maybe_sample_indices(rng, n, n_nodes_sample)

    deg = degrees(A)
    c = np.zeros(nodes.size, dtype=float)

    #Compute clustering by induced neighbor subgraph count
    for idx, i in enumerate(nodes):
        k = int(deg[i])
        if k < 2:
            c[idx] = 0.0
            continue

        nbrs = A.indices[A.indptr[i] : A.indptr[i + 1]]
        sub = A[nbrs][:, nbrs]
        e = int(sub.nnz // 2)
        c[idx] = (2.0 * e) / (k * (k - 1))

    return nodes, c


def clustering_distribution(
    A: sp.csr_matrix,
    rng: np.random.Generator,
    n_nodes_sample: int = 5000,
    n_bins: int = 30,
    save_path: Optional[Path] = None,
    title: str = "",
) -> Dict[str, np.ndarray | float | int]:
    nodes, c = local_clustering_coefficients(A, rng=rng, n_nodes_sample=n_nodes_sample)
    mean_c = float(np.mean(c)) if c.size else np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    hist, edges = np.histogram(c, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if save_path is not None:
        plt.figure()
        plt.plot(centers, hist, marker="o", linestyle="-")
        plt.xlabel("Local clustering coefficient c")
        plt.ylabel("Density")
        if title:
            plt.title(title)
        savefig(save_path)

    return {
        "nodes": nodes,
        "c_values": c,
        "mean_c": mean_c,
        "bin_centers": centers,
        "density": hist,
        "n_nodes_used": int(nodes.size),
    }

def eigen_spectrum(
    A: sp.csr_matrix,
    k: int = EIGEN_TOP_K,
    save_path: Optional[Path] = None,
    title: str = "",
) -> Dict[str, np.ndarray | float | int]:
    from scipy.sparse.linalg import ArpackNoConvergence

    A = _ensure_csr_binary_undirected(A)
    n = A.shape[0]
    if n < 3:
        return {"eigenvalues": np.array([], dtype=float), "spectral_gap": np.nan, "k_used": 0}

    k_used = int(min(k, n - 2))
    Af = A.astype(float)

    vals = None
    try:
        vals = eigsh(Af, k=k_used, which="LM", return_eigenvectors=False, tol=1e-3, maxiter=5000)
    except ArpackNoConvergence as e:
        if e.eigenvalues is not None and len(e.eigenvalues) >= 2:
            vals = e.eigenvalues
            k_used = int(len(vals))
        else:
            k_used = int(min(30, n - 2))
            vals = eigsh(Af, k=k_used, which="LM", return_eigenvectors=False, tol=1e-2, maxiter=10000)

    vals = np.asarray(vals, dtype=float)
    order = np.argsort(np.abs(vals))[::-1]
    vals_sorted = vals[order]

    spectral_gap = float(abs(vals_sorted[0]) - abs(vals_sorted[1])) if vals_sorted.size >= 2 else np.nan

    if save_path is not None:
        plt.figure()
        ranks = np.arange(1, vals_sorted.size + 1, dtype=np.int64)
        plt.plot(ranks, np.abs(vals_sorted), marker="o", linestyle="-")
        plt.xlabel("Rank (by |λ|)")
        plt.ylabel("|Eigenvalue|")
        if title:
            plt.title(title)
        savefig(save_path)

    return {"eigenvalues": vals_sorted, "spectral_gap": spectral_gap, "k_used": k_used}

def degree_correlation(
    A: sp.csr_matrix,
    rng: np.random.Generator,
    max_points: int = MAX_SCATTER_POINTS,
    save_path: Optional[Path] = None,
    title: str = "",
) -> Dict[str, float | int]:
    A = _ensure_csr_binary_undirected(A)
    deg = degrees(A)

    U = sp.triu(A, k=1).tocoo()
    u = U.row.astype(np.int64, copy=False)
    v = U.col.astype(np.int64, copy=False)
    m_total = int(u.size)

    if m_total == 0:
        return {"corr": np.nan, "n_edges_used": 0, "n_edges_total": 0}

    du_all = deg[u].astype(float, copy=False)
    dv_all = deg[v].astype(float, copy=False)
    corr = float(np.corrcoef(du_all, dv_all)[0, 1]) if du_all.size >= 2 else np.nan

    if m_total > max_points:
        idx = rng.choice(m_total, size=max_points, replace=False)
        du = du_all[idx]
        dv = dv_all[idx]
        m_used = int(max_points)
    else:
        du = du_all
        dv = dv_all
        m_used = m_total

    if save_path is not None:
        plt.figure()
        if du.size > 50_000:
            plt.hexbin(du, dv, gridsize=60, bins="log")
        else:
            plt.scatter(du, dv, s=4, alpha=0.35)

        plt.xlabel("Degree of node i (d_i)")
        plt.ylabel("Degree of node j (d_j)")
        if title:
            plt.title(title)
        savefig(save_path)

    return {"corr": corr, "n_edges_used": m_used, "n_edges_total": m_total}

def degree_clustering_relation(
    A: sp.csr_matrix,
    rng: np.random.Generator,
    n_nodes_sample: int = 5000,
    save_path: Optional[Path] = None,
    title: str = "",
    add_binned_curve: bool = True,
) -> Dict[str, float | int]:
    A = _ensure_csr_binary_undirected(A)
    deg = degrees(A)

    nodes, c = local_clustering_coefficients(A, rng=rng, n_nodes_sample=n_nodes_sample)
    d = deg[nodes].astype(float, copy=False)

    if save_path is not None:
        plt.figure()
        if d.size > 50_000:
            plt.hexbin(d, c, gridsize=60, bins="log")
        else:
            plt.scatter(d, c, s=6, alpha=0.35)

        if add_binned_curve and d.size >= 200:
            dpos = d[d > 0]
            if dpos.size > 0:
                dmin, dmax = float(dpos.min()), float(dpos.max())
                nb = 12
                bins = np.logspace(np.log10(dmin), np.log10(dmax + EPS), nb)
                bin_idx = np.digitize(d, bins)
                x_curve = []
                y_curve = []
                for b in range(1, nb):
                    mask = bin_idx == b
                    if np.any(mask):
                        x_curve.append(float(np.mean(d[mask])))
                        y_curve.append(float(np.mean(c[mask])))
                if len(x_curve) >= 2:
                    plt.plot(x_curve, y_curve, marker="o", linestyle="-")

        plt.xlabel("Degree d")
        plt.ylabel("Local clustering coefficient c")
        if title:
            plt.title(title)
        savefig(save_path)

    return {"n_nodes_used": int(nodes.size)}

def compute_all_metrics(
    A: sp.csr_matrix,
    rng: np.random.Generator,
    sp_sources: int,
    clust_nodes: int,
    out_prefix: Optional[Path] = None,
    name_for_titles: str = "",
) -> Dict[str, object]:
    A = _ensure_csr_binary_undirected(A)
    deg = degrees(A)

    a = basic_stats(A)

    dd = degree_distribution(A)
    fit = powerlaw_fit_loglog(dd["k"], dd["p"], tail_percentile=DEGREE_TAIL_PERCENTILE)

    if out_prefix is not None:
        plot_degree_distribution(
            dd["k"], dd["p"], fit,
            save_path=Path(str(out_prefix) + "_b_degree.png"),
            title=f"{name_for_titles} Degree distribution"
        )

    c = shortest_paths_distribution(
        A, rng=rng, n_sources=sp_sources,
        save_path=(Path(str(out_prefix) + "_c_shortestpaths.png") if out_prefix else None),
        title=f"{name_for_titles} Shortest paths (sampled)"
    )

    d = clustering_distribution(
        A, rng=rng, n_nodes_sample=clust_nodes,
        save_path=(Path(str(out_prefix) + "_d_clustering.png") if out_prefix else None),
        title=f"{name_for_titles} Local clustering (sampled)"
    )

    e = eigen_spectrum(
        A, k=EIGEN_TOP_K,
        save_path=(Path(str(out_prefix) + "_e_eigenspec.png") if out_prefix else None),
        title=f"{name_for_titles} Eigen spectrum"
    )

    f = degree_correlation(
        A, rng=rng,
        save_path=(Path(str(out_prefix) + "_f_degreecorr.png") if out_prefix else None),
        title=f"{name_for_titles} Degree correlation (sampled edges)"
    )

    g = degree_clustering_relation(
        A, rng=rng, n_nodes_sample=clust_nodes,
        save_path=(Path(str(out_prefix) + "_g_deg_vs_clust.png") if out_prefix else None),
        title=f"{name_for_titles} Degree vs clustering"
    )

    return {
        "n": a["n"],
        "m": a["m"],
        "n_components": a["n_components"],
        "gcc_size": a["gcc_size"],
        "degree_fit_slope": fit.slope,
        "degree_fit_kmin": fit.kmin,
        "avg_shortest_path": c["avg_dist"],
        "avg_clustering": d["mean_c"],
        "spectral_gap": e["spectral_gap"],
        "deg_corr": f["corr"],
        "sp_sources_used": c["n_sources_used"],
        "clust_nodes_used": d["n_nodes_used"],
    }