from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from config import (
    RAW_DIR,
    DATASETS,
    FIGURES_DIR,
    TABLES_DIR,
    RANDOM_SEED,
    sampling_plan,
    ensure_project_dirs,
)
from load_graph import load_graph, giant_connected_component
from metrics import (
    basic_stats,
    degree_distribution,
    powerlaw_fit_loglog,
    plot_degree_distribution,
    shortest_paths_distribution,
    clustering_distribution,
    eigen_spectrum,
    degree_correlation,
    degree_clustering_relation,
    undirected_edge_count,
)
from ba_model import generate_ba_graph


def find_dataset_file(filename: str) -> Path:
    matches = list(Path(RAW_DIR).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {RAW_DIR}")
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


@dataclass(frozen=True)
class AnalysisResult:
    summary: Dict[str, object]
    A_full: sp.csr_matrix
    A_gcc: sp.csr_matrix


def analyze_one_graph(
    name: str,
    A_full: sp.csr_matrix,
    rng: np.random.Generator,
    out_prefix: Path,
) -> AnalysisResult:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    a = basic_stats(A_full)

    A_gcc, gcc_nodes = giant_connected_component(A_full)
    n_gcc = int(A_gcc.shape[0])
    m_gcc = undirected_edge_count(A_gcc)

    plan = sampling_plan(n_gcc, m_gcc)
    sp_sources = plan["sp_sources"]
    clust_nodes = plan["clust_nodes"]

    dd = degree_distribution(A_full)
    fit = powerlaw_fit_loglog(dd["k"], dd["p"], deg=dd["deg"])
    plot_degree_distribution(
        dd["k"], dd["p"], fit,
        save_path=Path(str(out_prefix) + "_b_degree.png"),
        title=f"{name}: degree distribution (full graph)"
    )

    c = shortest_paths_distribution(
        A_gcc, rng=rng, n_sources=sp_sources,
        save_path=Path(str(out_prefix) + "_c_shortestpaths.png"),
        title=f"{name}: shortest paths (GCC, sampled sources)"
    )

    d = clustering_distribution(
        A_gcc, rng=rng, n_nodes_sample=clust_nodes,
        save_path=Path(str(out_prefix) + "_d_clustering.png"),
        title=f"{name}: local clustering (GCC, sampled nodes)"
    )

    e = eigen_spectrum(
        A_gcc,
        save_path=Path(str(out_prefix) + "_e_eigenspec.png"),
        title=f"{name}: eigen spectrum (GCC)"
    )

    f = degree_correlation(
        A_full, rng=rng,
        max_points=plan["edges_plot"],
        save_path=Path(str(out_prefix) + "_f_degreecorr.png"),
        title=f"{name}: degree correlation (full graph, sampled edges)"
    )

    g = degree_clustering_relation(
        A_gcc, rng=rng, n_nodes_sample=clust_nodes,
        save_path=Path(str(out_prefix) + "_g_deg_vs_clust.png"),
        title=f"{name}: degree vs clustering (GCC, sampled nodes)"
    )

    summary = {
        "dataset": name,
        "n_full": a["n"],
        "m_full": a["m"],
        "n_components": a["n_components"],
        "gcc_size": a["gcc_size"],
        "n_gcc": n_gcc,
        "m_gcc": m_gcc,
        "powerlaw_slope": fit.slope,
        "powerlaw_kmin": fit.kmin,
        "avg_shortest_path_gcc": c["avg_dist"],
        "sp_sources_used": c["n_sources_used"],
        "avg_clustering_gcc": d["mean_c"],
        "clust_nodes_used": d["n_nodes_used"],
        "spectral_gap_gcc": e["spectral_gap"],
        "deg_corr_full": f["corr"],
        "edges_used_for_corr": f["n_edges_used"],
        "edges_total_for_corr": f["n_edges_total"],
    }

    return AnalysisResult(summary=summary, A_full=A_full, A_gcc=A_gcc)


def write_csv(path: Path, rows: list[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def estimate_ba_m_param(n_full: int, m_full: int) -> int:
    m_est = int(round(m_full / max(1, n_full)))
    m_est = max(1, min(m_est, n_full - 1))
    return m_est


def main() -> None:
    ensure_project_dirs()
    rng = np.random.default_rng(RANDOM_SEED)
    q1_rows: list[Dict[str, object]] = []
    real_graphs: Dict[str, sp.csr_matrix] = {}

    for name, fname in DATASETS.items():
        path = find_dataset_file(fname)
        gd = load_graph(path)
        real_graphs[name] = gd.A

        out_prefix = FIGURES_DIR / f"q1_{name}"
        res = analyze_one_graph(name=f"Q1 {name}", A_full=gd.A, rng=rng, out_prefix=out_prefix)
        q1_rows.append(res.summary)

    write_csv(TABLES_DIR / "q1_real_summary.csv", q1_rows)
    print(f"[run_all] Wrote: {TABLES_DIR / 'q1_real_summary.csv'}")

    q3_rows: list[Dict[str, object]] = []

    for name, A_full in real_graphs.items():
        a = basic_stats(A_full)
        n_full = int(a["n"])
        m_full = int(a["m"])
        m_param = estimate_ba_m_param(n_full, m_full)

        ba = generate_ba_graph(n=n_full, m=m_param, seed=RANDOM_SEED)
        out_prefix = FIGURES_DIR / f"q3_ba_{name}"

        res = analyze_one_graph(name=f"Q3 BA({name})", A_full=ba.A, rng=rng, out_prefix=out_prefix)
        res.summary["ba_m_param"] = m_param
        res.summary["target_n_full"] = n_full
        res.summary["target_m_full"] = m_full
        q3_rows.append(res.summary)

    write_csv(TABLES_DIR / "q3_ba_summary.csv", q3_rows)
    print(f"[run_all] Wrote: {TABLES_DIR / 'q3_ba_summary.csv'}")

    print("\n[run_all] Done.")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Tables : {TABLES_DIR}")


if __name__ == "__main__":
    main()