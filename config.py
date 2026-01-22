from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict

PROJECT_ROOT: Path = Path(__file__).resolve().parent

DEFAULT_ZIP_PATH = Path(r"C:\Users\dansu\Downloads\networks.zip")
ZIP_PATH: Path = Path(os.environ.get("COMP511_NETWORKS_ZIP", str(DEFAULT_ZIP_PATH)))

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"
CACHE_DIR: Path = RESULTS_DIR / "cache"

DATASETS = {
    "powergrid": "powergrid.edgelist.txt",
    "collaboration": "collaboration.edgelist.txt",
    "email": "email.edgelist.txt",
}

RANDOM_SEED: int = 0

#degree power-law fit: choose tail using percentile
DEGREE_TAIL_PERCENTILE: int = 80  #fit only k >= percentile(deg, 80)

#eigen spectrum: compute top-k eigenvalues
EIGEN_TOP_K: int = 100

#limit number of points for readability for dense scatters
MAX_SCATTER_POINTS: int = 200_000

#plotting safeguards
EPS: float = 1e-12

def sampling_plan(n_nodes_gcc: int, n_edges_gcc: int) -> Dict[str, int]:
    s_sources = min(1000, max(200, int(0.002 * n_nodes_gcc)))
    t_nodes = min(10_000, max(2_000, int(0.01 * n_nodes_gcc)))
    e_edges_plot = min(MAX_SCATTER_POINTS, max(10_000, n_edges_gcc))

    return {
        "sp_sources": int(s_sources),
        "clust_nodes": int(t_nodes),
        "edges_plot": int(e_edges_plot),
    }

def ensure_project_dirs() -> None:
    for d in [DATA_DIR, RAW_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def assert_zip_exists() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Could not find networks.zip at:\n  {ZIP_PATH}\n\n"
            "Fix options:\n"
            f"  1) Put the zip there, OR\n"
            f"  2) Edit DEFAULT_ZIP_PATH in config.py, OR\n"
            f"  3) Set environment variable COMP511_NETWORKS_ZIP to the correct path.\n"
        )
ensure_project_dirs()