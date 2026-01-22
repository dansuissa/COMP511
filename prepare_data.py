from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import ZIP_PATH, RAW_DIR, DATASETS, assert_zip_exists, ensure_project_dirs


def _list_zip_contents(zip_path: Path) -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.namelist()


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _find_file_recursively(root: Path, filename: str) -> Optional[Path]:
    matches = list(root.rglob(filename))
    if not matches:
        return None
    matches.sort(key=lambda p: len(p.parts))  #closest to root first
    return matches[0]


def _candidate_filenames(fname: str) -> List[str]:
    """
    Accept either:
      - foo.edgelist
      - foo.edgelist.txt
    regardless of what is given in config.
    """
    f = fname.strip()
    cands = [f]

    if f.endswith(".txt"):
        cands.append(f[:-4])  #remove .txt
    else:
        cands.append(f + ".txt")

    #de-duplicate while preserving order
    out = []
    seen = set()
    for x in cands:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def main() -> None:
    ensure_project_dirs()
    assert_zip_exists()

    print(f"[prepare_data] Using zip: {ZIP_PATH}")
    print(f"[prepare_data] Extracting into: {RAW_DIR}")

    _extract_zip(ZIP_PATH, RAW_DIR)

    found_map: Dict[str, Path] = {}
    missing: List[Tuple[str, List[str]]] = []

    for ds_name, fname in DATASETS.items():
        cands = _candidate_filenames(fname)
        found = None
        for cand in cands:
            found = _find_file_recursively(RAW_DIR, cand)
            if found is not None:
                break

        if found is None:
            missing.append((ds_name, cands))
        else:
            found_map[ds_name] = found

    if missing:
        print("\n[prepare_data] ERROR: Missing expected files after extraction:")
        for ds_name, cands in missing:
            print(f"  - {ds_name}: tried {cands}")

        print("\n[prepare_data] Zip contents (top-level preview):")
        contents = _list_zip_contents(ZIP_PATH)
        for item in contents[:80]:
            print(f"  {item}")

        raise FileNotFoundError(
            "Could not locate one or more expected edgelist files in the extracted zip."
        )

    print("\n[prepare_data] Success! Found required datasets:")
    for ds_name, path in found_map.items():
        print(f"  - {ds_name:14s} -> {path}")

    print("\n[prepare_data] Next step:")
    print("  We'll now write load_graph.py to load + clean these edgelists into sparse adjacency matrices.")


if __name__ == "__main__":
    main()