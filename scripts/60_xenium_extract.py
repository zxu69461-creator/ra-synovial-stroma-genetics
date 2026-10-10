"""60_xenium_extract.py — compact the Xenium 5k bundles (Zenodo 15058262) into one AnnData, then delete the tarballs.

Kept per cell: counts (cell_feature_matrix, gene-expression features only), centroid x/y, cell/nucleus area,
transcript counts, patient id; per patient: clinical metadata from clin_xenium. Dropped: morphology OME-TIFFs,
transcripts.parquet, boundaries (re-downloadable from Zenodo if subcellular analyses are ever needed).
Each tarball's SHA256 + URL is recorded in _raw_files before deletion.

Output: db/objects/xenium/xenium_pretreatment.h5ad; DuckDB: xenium_sample_qc
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import OBJECTS, read, register_raw, write_object, write_table  # noqa: E402

SRC = ROOT / "raw" / "xenium" / "zenodo15058262"
TMP = ROOT / "raw" / "xenium" / "_extract"
OUT = OBJECTS / "xenium" / "xenium_pretreatment.h5ad"
KEEP = ["cell_feature_matrix/matrix.mtx.gz", "cell_feature_matrix/features.tsv.gz",
        "cell_feature_matrix/barcodes.tsv.gz", "cells.parquet", "gene_panel.json"]


def one(tar: Path) -> ad.AnnData:
    name = tar.name.replace(".tar.gz", "")
    patient = int(name.split("patient")[1].split("_")[0])
    register_raw(tar, "Zenodo 15058262 (Xenium RA synovium)", f"https://zenodo.org/records/15058262/files/{tar.name}")
    TMP.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(tar), "-C", str(TMP), *[f"{name}/{k}" for k in KEEP]], check=True)
    d = TMP / name
    a = sc.read_10x_mtx(d / "cell_feature_matrix", gex_only=True)
    cells = pd.read_parquet(d / "cells.parquet").set_index("cell_id")
    cells.index = cells.index.astype(str)
    a.obs = a.obs.join(cells[[c for c in ["x_centroid", "y_centroid", "transcript_counts", "cell_area",
                                           "nucleus_area", "total_counts"] if c in cells]], how="left")
    a.obsm["spatial"] = a.obs[["x_centroid", "y_centroid"]].to_numpy()
    a.obs["patient"] = patient
    a.obs_names = [f"p{patient}_{b}" for b in a.obs_names]
    panel = json.loads((d / "gene_panel.json").read_text())
    a.uns[f"panel_p{patient}"] = panel.get("payload", {}).get("panel", {}).get("identity", {}).get("name", "")
    shutil.rmtree(d)
    return a


def main():
    tars = sorted(SRC.glob("xenium_output_patient*_pre_treatment.tar.gz"))
    parts = []
    for t in tars:
        a = one(t)
        parts.append(a)
        print(f"{t.name}: {a.n_obs} cells x {a.n_vars} genes", flush=True)
    x = ad.concat(parts, join="inner", merge="same", uns_merge="unique")
    clin = read("clin_xenium").set_index("patient_id")
    for c in ["pathotype", "response", "treatment", "das28_esr_baseline", "das28_esr_6_months"]:
        if c in clin:
            x.obs[c] = x.obs.patient.map(clin[c]).values
    x.obs["patient"] = x.obs.patient.astype(str).astype("category")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    x.write_h5ad(OUT, compression="gzip")
    write_object(OUT, "xenium_pretreatment", script=__file__, datasets=["Zenodo 15058262"], inputs=[],
                 n_rows=x.n_obs, note="17 pre-treatment RA synovia, Xenium 5k; counts + centroids + clinical")
    qc = x.obs.groupby("patient", observed=True).agg(n_cells=("transcript_counts", "size"),
                                                     median_transcripts=("transcript_counts", "median")).reset_index()
    write_table(qc, "xenium_sample_qc", script=__file__, datasets=["Zenodo 15058262"])
    # tarballs are registered (sha256 + url) and fully represented by OUT for our analyses -> free 27 GB
    for t in tars:
        t.unlink()
    shutil.rmtree(TMP, ignore_errors=True)
    print(x, flush=True)


if __name__ == "__main__":
    main()
