"""10_atlas_micheroli.py — build the open RA synovium scRNA-seq atlas (Fig 1), lineage level.

Input : raw/sc/micheroli2024 (E-MTAB-11791; Micheroli et al. 2024 iScience), filtered MatrixMarket (genes x cells)
Steps : load (chunked, int32) -> per-sample MAD QC -> HVG (seurat_v3, batch=sample) -> scVI (MPS) -> Leiden ->
        lineage calls from canonical markers.
Output: db/objects/atlas/micheroli_lineage.h5ad (counts in .layers['counts'], scVI latent in .obsm['X_scVI'])
        DuckDB: atlas_cells_lineage (per-cell obs), atlas_cluster_markers_lineage, atlas_qc_by_sample
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import OBJECTS, register_raw, write_object, write_table  # noqa: E402

RAW = ROOT / "raw" / "sc" / "micheroli2024"
OUT = OBJECTS / "atlas"
DS = ["E-MTAB-11791 (Micheroli 2024)"]
URL = "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/791/E-MTAB-11791/Files/"

LINEAGE_MARKERS = {
    "T": ["CD3E", "CD3D", "TRAC", "CD2"],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRF1"],
    "B": ["MS4A1", "CD79A", "BANK1", "CD19"],
    "Plasma": ["MZB1", "JCHAIN", "IGHG1", "XBP1"],
    "Myeloid": ["CD14", "C1QA", "CD68", "LYZ", "CSF1R"],
    "DC": ["CLEC10A", "CD1C", "FCER1A", "LAMP3"],
    "Mast": ["TPSAB1", "CPA3", "KIT"],
    "Fibroblast": ["PDGFRA", "COL1A1", "DCN", "PRG4"],
    "Mural": ["ACTA2", "RGS5", "MCAM", "NOTCH3"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "PLVAP"],
    "Cycling": ["MKI67", "TOP2A"],
}


def load_counts() -> ad.AnnData:
    mtx = RAW / "count_matrix_filtered.mtx"
    for f in ["count_matrix_filtered.mtx", "barcodes_filtered.txt", "genes_filtered.txt", "E-MTAB-11791.sdrf.txt"]:
        register_raw(RAW / f, DS[0], URL + f)
    with open(mtx) as fh:
        fh.readline()
        ng, nc, nnz = map(int, fh.readline().split())
    rows, cols, vals = [], [], []
    for ch in pd.read_csv(mtx, sep=" ", skiprows=2, header=None, names=["g", "c", "v"],
                          dtype={"g": np.int32, "c": np.int32, "v": np.int32}, chunksize=20_000_000):
        rows.append(ch.c.values - 1)
        cols.append(ch.g.values - 1)
        vals.append(ch.v.values.astype(np.float32))
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nc, ng))
    del rows, cols, vals
    bc = pd.read_csv(RAW / "barcodes_filtered.txt", header=None)[0].values
    genes = pd.read_csv(RAW / "genes_filtered.txt", header=None)[0].values
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=bc), var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    a.obs["sample"] = a.obs_names.str.split(".").str[0]
    assert a.n_obs == nc and X.nnz == nnz, "matrix size mismatch"
    return a


def mad_outlier(x: pd.Series, n: float) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x < med - n * mad) | (x > med + n * mad)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = load_counts()
    print(a, flush=True)

    a.var["mt"] = a.var_names.str.startswith("MT-")
    a.var["ribo"] = a.var_names.str.startswith(("RPS", "RPL"))
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt", "ribo"], inplace=True, percent_top=None, log1p=True)
    g = a.obs.groupby("sample")
    out = (g.log1p_total_counts.transform(lambda s: mad_outlier(s, 5))
           | g.log1p_n_genes_by_counts.transform(lambda s: mad_outlier(s, 5))
           | (a.obs.pct_counts_mt > 20))
    a.obs["qc_pass"] = ~out.astype(bool)
    qc = a.obs.groupby("sample").agg(n_cells=("qc_pass", "size"), n_pass=("qc_pass", "sum"),
                                     median_umi=("total_counts", "median"), median_genes=("n_genes_by_counts", "median"),
                                     median_pct_mt=("pct_counts_mt", "median")).reset_index()
    write_table(qc, "atlas_qc_by_sample", script=__file__, datasets=DS, note="5-MAD per sample on log counts/genes; mt<20%")
    a = a[a.obs.qc_pass].copy()
    print(f"after QC: {a.n_obs} cells", flush=True)

    a.layers["counts"] = a.X.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat_v3", batch_key="sample", layer="counts")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    import scvi
    scvi.settings.seed = 0
    h = a[:, a.var.highly_variable].copy()
    scvi.model.SCVI.setup_anndata(h, layer="counts", batch_key="sample")
    m = scvi.model.SCVI(h, n_latent=30, n_layers=2, gene_likelihood="nb")
    m.train(max_epochs=200, accelerator="mps", early_stopping=True, batch_size=1024)
    a.obsm["X_scVI"] = m.get_latent_representation()
    m.save(OUT / "micheroli_scvi_model", overwrite=True)

    sc.pp.neighbors(a, use_rep="X_scVI", n_neighbors=30)
    sc.tl.umap(a, min_dist=0.3)
    sc.tl.leiden(a, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2)

    for lin, genes in LINEAGE_MARKERS.items():
        sc.tl.score_genes(a, [g for g in genes if g in a.var_names], score_name=f"lin_{lin}")
    score_cols = [f"lin_{k}" for k in LINEAGE_MARKERS]
    cl = a.obs.groupby("leiden")[score_cols].mean()
    call = cl.idxmax(axis=1).str.replace("lin_", "")
    a.obs["lineage"] = a.obs.leiden.map(call).astype("category")

    sc.tl.rank_genes_groups(a, "leiden", method="wilcoxon", pts=True)
    mk = sc.get.rank_genes_groups_df(a, None)
    mk = mk[(mk.pvals_adj < 0.05) & (mk.logfoldchanges > 0.5)].groupby("group").head(50)
    mk["lineage"] = mk.group.map(call)
    write_table(mk, "atlas_cluster_markers_lineage", script=__file__, datasets=DS, note="Wilcoxon, top50/cluster")

    obs = a.obs[["sample", "leiden", "lineage", "total_counts", "n_genes_by_counts", "pct_counts_mt"]].copy()
    obs[["umap1", "umap2"]] = a.obsm["X_umap"]
    obs = obs.reset_index(names="cell")
    write_table(obs, "atlas_cells_lineage", script=__file__, datasets=DS, note="lineage-level calls from marker scores")

    f = OUT / "micheroli_lineage.h5ad"
    a.write_h5ad(f, compression="gzip")
    write_object(f, "atlas_micheroli_lineage", script=__file__, datasets=DS, n_rows=a.n_obs,
                 note="QC-passed cells; scVI latent (30d, batch=sample); leiden + lineage")
    print(a.obs.lineage.value_counts().to_string(), flush=True)
    print(pd.crosstab(a.obs.leiden, a.obs.lineage).idxmax(axis=1).to_string(), flush=True)


if __name__ == "__main__":
    main()
