"""14_atlas_gupta.py — independent RA-only replication atlas: GSE243917 (Gupta et al. 2023 Nat Genet; Weinand et al.
2024 Nat Commun), single-nucleus multiome of synovial tissue, RNA modality.

Input : raw/sc/gupta2023/GSE243917_genesXcells_ALL_raw.h5ad (31,547 nuclei x 36,601 genes, raw counts; obs ct,
        ct_subtype = the authors' cell-type / fine cell-state labels). Donor = barcode prefix.
Steps : drop the single OA donor (BRI-1281 = GSM7798338 "disease: OA") -> 11 RA donors; keep the authors' fine labels
        unchanged as states (independent annotation); compartment = T/NK (Tcell, NK), B/plasma (Bcell, plasma),
        Myeloid, Endothelial, Stromal (fibroblast, mural); genes detected in >= 10 nuclei.
Output: db/objects/atlas/gupta2023_states.h5ad (layers['counts'], obs sample/state/state_group/n_genes_by_counts)
        Parquet view gupta_state_pseudobulk_long (same definitions as state_pseudobulk_long)
        DuckDB gupta_state_summary (state, state_group, n_cells, n_donors)
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
from archive import OBJECTS, register_raw, write_object, write_parquet_view, write_table  # noqa: E402

RAW = ROOT / "raw/sc/gupta2023/GSE243917_genesXcells_ALL_raw.h5ad"
URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243917/suppl/GSE243917_genesXcells_ALL_raw.h5ad.gz"
COMP = {"Tcell": "T/NK", "NK": "T/NK", "Bcell": "B/plasma", "plasma": "B/plasma", "myeloid": "Myeloid",
        "endothelial": "Endothelial", "fibroblast": "Stromal", "mural": "Stromal"}
OA_DONOR = "BRI-1281"


def main():
    register_raw(Path(str(RAW) + ".gz"), dataset="GSE243917", url=URL)
    a = ad.read_h5ad(RAW)
    a.obs["sample"] = a.obs_names.str.split("_").str[0]
    a = a[a.obs["sample"] != OA_DONOR].copy()
    a.X = sp.csr_matrix(a.X).astype(np.float32)
    a.layers["counts"] = a.X.copy()
    a.obs["state"] = a.obs.ct_subtype.astype(str)
    a.obs["state_group"] = a.obs.ct.astype(str).map(COMP)
    a.obs["n_genes_by_counts"] = np.asarray((a.X > 0).sum(axis=1)).ravel()
    sc.pp.filter_genes(a, min_cells=10)
    out = OBJECTS / "atlas" / "gupta2023_states.h5ad"
    a.write_h5ad(out, compression="gzip")
    write_object(out, "atlas_gupta2023_states", script=__file__, datasets=["GSE243917"],
                 note="11 RA donors (OA donor BRI-1281 removed); authors' fine labels as states; raw counts")
    summ = (a.obs.groupby("state", observed=True)
            .agg(state_group=("state_group", "first"), n_cells=("sample", "size"), n_donors=("sample", "nunique"))
            .reset_index())
    write_table(summ, "gupta_state_summary", script=__file__, datasets=["GSE243917"])
    # pseudobulk (identical definitions to 13_state_pseudobulk.py)
    b = a.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    sc.pp.filter_genes(b, min_cells=20)
    grp = summ.set_index("state").state_group
    means, pcts = {}, {}
    for st in grp.index:
        X = b[(b.obs.state == st).values].X
        means[st] = np.asarray(X.mean(axis=0)).ravel()
        pcts[st] = np.asarray((X > 0).mean(axis=0)).ravel()
    M, P = pd.DataFrame(means, index=b.var_names), pd.DataFrame(pcts, index=b.var_names)
    Z = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1) + 1e-6, axis=0)
    long = (M.stack().rename("mean_lognorm").to_frame().join(P.stack().rename("pct_detected"))
            .join(Z.stack().rename("spec_z")).rename_axis(["gene", "state"]).reset_index())
    long["state_group"] = long.state.map(grp)
    for c in ["mean_lognorm", "pct_detected", "spec_z"]:
        long[c] = long[c].astype("float32")
    write_parquet_view(long, "gupta_state_pseudobulk_long", script=__file__, datasets=["GSE243917"],
                       note="per state x gene: mean log1p CP10k, detection, z across state means (11 RA donors)")
    print(a.shape, summ.groupby("state_group").agg(states=("state", "size"), cells=("n_cells", "sum")).to_dict())


if __name__ == "__main__":
    main()
