"""11_fine_states.py — fine cell states within each lineage, named against the AMP2 atlas (Fig 1).

AMP2 data themselves are controlled, but Zhang et al. 2023 Nature Supplementary Table 5 (MOESM5) lists the marker
genes of all 77 AMP2 fine clusters; each of our subclusters is scored against those marker sets and takes the
best-matching AMP2 name (weak matches keep a data-driven name from top markers, flagged `amp2_match=False`).

Per lineage group: HVG (batch=sample) -> lineage-specific scVI -> Leiden -> AMP2 marker-set scoring.
Output: db/objects/atlas/micheroli_fine.h5ad (obs: lineage_group, fine_cluster, fine_state, amp2_match)
DuckDB: atlas_fine_state_map (subcluster -> AMP2 label, score, margin), atlas_cells_fine, atlas_fine_markers
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import OBJECTS, register_raw, write_object, write_table  # noqa: E402

AMP2_SUPP = ROOT / "refs" / "amp2_nature2023_supp" / "41586_2023_6708_MOESM5_ESM.xlsx"
AMP2_URL = ("https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-023-06708-y/"
            "MediaObjects/41586_2023_6708_MOESM5_ESM.xlsx")
GROUPS = {  # our lineage calls -> AMP2 marker sheet
    "T_NK": (["T", "NK"], ["T cell cluster marker", "NK cell cluster marker"]),
    "B_plasma": (["B", "Plasma"], ["B cell cluster marker"]),
    "Myeloid": (["Myeloid", "DC"], ["Myeloid cell cluster marker"]),
    "Stromal": (["Fibroblast", "Mural"], ["Stromal cluster marker"]),
    "Endothelial": (["Endothelial"], ["Endothelial cluster marker"]),
}
DROP = ("MT-high", "low quality", "Proliferating")  # AMP2 technical clusters are not used as reference labels


def amp2_markers(sheets):
    out = {}
    for sh in sheets:
        v = pd.read_excel(AMP2_SUPP, sheet_name=sh, header=None)
        h = v.index[v.iloc[:, 0].astype(str).str.contains("Cluster number")][0]
        d = v.iloc[h + 1:].copy()
        d.columns = v.iloc[h].tolist()
        for name, g in d.groupby("Cluster annotation"):
            if not any(k in name for k in DROP):
                out[name] = g.sort_values("FC", ascending=False)["Gene"].astype(str).head(15).tolist()
    return out


def subcluster(a: ad.AnnData, key: str, res: float):
    import scvi
    a.X = a.layers["counts"].copy()
    # seurat_v3 loess is unstable for tiny per-sample batches (e.g. B/plasma); select HVGs on samples with >=50 cells
    s = a.obs["sample"].astype(str)
    big = s.map(s.value_counts()).astype(int) >= 50
    hv = a[big.values].copy()
    sc.pp.highly_variable_genes(hv, n_top_genes=2000, flavor="seurat_v3", batch_key="sample", layer="counts",
                                span=0.5)
    a.var["highly_variable"] = hv.var["highly_variable"]
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    h = a[:, a.var.highly_variable].copy()
    scvi.model.SCVI.setup_anndata(h, layer="counts", batch_key="sample")
    m = scvi.model.SCVI(h, n_latent=20, n_layers=2)
    m.train(max_epochs=300, accelerator="mps", early_stopping=True, batch_size=512)
    a.obsm["X_scVI_sub"] = m.get_latent_representation()
    sc.pp.neighbors(a, use_rep="X_scVI_sub", n_neighbors=20)
    sc.tl.umap(a, min_dist=0.3)
    sc.tl.leiden(a, resolution=res, key_added=key, flavor="igraph", n_iterations=2)
    return a


def name_clusters(a: ad.AnnData, key: str, markers: dict, group: str) -> pd.DataFrame:
    cols = []
    for name, genes in markers.items():
        g = [x for x in genes if x in a.var_names]
        if len(g) >= 3:
            c = f"amp2::{name}"
            sc.tl.score_genes(a, g, score_name=c)
            cols.append(c)
    M = a.obs.groupby(key)[cols].mean()
    Z = (M - M.mean()) / M.std(ddof=0)          # z across subclusters, per AMP2 state
    best = Z.idxmax(axis=1)
    top2 = np.sort(Z.values, axis=1)[:, -2:]
    df = pd.DataFrame({"lineage_group": group, "fine_cluster": M.index.astype(str),
                       "amp2_label": best.str.replace("amp2::", "").values,
                       "z_best": top2[:, 1], "z_margin": top2[:, 1] - top2[:, 0],
                       "n_cells": a.obs[key].value_counts().reindex(M.index).values})
    df["amp2_match"] = (df.z_best > 1.0) & (df.z_margin > 0.1) & (df.n_cells >= 30)
    a.obs.drop(columns=cols, inplace=True)
    return df


def main():
    register_raw(AMP2_SUPP, "Zhang 2023 Nature Supp Table 5 (AMP2 cluster markers)", AMP2_URL)
    full = sc.read_h5ad(OBJECTS / "atlas" / "micheroli_lineage.h5ad")
    maps, parts, mks = [], [], []
    for group, (lins, sheets) in GROUPS.items():
        key = "fine_cluster"
        cache = OBJECTS / "atlas" / f"sub_{group}.h5ad"   # regenerable cache of the lineage scVI run
        if cache.exists():
            a = sc.read_h5ad(cache)
        else:
            a = full[full.obs.lineage.isin(lins)].copy()
            a = subcluster(a, key, res=1.2 if group in ("T_NK", "Stromal", "Myeloid") else 0.8)
            a.write_h5ad(cache)
        mp = name_clusters(a, key, amp2_markers(sheets), group)
        # unmatched subclusters keep a data-driven name from their top-2 markers (reviewed manually before figures)
        sc.tl.rank_genes_groups(a, key, method="wilcoxon", use_raw=False, n_genes=2)
        top2 = pd.DataFrame(a.uns["rank_genes_groups"]["names"]).T.apply(lambda r: "/".join(r), axis=1)
        mp["top_markers"] = mp.fine_cluster.map(top2)
        lab = mp.set_index("fine_cluster").apply(
            lambda r: r.amp2_label if r.amp2_match else f"{group}-c{r.name} {r.top_markers}", axis=1)
        a.obs["fine_state"] = pd.Categorical(a.obs[key].astype(str).map(lab).values)
        a.obs["fine_cluster"] = group + "-c" + a.obs[key].astype(str)
        a.obs["lineage_group"] = group
        a.obs["fine_state"] = a.obs["fine_state"].cat.remove_unused_categories()
        print(f"== {group}: {a.n_obs} cells\n{mp.round(2).to_string(index=False)}", flush=True)
        sc.tl.rank_genes_groups(a, "fine_state", method="wilcoxon", use_raw=False)
        mk = sc.get.rank_genes_groups_df(a, None)
        mk = mk[(mk.pvals_adj < 0.05) & (mk.logfoldchanges > 0.5)].groupby("group").head(30)
        mk.insert(0, "lineage_group", group)
        mks.append(mk)
        maps.append(mp)
        o = a.obs[["sample", "lineage", "lineage_group", "fine_cluster", "fine_state"]].copy()
        o[["sub_umap1", "sub_umap2"]] = a.obsm["X_umap"]
        parts.append(o)

    M = pd.concat(maps, ignore_index=True)
    O = pd.concat(parts)
    write_table(M, "atlas_fine_state_map", script=__file__, datasets=["E-MTAB-11791", "AMP2 Supp Table 5"],
                note="z of AMP2 marker-set score across subclusters; match if z_best>1 & margin>0.1 & n>=30")
    write_table(pd.concat(mks, ignore_index=True), "atlas_fine_markers", script=__file__, datasets=["E-MTAB-11791"])
    write_table(O.reset_index(names="cell"), "atlas_cells_fine", script=__file__, datasets=["E-MTAB-11791"])

    fine = full[O.index].copy()
    for c in ["lineage_group", "fine_cluster", "fine_state"]:
        fine.obs[c] = O.loc[fine.obs_names, c].astype("category")
    f = OBJECTS / "atlas" / "micheroli_fine.h5ad"
    fine.write_h5ad(f, compression="gzip")
    write_object(f, "atlas_micheroli_fine", script=__file__, datasets=["E-MTAB-11791"], n_rows=fine.n_obs,
                 note="lineage-group subclustering, AMP2-named fine states (mast/cycling excluded)")


if __name__ == "__main__":
    main()
