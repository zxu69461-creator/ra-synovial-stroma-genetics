"""61_xenium_annotate.py — hierarchical annotation of Xenium cells with the curated atlas states (Fig 6).

v1 (single 54-class CellTypist SGD model) was rejected: 316k cells fell into a T/NK "sink" without T markers and
median confidence was 0. v2:
  A. Xenium-native compartments: cells >= 20 transcripts -> log-norm -> PCA(40) -> Harmony (patient) -> Leiden;
     each cluster gets a compartment from canonical marker scores (T/NK, B/plasma, Myeloid, Stromal, Endothelial,
     Mast); clusters with no clear winner (z margin < 0.2) or a weak best raw marker score (< 0.3) are 'Unresolved'.
  B. Within each compartment, a multinomial logistic regression (scikit-learn; CellTypist-style: log-norm, per-gene
     scaling, L2) trained only on that compartment's atlas states (<= 2,000 cells/state, panel genes) assigns the
     fine state; confidence = max probability. (CellTypist 1.7.1 is incompatible with scikit-learn >= 1.7.)
Output: obs (compartment, state, state_group, state_conf, xen_cluster) in db/objects/xenium/xenium_pretreatment.h5ad
Parquet view: xenium_cells; DuckDB: xenium_cluster_compartment, xenium_marker_check, xenium_state_counts
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import OBJECTS, write_object, write_parquet_view, write_table  # noqa: E402

XEN = OBJECTS / "xenium" / "xenium_pretreatment.h5ad"
MARKERS = {"T/NK": ["CD3E", "CD3D", "CD2", "TRAC", "NKG7", "GZMA"], "B/plasma": ["MS4A1", "CD79A", "MZB1", "JCHAIN"],
           "Myeloid": ["CD68", "CD14", "C1QA", "CD163", "LYZ"], "Stromal": ["COL1A1", "PDGFRA", "PRG4", "DCN", "THY1"],
           "Endothelial": ["PECAM1", "VWF", "CDH5", "PLVAP"], "Mast": ["TPSAB1", "CPA3", "KIT"]}


def compartments(q):
    import harmonypy
    sc.pp.pca(q, n_comps=40)
    ho = harmonypy.run_harmony(q.obsm["X_pca"], q.obs, "patient", max_iter_harmony=10)
    z = np.asarray(ho.Z_corr)
    q.obsm["X_harmony"] = (z.T if z.shape[0] == 40 else z).astype("float32")
    sc.pp.neighbors(q, use_rep="X_harmony", n_neighbors=15)
    sc.tl.leiden(q, resolution=1.0, key_added="xen_cluster", flavor="igraph", n_iterations=2)
    return recall_compartments(q)


def recall_compartments(q):
    cols = []
    for g, ms in MARKERS.items():
        sc.tl.score_genes(q, [m for m in ms if m in q.var_names], score_name=f"m::{g}")
        cols.append(f"m::{g}")
    S = q.obs.groupby("xen_cluster", observed=True)[cols].mean()
    Z = (S - S.mean()) / S.std(ddof=0)
    top = Z.idxmax(axis=1).str.replace("m::", "")
    srt = np.sort(Z.values, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    best_raw = S.max(axis=1).values
    comp = top.where((margin >= 0.2) & (best_raw >= 0.3), "Unresolved")
    tab = S.copy()
    tab.columns = [c.replace("m::", "") for c in tab.columns]
    tab["compartment"], tab["margin"] = comp, margin
    tab["n_cells"] = q.obs.xen_cluster.value_counts().reindex(tab.index).values
    q.obs["compartment"] = q.obs.xen_cluster.map(comp).astype(str)
    q.obs.drop(columns=cols, inplace=True)
    return tab.reset_index()


def main():
    atlas = sc.read_h5ad(OBJECTS / "atlas" / "micheroli_states.h5ad")
    x = sc.read_h5ad(XEN)
    genes = sorted(set(atlas.var_names) & set(x.var_names))
    for c in ["state", "state_group", "state_conf", "passed"]:
        if c in x.obs:
            del x.obs[c]

    q = x[(x.obs.transcript_counts >= 20).values, genes].copy()
    sc.pp.normalize_total(q, target_sum=1e4)
    sc.pp.log1p(q)
    cache = OBJECTS / "xenium" / "_cluster_cache.parquet"   # regenerable; removed at the end
    if cache.exists():
        cc = pd.read_parquet(cache).reindex(q.obs_names)
        q.obs["xen_cluster"] = cc.xen_cluster.astype(str).values
        q.obsm["X_harmony"] = cc.drop(columns="xen_cluster").to_numpy("float32")
        tab = recall_compartments(q)
    else:
        tab = compartments(q)
        cc = pd.DataFrame(q.obsm["X_harmony"], index=q.obs_names, columns=[f"h{i}" for i in range(40)])
        cc["xen_cluster"] = q.obs.xen_cluster.astype(str).values
        cc.to_parquet(cache)
    write_table(tab, "xenium_cluster_compartment", script=__file__, datasets=["Zenodo 15058262"],
                note="Leiden on Harmony(patient) PCA; compartment = max marker z across clusters, margin>=0.2")
    print(tab.round(2).to_string(index=False), flush=True)

    rng = np.random.default_rng(0)
    q.obs["state"], q.obs["state_conf"] = "NA", np.nan
    for comp in ["T/NK", "B/plasma", "Myeloid", "Stromal", "Endothelial"]:
        ref = atlas[(atlas.obs.state_group == comp).values]
        states = ref.obs.state.astype(str)
        idx = np.concatenate([rng.choice(np.where(states == s)[0], min(2000, (states == s).sum()), replace=False)
                              for s in states.unique()])
        tr = ref[idx, genes].copy()
        tr.X = tr.layers["counts"].copy()
        sc.pp.normalize_total(tr, target_sum=1e4)
        sc.pp.log1p(tr)
        tr.obs["state"] = tr.obs.state.astype(str)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Xtr = tr.X.toarray() if hasattr(tr.X, "toarray") else tr.X
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.1, max_iter=500, n_jobs=8).fit(scaler.transform(Xtr).clip(-10, 10),
                                                                    tr.obs.state.values)
        sel = np.where(q.obs.compartment.values == comp)[0]
        for i in range(0, len(sel), 40_000):          # dense scaled chunks; keeps peak RAM ~3 GB
            chunk = q[sel[i:i + 40_000]]
            Xc = chunk.X.toarray() if hasattr(chunk.X, "toarray") else chunk.X
            pr = clf.predict_proba(scaler.transform(Xc).clip(-10, 10))
            q.obs.loc[chunk.obs_names, "state"] = clf.classes_[pr.argmax(1)]
            q.obs.loc[chunk.obs_names, "state_conf"] = pr.max(1)
        print(f"{comp}: {len(sel)} cells, median conf "
              f"{q.obs.loc[q.obs.compartment == comp, 'state_conf'].median():.2f}", flush=True)
    q.obs.loc[q.obs.compartment == "Mast", "state"] = "Mast"

    rows = []
    for g, ms in MARKERS.items():
        ms = [m for m in ms if m in q.var_names]
        e = pd.Series(np.asarray(q[:, ms].X.mean(axis=1)).ravel(), index=q.obs_names)
        rows.append(e.groupby(q.obs.compartment).mean().rename(g))
    chk = pd.DataFrame(rows).T.reset_index(names="compartment")
    write_table(chk, "xenium_marker_check", script=__file__, datasets=["Zenodo 15058262"],
                note="mean log-normalised marker-set expression per assigned compartment")
    print(chk.round(2).to_string(index=False), flush=True)

    for c in ["compartment", "state", "state_conf", "xen_cluster"]:
        x.obs[c] = q.obs[c].reindex(x.obs_names).values
    x.obs["compartment"] = x.obs.compartment.fillna("low_counts")
    x.obs["state"] = x.obs.state.fillna("low_counts")
    x.obs["xen_cluster"] = x.obs.xen_cluster.astype(str)
    x.obs["state_group"] = x.obs.compartment
    for c in ["compartment", "state", "state_group", "xen_cluster"]:
        x.obs[c] = x.obs[c].astype(str).astype("category")
    H = np.zeros((x.n_obs, q.obsm["X_harmony"].shape[1]), dtype="float32")
    H[x.obs_names.get_indexer(q.obs_names)] = q.obsm["X_harmony"]
    x.obsm["X_harmony"] = H
    x.write_h5ad(XEN, compression="gzip")
    write_object(XEN, "xenium_pretreatment", script=__file__, datasets=["Zenodo 15058262", "E-MTAB-11791"],
                 n_rows=x.n_obs, note=f"hierarchical annotation v2 on {len(genes)} shared genes")
    cells = x.obs[["patient", "x_centroid", "y_centroid", "transcript_counts", "xen_cluster", "compartment",
                   "state", "state_conf"]].reset_index(names="cell")
    for c in ["patient", "xen_cluster", "compartment", "state"]:
        cells[c] = cells[c].astype(str)
    write_parquet_view(cells, "xenium_cells", script=__file__, datasets=["Zenodo 15058262"],
                       note="per-cell coordinates + Xenium compartment + transferred atlas state (v2)")
    cnt = cells[~cells.state.isin(["low_counts", "NA"])].groupby(["patient", "compartment", "state"]).size()
    write_table(cnt.rename("n").reset_index(), "xenium_state_counts", script=__file__, datasets=["Zenodo 15058262"])
    print(cells.compartment.value_counts().to_string(), flush=True)
    cache.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
