"""21_scdrs.py — cell-level GWAS enrichment (scDRS) across atlas states (Fig 2: "genetic load" per state).

Gene sets: top-1000 MAGMA genes per trait, z-score weighted (scDRS munge-gs defaults), built from our own
magma_genes_<tag> tables (20_magma.py). Traits: RA (Ishigaki EUR, seropositive, noMHC), FinnGen RA / seropos /
seroneg, OA (contrast), height (negative control).
Covariates: log n_genes + sample (as in scDRS paper).  1,000 control gene sets.
Group test: scDRS downstream group-analysis on `state_col` (default: fine_state if present else lineage).

Usage: python scripts/21_scdrs.py [--h5ad db/objects/atlas/<file>.h5ad] [--state-col fine_state] [--traits ra_eur ...]
       replication atlas: --tag gupta --datasets GSE243917  (tables scdrs_group_gupta_<col>, scdrs_cell_scores_gupta)
DuckDB: scdrs_group_<state_col> (trait x state: assoc_mcp, assoc_mcz, hetero_mcp, n_fdr_0.1)
Parquet view: scdrs_cell_scores (cell x trait normalised scores)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import OBJECTS, read, write_parquet_view, write_table  # noqa: E402

os.environ.setdefault("TQDM_DISABLE", "1")
if not hasattr(np, "float_"):  # scDRS 1.0.2 gearys_c still uses np.float_ (removed in NumPy 2)
    np.float_ = np.float64
CACHE = OBJECTS / "scdrs_cache"  # per-trait cell scores, so a downstream failure does not redo scoring

TRAITS = ["ra_eur", "ra_eur_noMHC", "ra_eur_seropos", "fg_ra", "fg_ra_noMHC", "fg_ra_seropos", "fg_ra_seroneg",
          "fg_oa", "fg_height"]


def build_gs(tags, genes_in_data, n_top=1000):
    """Top-n MAGMA genes by z-score, weights = z clipped at 10 (scDRS munge-gs 'zscore' convention)."""
    gs = {}
    for t in tags:
        m = read(f"magma_genes_{t}")[["SYMBOL", "ZSTAT"]].dropna().drop_duplicates("SYMBOL")
        m = m[m.SYMBOL.isin(genes_in_data)].sort_values("ZSTAT", ascending=False).head(n_top)
        gs[t] = (m.SYMBOL.tolist(), np.clip(m.ZSTAT.values, 0, 10).tolist())
    return gs


def main():
    import scdrs
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(OBJECTS / "atlas" / "micheroli_fine.h5ad"))
    ap.add_argument("--state-col", default=None)
    ap.add_argument("--traits", nargs="*", default=TRAITS)
    ap.add_argument("--n-top", type=int, default=1000, help="MAGMA genes per trait (sensitivity: 500 / 2000)")
    ap.add_argument("--tag", default="", help="atlas tag for a replication atlas (tables scdrs_group_<tag>_<col>)")
    ap.add_argument("--datasets", nargs="*", default=["E-MTAB-11791"])
    a = ap.parse_args()

    adata = sc.read_h5ad(a.h5ad)
    col = a.state_col or ("fine_state" if "fine_state" in adata.obs else "lineage")
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if "connectivities" not in adata.obsp:      # replication atlases ship without a kNN graph (needed for heterogeneity)
        tmp = adata.copy()
        sc.pp.highly_variable_genes(tmp, n_top_genes=3000)
        sc.pp.pca(tmp, n_comps=30, mask_var="highly_variable")
        sc.pp.neighbors(tmp, n_neighbors=15)
        adata.obsp["connectivities"], adata.obsp["distances"] = tmp.obsp["connectivities"], tmp.obsp["distances"]
        adata.uns["neighbors"] = tmp.uns["neighbors"]
        del tmp
    cov = pd.DataFrame(index=adata.obs_names)
    cov["const"] = 1.0
    cov["log_n_genes"] = np.log(adata.obs.n_genes_by_counts.astype(float))
    cov = cov.join(pd.get_dummies(adata.obs["sample"], prefix="s", drop_first=True, dtype=float))
    scdrs.preprocess(adata, cov=cov, n_mean_bin=20, n_var_bin=20, copy=False)

    gs = build_gs(a.traits, set(adata.var_names), n_top=a.n_top)
    sfx = "" if a.n_top == 1000 else f"_top{a.n_top}"
    scores, groups = {}, []
    for t, (genes, w) in gs.items():
        cf = CACHE / f"{Path(a.h5ad).stem}__{t}{sfx}.parquet"
        if cf.exists():
            df = pd.read_parquet(cf)
        else:
            df = scdrs.score_cell(adata, genes, gene_weight=w, ctrl_match_key="mean_var", n_ctrl=1000,
                                  weight_opt="vs", return_ctrl_raw_score=False, return_ctrl_norm_score=True,
                                  verbose=False)
            CACHE.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cf)
        scores[t] = df.norm_score
        g = scdrs.method.downstream_group_analysis(adata, df, group_cols=[col])[col]
        g = g.reset_index(names="state")
        g.insert(0, "trait", t)
        groups.append(g)
        top = g.sort_values("assoc_mcz", ascending=False).head(5)[["state", "assoc_mcz", "assoc_mcp"]]
        print(f"[{t}] top states:\n{top.to_string(index=False)}", flush=True)

    G = pd.concat(groups, ignore_index=True)
    G.columns = [c.replace(".", "_") for c in G.columns]
    pre = f"{a.tag}_" if a.tag else ""
    write_table(G, f"scdrs_group_{pre}{col}{sfx}", script=__file__, datasets=a.datasets + ["Ishigaki 2022", "FinnGen R13"],
                note=f"scDRS v1 group analysis; top-{a.n_top} MAGMA z genes; cov = log n_genes + sample; 1000 ctrl sets")
    if not sfx:   # per-cell scores archived for the primary gene-set size only
        S = pd.DataFrame(scores).astype("float32").reset_index(names="cell")
        write_parquet_view(S, f"scdrs_cell_scores{'_' + a.tag if a.tag else ''}", script=__file__, datasets=a.datasets,
                           note="scDRS normalised disease scores per cell per trait")
    # cache holds 1,000 control scores per cell per trait (~0.5 GB/trait); only needed until results are archived
    for t in a.traits:
        (CACHE / f"{Path(a.h5ad).stem}__{t}{sfx}.parquet").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
