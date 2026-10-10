"""29_spikein_power.py — power of the gene-level map to detect a minority stromal component of RA risk.

Spike-in: in the replication atlas (GSE243917, 11 RA donors, 76 states), the RA gene set of the primary scDRS analysis
(top 1,000 MAGMA genes, Ishigaki EUR, MHC removed, z-weighted) has a fraction k of its genes (chosen at random)
replaced by the top Dupuytren's-contracture MAGMA genes not already in the set (a fibroblast-acting signal, each with
its own z weight). k = 0, 5, 10, 20, 30, 50%; two random seeds per k > 0. For each spiked set, scDRS is run exactly as
in 21_scdrs.py (1,000 matched control sets, covariates log n_genes + sample) and the stromal readout is recorded:
stromal states at BH-FDR < 0.05 (enrichment direction) and the median stromal z. The minimum k that makes stromal
states detectable is the detection limit of the gene-level map for a stromal share of the risk genes.

DuckDB: spikein_power (k, seed, n_stromal_sig, n_stromal, median_stromal_z, n_TNK_sig, n_TNK, median_TNK_z)
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_table  # noqa: E402

spec = importlib.util.spec_from_file_location("s21", ROOT / "scripts/21_scdrs.py")
s21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s21)

os.environ.setdefault("TQDM_DISABLE", "1")
KS = [0.0, 0.10, 0.20, 0.30, 0.50]
SEEDS = [1]
PART = ROOT / "db/objects/spikein"          # one TSV per (k, seed) run, so runs can go in parallel processes
H5AD = ROOT / "db/objects/atlas/gupta2023_states.h5ad"


def main(ks):
    import scanpy as sc
    import scdrs
    from statsmodels.stats.multitest import multipletests
    adata = sc.read_h5ad(H5AD)
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if "connectivities" not in adata.obsp:      # as 21_scdrs.py: kNN graph needed for the heterogeneity statistic
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
    genes = set(adata.var_names)
    ra_g, ra_w = s21.build_gs(["ra_eur_noMHC"], genes)["ra_eur_noMHC"]
    dup = read("magma_genes_fg_dupuytren")[["SYMBOL", "ZSTAT"]].dropna().drop_duplicates("SYMBOL")
    dup = dup[dup.SYMBOL.isin(genes) & ~dup.SYMBOL.isin(ra_g)].sort_values("ZSTAT", ascending=False)
    grp = adata.obs.state_group.astype(str)
    rows = []
    for k in ks:
        for seed in ([0] if k == 0 else SEEDS):
            rng = np.random.default_rng(seed)
            n = int(round(k * len(ra_g)))
            drop = set(rng.choice(len(ra_g), n, replace=False)) if n else set()
            g = [x for i, x in enumerate(ra_g) if i not in drop] + dup.SYMBOL.head(n).tolist()
            w = [x for i, x in enumerate(ra_w) if i not in drop] + np.clip(dup.ZSTAT.head(n), 0, 10).tolist()
            df = scdrs.score_cell(adata, g, gene_weight=w, ctrl_match_key="mean_var", n_ctrl=1000, weight_opt="vs",
                                  return_ctrl_raw_score=False, return_ctrl_norm_score=True, verbose=False)
            G = scdrs.method.downstream_group_analysis(adata, df, group_cols=["state"])["state"].reset_index(names="state")
            G["fdr"] = multipletests(G.assoc_mcp, method="fdr_bh")[1]
            G["comp"] = G.state.map(adata.obs.drop_duplicates("state").set_index("state").state_group.astype(str))
            st, tn = G[G.comp == "Stromal"], G[G.comp == "T/NK"]
            sig = lambda x: int(((x.fdr < 0.05) & (x.assoc_mcz > 0)).sum())
            rows.append(dict(k=k, seed=seed, n_spiked=n, n_stromal_sig=sig(st), n_stromal=len(st),
                             median_stromal_z=float(st.assoc_mcz.median()), n_TNK_sig=sig(tn), n_TNK=len(tn),
                             median_TNK_z=float(tn.assoc_mcz.median())))
            print(rows[-1], flush=True)
            PART.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([rows[-1]]).to_csv(PART / f"k{k:.2f}_s{seed}.tsv", sep="\t", index=False)


def collect():
    R = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(PART.glob("*.tsv"))], ignore_index=True).sort_values("k")
    write_table(R, "spikein_power", script=__file__, datasets=["GSE243917", "Ishigaki 2022", "FinnGen R13 M13_DUPUTRYEN"],
                note="scDRS spike-in: fraction k of the RA top-1000 genes replaced by top Dupuytren genes; stromal "
                     "detection in the replication atlas")
    print(R.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["collect"]:
        collect()
    else:
        main([float(x) for x in sys.argv[1:]] or KS)
