"""23_magma_celltype.py — orthogonal genetic method: MAGMA gene-property ("cell-type") analysis of synovial states.

For each state: mean CP10k per gene across the state's cells -> log2(x + 1) (FUMA cell-typing convention); the
covariate file holds every state plus the across-state average. MAGMA tests each state's expression against gene-level
GWAS association, conditional on the average (model condition-hide=Average direction=greater; one-sided).
Unlike scDRS (top-1,000 genes, cell-level scores) this uses all ~18k genes and a linear regression framework.

Usage: python scripts/23_magma_celltype.py --atlas micheroli [--traits ra_eur_noMHC fg_ra_noMHC ...]
       python scripts/23_magma_celltype.py --atlas gupta
DuckDB: magma_celltype (atlas, trait, state, state_group, ngenes, beta, beta_std, se, p, fdr)   [rows of the atlas
        being run are replaced; other atlases kept]
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import connect, read, write_table  # noqa: E402

MAGMA = ROOT / "tools/magma/magma"
GENELOC = ROOT / "tools/magma/NCBI38.gene.loc"
RAW = ROOT / "db/objects/magma"
ATLAS = {"micheroli": dict(path=ROOT / "db/objects/atlas/micheroli_states.h5ad", state="state", group="state_group",
                           layer="counts"),
         "gupta": dict(path=ROOT / "db/objects/atlas/gupta2023_states.h5ad", state="state", group="state_group",
                       layer="counts")}
DEFAULT_TRAITS = ["ra_eur_noMHC", "ra_eur", "fg_ra_noMHC", "ra_eas_noMHC", "fg_oa", "fg_height"]


def state_means(cfg):
    a = ad.read_h5ad(cfg["path"])
    X = a.layers[cfg["layer"]] if cfg["layer"] in a.layers else a.X
    X = sp.csr_matrix(X, dtype=np.float32)
    lib = np.asarray(X.sum(axis=1)).ravel()
    X = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    st = a.obs[cfg["state"]].astype(str).to_numpy()
    out = {}
    for s in pd.unique(st):
        out[s] = np.log2(np.asarray(X[st == s].mean(axis=0)).ravel() + 1)
    E = pd.DataFrame(out, index=a.var_names)
    grp = a.obs[[cfg["state"], cfg["group"]]].astype(str).drop_duplicates().set_index(cfg["state"])[cfg["group"]]
    return E, grp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="micheroli", choices=list(ATLAS))
    ap.add_argument("--traits", nargs="*", default=DEFAULT_TRAITS)
    a = ap.parse_args()
    E, grp = state_means(ATLAS[a.atlas])
    loc = pd.read_csv(GENELOC, sep="\t", header=None, usecols=[0, 5], names=["ENTREZ", "SYMBOL"])
    loc = loc.drop_duplicates("SYMBOL").set_index("SYMBOL").ENTREZ
    E = E[E.index.isin(loc.index)]
    E.index = loc.reindex(E.index).values
    E = E[~E.index.duplicated()]
    names = {f"S{i:02d}": s for i, s in enumerate(E.columns)}
    E.columns = list(names)
    E["Average"] = E.mean(axis=1)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        cov = Path(td) / "covar.txt"
        E.rename_axis("GENE").reset_index().to_csv(cov, sep="\t", index=False)
        for t in a.traits:
            raw = RAW / f"{t}.genes.raw"
            if not raw.exists():
                print(f"skip {t}: no {raw.name}")
                continue
            out = Path(td) / t
            subprocess.run([str(MAGMA), "--gene-results", str(raw), "--gene-covar", str(cov),
                            "--model", "condition-hide=Average", "direction=greater", "--out", str(out)],
                           check=True, stdout=subprocess.DEVNULL)
            g = pd.read_csv(f"{out}.gsa.out", sep=r"\s+", comment="#")
            g = g[g.VARIABLE.isin(names)]
            g["state"] = g.VARIABLE.map(names)
            g["fdr"] = multipletests(g.P, method="fdr_bh")[1]
            for r in g.itertuples():
                rows.append(dict(atlas=a.atlas, trait=t, state=r.state, state_group=grp.get(r.state), ngenes=r.NGENES,
                                 beta=r.BETA, beta_std=r.BETA_STD, se=r.SE, p=r.P, fdr=r.fdr))
    R = pd.DataFrame(rows)
    con = connect(read_only=True)
    try:
        have = "magma_celltype" in set(con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()
    if have:
        old = read("magma_celltype")
        R = pd.concat([old[old.atlas != a.atlas], R], ignore_index=True)
    write_table(R, "magma_celltype", script=__file__, datasets=[a.atlas, "MAGMA genes.raw"],
                note="MAGMA gene-property, condition-hide=Average, one-sided; FDR within atlas x trait")
    S = R[R.atlas == a.atlas]
    pd.set_option("display.width", 200)
    print(S.pivot_table(index="state_group", columns="trait", values="fdr",
                        aggfunc=lambda f: f"{(f < 0.05).sum()}/{len(f)}"))
    print(S.pivot_table(index="state_group", columns="trait", values="p", aggfunc="median").round(3))

if __name__ == "__main__":
    main()
