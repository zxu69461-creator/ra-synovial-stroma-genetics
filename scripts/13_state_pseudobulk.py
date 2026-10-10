"""13_state_pseudobulk.py — per-state expression summary of the curated atlas (used by Fig 5/7 gene-to-state mapping).

For each state: mean log1p(CP10k) and detection fraction per gene; plus a specificity z (gene-wise z-score of the
state means across the 54 states), so "which state expresses gene G most specifically" is one lookup.

Parquet views: state_pseudobulk_long (state, state_group, gene, mean_lognorm, pct_detected, spec_z)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import OBJECTS, write_parquet_view  # noqa: E402


def main():
    a = sc.read_h5ad(OBJECTS / "atlas" / "micheroli_states.h5ad")
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.filter_genes(a, min_cells=20)
    grp = a.obs.groupby("state", observed=True).state_group.first().astype(str)
    means, pcts = {}, {}
    for st in grp.index:
        X = a[(a.obs.state == st).values].X
        means[st] = np.asarray(X.mean(axis=0)).ravel()
        pcts[st] = np.asarray((X > 0).mean(axis=0)).ravel()
    M = pd.DataFrame(means, index=a.var_names)
    P = pd.DataFrame(pcts, index=a.var_names)
    Z = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1) + 1e-6, axis=0)
    long = (M.stack().rename("mean_lognorm").to_frame()
            .join(P.stack().rename("pct_detected")).join(Z.stack().rename("spec_z"))
            .rename_axis(["gene", "state"]).reset_index())
    long["state_group"] = long.state.map(grp)
    for c in ["mean_lognorm", "pct_detected", "spec_z"]:
        long[c] = long[c].astype("float32")
    write_parquet_view(long, "state_pseudobulk_long", script=__file__, datasets=["E-MTAB-11791 (curated states)"],
                       note="per state x gene: mean log1p CP10k, detection fraction, z across 54 state means")
    print(long.shape, long.state.nunique(), long.gene.nunique())


if __name__ == "__main__":
    main()
