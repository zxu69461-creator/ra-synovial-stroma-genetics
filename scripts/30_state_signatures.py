"""30_state_signatures.py — state-specific gene signatures from the curated atlas, for scoring bulk biopsies (Fig 3).

For each curated state: one-vs-rest Wilcoxon WITHIN its compartment (T/NK, B/plasma, Myeloid, Stromal, Endothelial),
then keep genes that are
  - logFC > 0.5 (within compartment), adj p < 0.01, detected in >= 25% of the state's cells
  - specific at the compartment level: mean expression in the state's own compartment >= 2x the highest other
    compartment (stops lineage-wide genes such as PTPRC or COL1A1 dominating a fine-state signature)
and take the top 50 by score. States with < 10 surviving genes are flagged (scored but interpreted cautiously).

DuckDB: state_signatures (state, state_group, gene, rank, logfc, pct_in, score), state_signature_summary
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import OBJECTS, write_table  # noqa: E402

N_TOP, MIN_GENES = 50, 10


def main():
    a = sc.read_h5ad(OBJECTS / "atlas" / "micheroli_states.h5ad")
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.filter_genes(a, min_cells=50)

    # compartment-level mean expression (linear scale) for the specificity filter
    grp = a.obs.state_group.astype(str)
    comp_mean = pd.DataFrame({g: np.asarray(np.expm1(a[grp == g].X).mean(0)).ravel()
                              for g in grp.unique()}, index=a.var_names)

    # rank WITHIN each compartment so fine-state markers (e.g. FOXP3, CXCL13) outrank lineage-wide genes (CD3D, IL32)
    rows = []
    for own in grp.unique():
        sub = a[(grp == own).values].copy()
        sc.tl.rank_genes_groups(sub, "state", method="wilcoxon", pts=True, use_raw=False)
        df = sc.get.rank_genes_groups_df(sub, None)
        spec = comp_mean[own] / (comp_mean.drop(columns=own).max(axis=1) + 1e-3)
        for st, d in df.groupby("group"):
            d = d[(d.logfoldchanges > 0.5) & (d.pvals_adj < 0.01) & (d.pct_nz_group >= 0.25)]
            d = d[d.names.map(spec).fillna(0) >= 2].sort_values("scores", ascending=False).head(N_TOP)
            d = d.assign(state=st, state_group=own, rank=np.arange(1, len(d) + 1),
                         comp_specificity=d.names.map(spec).values)
            rows.append(d.rename(columns={"names": "gene", "logfoldchanges": "logfc", "pct_nz_group": "pct_in",
                                          "scores": "score"})[["state", "state_group", "gene", "rank", "logfc",
                                                               "pct_in", "score", "comp_specificity"]])
    state2grp = a.obs.groupby("state", observed=True).state_group.first().astype(str)
    sig = pd.concat(rows, ignore_index=True)
    summ = sig.groupby(["state_group", "state"]).agg(n_genes=("gene", "size"),
                                                     top_genes=("gene", lambda g: " ".join(g[:8]))).reset_index()
    summ["usable"] = summ.n_genes >= MIN_GENES
    all_states = state2grp.reset_index().rename(columns={0: "state_group"})
    summ = all_states.merge(summ, on=["state", "state_group"], how="left").fillna({"n_genes": 0, "usable": False})
    ds = ["E-MTAB-11791 (curated states)"]
    write_table(sig, "state_signatures", script=__file__, datasets=ds,
                note=f"within-compartment Wilcoxon; logFC>0.5, padj<0.01, pct>=25%, compartment specificity>=2x; top {N_TOP}")
    write_table(summ, "state_signature_summary", script=__file__, datasets=ds)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
