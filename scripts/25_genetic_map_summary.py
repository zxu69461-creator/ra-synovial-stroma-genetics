"""25_genetic_map_summary.py — one replication matrix for the genetic map (Fig 3): every method x atlas x GWAS.

Rows (analysis), per compartment: n states, n significant (FDR < 0.05, enrichment direction), and a signed summary
  scDRS            median assoc_mcz of the compartment's states      (scdrs_state_load; scdrs_group_gupta_state;
                                                                       scdrs_group_eas_state, scdrs_group_gupta_eas_state)
  MAGMA cell-type  median signed -log10 P (one-sided enrichment)       (magma_celltype)
  causal genes     compartment enrichment z vs matched random sets     (causal_gene_state_enrichment)
Traits: RA (Ishigaki EUR, MHC removed) primary; FinnGen RA (MHC removed); Ishigaki EAS (MHC removed);
contrast traits OA and height; fibroblast-disease positive control Dupuytren's contracture (scDRS in the
tagged tables scdrs_group_dup_state / scdrs_group_gupta_dup_state). Missing analyses are skipped (printed).
DuckDB: genetic_map_matrix, genetic_map_concordance (state-level Spearman between GWAS / methods per atlas)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import connect, read, write_table  # noqa: E402

COMP = ["T/NK", "B/plasma", "Myeloid", "Endothelial", "Stromal"]
TRAIT = {"ra_eur_noMHC": "RA EUR", "fg_ra_noMHC": "RA FinnGen", "ra_eas_noMHC": "RA EAS", "fg_oa": "OA",
         "fg_height": "Height", "fg_dupuytren": "Dupuytren"}
ATLAS = {"micheroli": "Discovery (E-MTAB-11791)", "gupta": "Replication (GSE243917, RA only)"}


def tables():
    con = connect(read_only=True)
    try:
        return set(con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()


def main():
    have = tables()
    rows = []
    grp_disc = read("SELECT DISTINCT state, state_group FROM atlas_cells_state")
    grp_disc = grp_disc.assign(state_group=grp_disc.state_group.astype(str)).set_index("state").state_group
    grp_rep = read("gupta_state_summary").set_index("state").state_group if "gupta_state_summary" in have else None
    scdrs_src = [("micheroli", "scdrs_group_state", grp_disc), ("micheroli", "scdrs_group_eas_state", grp_disc),
                 ("micheroli", "scdrs_group_dup_state", grp_disc),
                 ("gupta", "scdrs_group_gupta_state", grp_rep), ("gupta", "scdrs_group_gupta_eas_state", grp_rep),
                 ("gupta", "scdrs_group_gupta_dup_state", grp_rep)]
    for atlas, tab, grp in scdrs_src:
        if tab not in have or grp is None:
            print("skip", tab)
            continue
        g = read(tab)
        for t, d in g.groupby("trait"):
            if t not in TRAIT:
                continue
            d = d.assign(fdr=multipletests(d.assoc_mcp, method="fdr_bh")[1], state_group=d.state.map(grp))
            for c in COMP:
                x = d[d.state_group == c]
                rows.append(dict(method="scDRS", atlas=atlas, trait=t, compartment=c, n_states=len(x),
                                 n_sig=int(((x.fdr < 0.05) & (x.assoc_mcz > 0)).sum()), summary=x.assoc_mcz.median(),
                                 summary_type="median z"))
    if "magma_celltype" in have:
        m = read("magma_celltype")
        for (atlas, t), d in m.groupby(["atlas", "trait"]):
            if t not in TRAIT:
                continue
            for c in COMP:
                x = d[d.state_group == c]
                rows.append(dict(method="MAGMA cell-type", atlas=atlas, trait=t, compartment=c, n_states=len(x),
                                 n_sig=int((x.fdr < 0.05).sum()), summary=np.median(-np.log10(x.p)),
                                 summary_type="median -log10 P"))
    if "causal_gene_state_enrichment" in have:
        e = read("causal_gene_state_enrichment")
        for (atlas, gs), d in e.groupby(["atlas", "gene_set"]):
            st = d[~d.state.str.startswith("[")]
            for c in COMP:
                comp = d[d.state == f"[compartment] {c}"]
                x = st[st.state_group == c]
                rows.append(dict(method=f"Causal genes ({gs})", atlas=atlas, trait="ra_causal", compartment=c,
                                 n_states=len(x), n_sig=int((x.fdr < 0.05).sum()),
                                 summary=float(comp.enrich_z.iloc[0]) if len(comp) else np.nan,
                                 summary_type="compartment enrichment z", p_compartment=float(comp.p.iloc[0])
                                 if len(comp) else np.nan))
    R = pd.DataFrame(rows)
    R["frac_sig"] = R.n_sig / R.n_states.replace(0, np.nan)
    R["atlas_label"] = R.atlas.map(ATLAS)
    R["trait_label"] = R.trait.map(TRAIT).fillna("RA causal genes")
    write_table(R, "genetic_map_matrix", script=__file__, datasets=["E-MTAB-11791", "GSE243917", "Ishigaki 2022",
                                                                    "FinnGen R13"],
                note="per method x atlas x trait x compartment: states significant (FDR<0.05) and signed summary")
    # concordance of state-level scores between GWAS (and between methods) within each atlas
    from scipy import stats
    conc = []
    zsrc = {"micheroli": [t for t in ["scdrs_group_state", "scdrs_group_eas_state", "scdrs_group_dup_state"] if t in have],
            "gupta": [t for t in ["scdrs_group_gupta_state", "scdrs_group_gupta_eas_state", "scdrs_group_gupta_dup_state"]
                      if t in have]}
    for atlas, tabs in zsrc.items():
        if not tabs:
            continue
        z = pd.concat([read(t) for t in tabs]).pivot_table(index="state", columns="trait", values="assoc_mcz")
        for a_, b_ in [("ra_eur_noMHC", "ra_eas_noMHC"), ("ra_eur_noMHC", "fg_ra_noMHC"), ("ra_eur_noMHC", "fg_height"),
                       ("ra_eur_noMHC", "fg_oa"), ("ra_eur_noMHC", "fg_dupuytren")]:
            if {a_, b_} <= set(z.columns):
                r, p = stats.spearmanr(z[a_], z[b_], nan_policy="omit")
                conc.append(dict(atlas=atlas, method="scDRS z", trait_a=a_, trait_b=b_, n_states=int(z[[a_, b_]].dropna().shape[0]),
                                 rho=r, p=p))
        if "magma_celltype" in have:
            m = read("magma_celltype").query("atlas == @atlas").pivot_table(index="state", columns="trait", values="p")
            sz = z["ra_eur_noMHC"].reindex(m.index)
            r, p = stats.spearmanr(sz, -np.log10(m["ra_eur_noMHC"]), nan_policy="omit")
            conc.append(dict(atlas=atlas, method="scDRS z vs MAGMA -log10P", trait_a="ra_eur_noMHC", trait_b="ra_eur_noMHC",
                             n_states=int(sz.notna().sum()), rho=r, p=p))
    C = pd.DataFrame(conc)
    write_table(C, "genetic_map_concordance", script=__file__, datasets=["E-MTAB-11791", "GSE243917"],
                note="state-level Spearman between GWAS and between methods, per atlas")
    print(C.round(4).to_string(index=False))
    pd.set_option("display.width", 220)
    R["cell"] = R.n_sig.astype(str) + "/" + R.n_states.astype(str)
    print(R.pivot_table(index=["method", "atlas", "trait"], columns="compartment", values="cell", aggfunc="first")
          [COMP].to_string())


if __name__ == "__main__":
    main()
