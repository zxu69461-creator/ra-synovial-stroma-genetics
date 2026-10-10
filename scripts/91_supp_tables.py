"""91_supp_tables.py — Supplementary Tables S1–S17 (one xlsx workbook) exported from archived DuckDB tables.

Output: manuscript/Supplementary_Tables.xlsx (sheet "README" describes each table + its source table/script).
Registered in DuckDB as object supplementary_tables_xlsx (sha256 + provenance) via archive.write_object.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, sha256, write_table  # noqa: E402

OUT = ROOT / "results" / "Supplementary_Tables.xlsx"


def cohorts():
    M = read("bulk_samples")
    b = M[M.timepoint == "baseline"]
    rows = []
    info = {"R4RA": ("E-MTAB-11611", "rituximab or tocilizumab (randomised; arm not public)", "CDAI response (week 16)",
                     "Salmon, first 20 M read pairs"),
            "STRAP": ("E-MTAB-13733", "etanercept / tocilizumab / rituximab (randomised)",
                      "ACR20 (week 16), linked for 135/210", "Salmon, first 20 M read pairs"),
            "PEAC": ("E-MTAB-6141", "treatment-naive early RA", "none (pathotype, DAS28, CRP)",
                     "Salmon, first 20 M read pairs"),
            "GSE198520": ("GSE198520", "anti-TNF", "EULAR (week 12)", "deposited counts (~1 M reads)"),
            "GSE24742": ("GSE24742", "rituximab", "EULAR", "GPL570 array, RMA"),
            "GSE45867": ("GSE45867", "tocilizumab / methotrexate", "EULAR from DAS28-CRP", "GPL570 array, RMA")}
    for coh, g in b.groupby("cohort"):
        r = g.response_binary.dropna()
        acc, drug, resp, quant = info[coh]
        rows.append(dict(cohort=coh, accession=acc, treatment=drug, outcome=resp, quantification=quant,
                         n_baseline=len(g), n_with_binary_response=len(r), n_non_responders=int((r == 0).sum()),
                         n_post_treatment=int((M[(M.cohort == coh)].timepoint == "post").sum()),
                         pathotype_available=bool(g.pathotype.notna().any())))
    return pd.DataFrame(rows)


def states():
    cur = read("atlas_state_curation")
    n = read("SELECT state, state_group, count(*) n_cells FROM atlas_cells_state GROUP BY 1, 2")
    n["state_group"] = n.state_group.astype(str)
    sig = read("state_signature_summary")[["state", "n_genes", "top_genes", "usable"]]
    why = cur[cur.state.notna()].groupby("state").reason.apply(lambda x: " | ".join(sorted(set(x))))
    d = n.merge(sig, on="state", how="left").merge(why.rename("curation_reason"), left_on="state", right_index=True,
                                                    how="left")
    return d.rename(columns={"state_group": "compartment", "n_genes": "signature_genes",
                             "top_genes": "top_signature_genes", "usable": "scored_in_bulk"})


def have_tables():
    from archive import connect
    con = connect(read_only=True)
    try:
        return set(con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()


def scdrs_all():
    T = have_tables()
    out = []
    grp_d = read("SELECT DISTINCT state, state_group FROM atlas_cells_state")
    grp_d = grp_d.assign(state_group=grp_d.state_group.astype(str)).set_index("state").state_group
    grp_r = read("gupta_state_summary").set_index("state").state_group
    for tab, atlas, grp in [("scdrs_group_state", "discovery", grp_d), ("scdrs_group_eas_state", "discovery", grp_d),
                            ("scdrs_group_dup_state", "discovery", grp_d),
                            ("scdrs_group_gupta_state", "replication", grp_r),
                            ("scdrs_group_gupta_eas_state", "replication", grp_r),
                            ("scdrs_group_gupta_dup_state", "replication", grp_r)]:
        if tab in T:
            g = read(tab)
            from statsmodels.stats.multitest import multipletests
            g["fdr"] = g.groupby("trait").assoc_mcp.transform(lambda p: multipletests(p, method="fdr_bh")[1])
            out.append(g.assign(atlas=atlas, state_group=g.state.map(grp))[
                ["atlas", "trait", "state", "state_group", "n_cell", "assoc_mcz", "assoc_mcp", "fdr", "hetero_mcz",
                 "hetero_mcp"]])
    return pd.concat(out, ignore_index=True)


def main():
    T = have_tables()
    resp = pd.concat([read("response_assoc_state").assign(analysis="per-state response"),
                      read("gli_response").assign(analysis="GLI vs non-response"),
                      read("strap_drug_specific").assign(analysis="STRAP drug-specific"),
                      read("activity_longitudinal").assign(analysis="within-patient change vs change in activity"),
                      read("remodel_state").assign(analysis="treatment-induced remodelling")], ignore_index=True)
    sheets = {
        "S1_cohorts": (cohorts(), "Bulk synovial cohorts, treatments, outcomes and sample counts", "bulk_samples"),
        "S2_atlas_states": (states(), "Discovery atlas states: cells, compartment, bulk signature, curation reason",
                            "atlas_state_curation, atlas_cells_state, state_signature_summary"),
        "S3_replication_states": (read("gupta_state_summary"), "Replication atlas (GSE243917, 11 RA donors) states",
                                  "gupta_state_summary"),
        "S4_signatures": (read("state_signatures"), "State signature genes (within-compartment Wilcoxon)",
                          "state_signatures"),
        "S5_scDRS": (scdrs_all(), "scDRS genetic load per state, trait and atlas (BH-FDR within trait)",
                     "scdrs_group_state, scdrs_group_eas_state, scdrs_group_gupta_state, scdrs_group_gupta_eas_state"),
        "S6_MAGMA_celltype": (read("magma_celltype"), "MAGMA gene-property analysis per state, trait and atlas",
                              "magma_celltype"),
        "S7_causal_genes": (read("ra_causal_genes"), "Likely causal RA genes (Ishigaki 2022 evidence)",
                            "ra_causal_genes"),
        "S8_causal_enrichment": (read("causal_gene_state_enrichment"),
                                 "Enrichment of likely causal genes per state / compartment", "causal_gene_state_enrichment"),
        "S9_map_matrix": (read("genetic_map_matrix"), "Replication matrix: method x atlas x trait x compartment",
                          "genetic_map_matrix"),
        "S10_pathotype": (read("pathotype_assoc_state"), "State scores by histological pathotype (Kruskal-Wallis)",
                          "pathotype_assoc_state"),
        "S11_activity": (pd.concat([read("activity_assoc").assign(level="cohort"),
                                    read("activity_assoc_pooled").assign(level="pooled"),
                                    read("activity_state").assign(level="state")], ignore_index=True),
                         "Compartment / state scores vs baseline disease activity (Spearman; Fisher-z pooled)",
                         "activity_assoc, activity_assoc_pooled, activity_state"),
        "S12_activity_clinical": (read("clin_activity_harmonised"),
                                  "Harmonised disease activity per biopsy (DAS28-ESR computed for R4RA/STRAP)",
                                  "clin_activity_harmonised"),
        "S13_targets": (read("ot_target_state_summary"), "Open Targets RA target evidence per state",
                        "ot_target_state_summary"),
        "S14_response": (resp, "Supplementary treatment-response analyses (per-state, GLI, drug-specific, "
                         "within-patient change, remodelling)",
                         "response_assoc_state, gli_response, strap_drug_specific, activity_longitudinal, remodel_state"),
        "S15_STRAP_linkage": (read("strap_sample_match"), "STRAP biopsy-to-patient linkage", "strap_sample_match"),
        "S16_SLDSC_annotations": (read("sldsc_annotations"), "S-LDSC chromatin annotations (in vivo compartment-specific "
                                  "snATAC peaks; Ge 2021 cultured FLS peaks/regions, 3 RA, 3 OA and 1 healthy donor; open-chromatin control)",
                                  "sldsc_annotations"),
        "S17_SLDSC_results": (read("sldsc_chromatin"), "S-LDSC results per annotation and trait (baselineLD v2.2 + "
                              "open-chromatin control; one-sided coefficient P, BH-FDR within trait)", "sldsc_chromatin"),
        "S18_approved_targets": (read("approved_target_compartment"), "Targets of biologic / targeted-synthetic DMARDs "
                                 "approved for RA (ChEMBL max phase 4) and their most specific synovial state",
                                 "approved_target_compartment"),
        "S19_target_tests": (read("target_tests"), "Genetic (Open Targets >= 0.1/0.3/0.5) and clinical target shares of "
                             "immune- vs stromal-specific genes under three assignments; approved targets vs base rate",
                             "target_tests"),
        "S20_stromal_candidates": (read("stromal_genetic_candidates"), "Stroma-specific genes with Open Targets genetic "
                                   "evidence >= 0.1 (candidate genetically anchored stromal targets)",
                                   "stromal_genetic_candidates"),
    }
    T = have_tables()
    extra = {"S21_SLDSC_sensitivity": ("sldsc_chromatin_r2b", "S-LDSC sensitivity: FLS active regions per line, Roadmap T/B "
                                  "active regions, in vivo accessible chromatin and fibroblast sub-compartments "
                                  "(AllPeaks control); primary annotations with baseline only"),
             "S22_finemap": ("finemap_overlap", "PIP-weighted share of fine-mapped RA variants (Ishigaki 2022, PIP > 0.1) "
                             "in each chromatin annotation"),
             "S23_spikein": ("spikein_power", "scDRS spike-in power: share of RA genes replaced by Dupuytren genes and "
                             "stromal detection in the replication atlas"),
             "S24_activity_partial": ("activity_partial", "Partial Spearman of compartment scores vs activity "
                                      "(stromal adjusted for total immune score; T/NK adjusted for stromal score)")}
    for k, (tab, desc) in extra.items():
        if tab in T:
            df = read(tab)
            if tab == "sldsc_chromatin_r2b" and "sldsc_annotations_r2b" in T:
                df = df.merge(read("sldsc_annotations_r2b"), on="annotation", how="left")
            sheets[k] = (df, desc, tab)
    # number and order the sheets as the manuscript cites them (config/supp_table_order.json, written by
    # 92_build_manuscript.py); sheets the text does not cite are appended and reported
    import json
    order = json.loads((ROOT / "config" / "supp_table_order.json").read_text())
    bykey = {k.split("_", 1)[1]: k for k in sheets}
    missing = [k for k in order if k not in bykey]
    if missing:
        sys.exit(f"cited tables without a sheet: {missing}")
    uncited = [k for k in bykey if k not in order]
    keys = list(order) + uncited
    sheets = {f"S{i}_{k}"[:31]: sheets[bykey[k]] for i, k in enumerate(keys, 1)}
    if uncited:
        print("NOT cited in the manuscript (appended):", uncited)
    readme = pd.DataFrame([dict(sheet=k, description=v[1], rows=len(v[0]), source_duckdb_table=v[2])
                           for k, v in sheets.items()])
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        for k, (df, _, _) in sheets.items():
            df = df.copy()
            for c in df.columns:
                if isinstance(df[c].dtype, pd.CategoricalDtype):
                    df[c] = df[c].astype(str)
            df.round(6).to_excel(xw, sheet_name=k, index=False)
    write_table(readme.assign(file=str(OUT.relative_to(ROOT)), sha256=sha256(OUT)), "supplementary_tables",
                script=__file__, datasets=["archived result tables"], note="index of Supplementary_Tables.xlsx")
    print(readme.to_string(index=False))

if __name__ == "__main__":
    main()
