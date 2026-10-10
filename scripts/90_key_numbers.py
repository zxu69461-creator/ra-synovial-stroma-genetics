"""90_key_numbers.py — every number quoted in the manuscript, computed from archived tables (no hand-typed values).

DuckDB: key_numbers (section, key, value, text, source)
Usage: python scripts/90_key_numbers.py   (prints the table grouped by section)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_table  # noqa: E402

COMP = ["T/NK", "B/plasma", "Myeloid", "Endothelial", "Stromal"]
IMMUNE = {"T/NK", "B/plasma", "Myeloid"}
TRAITS = {"ra_eur": "RA Ishigaki", "ra_eur_noMHC": "RA Ishigaki noMHC", "ra_eur_seropos": "RA Ishigaki seropositive",
          "fg_ra": "RA FinnGen", "fg_ra_noMHC": "RA FinnGen noMHC", "fg_ra_seropos": "RA FinnGen seropositive",
          "fg_ra_seroneg": "RA FinnGen seronegative", "fg_oa": "OA FinnGen", "fg_height": "Height FinnGen",
          "fg_dupuytren": "Dupuytren FinnGen"}
ROWS = []


def add(section, key, value, text="", source=""):
    if isinstance(value, (float, np.floating)):
        value = f"{float(value):.2f}"
    ROWS.append(dict(section=section, key=key, value=str(value), text=text, source=source))


def fp(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def dl(b, se):
    w = 1 / se ** 2
    fe = np.sum(w * b) / np.sum(w)
    q = np.sum(w * (b - fe) ** 2)
    df = len(b) - 1
    tau2 = max(0.0, (q - df) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if df > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    beta, s = np.sum(wr * b) / np.sum(wr), np.sqrt(1 / np.sum(wr))
    return beta, s, 2 * stats.norm.sf(abs(beta / s)), (max(0.0, (q - df) / q) if q > 0 and df > 0 else 0.0)


def atlas():
    qc = read("atlas_qc_by_sample")
    add("atlas", "n_samples", len(qc), "synovial samples in Micheroli 2024 (E-MTAB-11791)", "atlas_qc_by_sample")
    add("atlas", "n_cells_qc", int(qc.n_pass.sum()), "cells passing QC", "atlas_qc_by_sample")
    st = read("SELECT state, state_group, count(*) n FROM atlas_cells_state GROUP BY 1, 2")
    st["state_group"] = st.state_group.astype(str)
    add("atlas", "n_cells_states", int(st.n.sum()), "cells in curated states", "atlas_cells_state")
    add("atlas", "n_states", len(st), "curated cell states", "atlas_cells_state")
    for c in COMP:
        add("atlas", f"n_states_{c}", int((st.state_group == c).sum()), f"states in {c}", "atlas_cells_state")
    sig = read("state_signature_summary")
    add("atlas", "n_states_usable_sig", int(sig.usable.sum()), "states with a usable bulk signature (>=10 genes)",
        "state_signature_summary")
    add("atlas", "median_sig_genes", int(sig[sig.usable].n_genes.median()), "median signature size",
        "state_signature_summary")


def genetics():
    L = read("scdrs_state_load")
    L["state_group"] = L.state_group.astype(str)
    for t, lab in TRAITS.items():
        d = L[L.trait == t]
        if d.empty:
            continue
        for c in COMP:
            s = d[d.state_group == c]
            add("genetics", f"{t}_median_z_{c}", round(s.assoc_mcz.median(), 2), f"{lab}: median scDRS z, {c}",
                "scdrs_state_load")
            add("genetics", f"{t}_nfdr_{c}", f"{int((s.fdr < 0.05).sum())}/{len(s)}",
                f"{lab}: states FDR<0.05, {c}", "scdrs_state_load")
        add("genetics", f"{t}_nfdr_all", f"{int((d.fdr < 0.05).sum())}/{len(d)}", f"{lab}: states FDR<0.05",
            "scdrs_state_load")
        s = d[d.state_group == "Stromal"]
        add("genetics", f"{t}_stromal_n_negative", f"{int((s.assoc_mcz < 0).sum())}/{len(s)}",
            f"{lab}: stromal states with z<0", "scdrs_state_load")
        imm = d[d.state_group.isin(IMMUNE)].assoc_mcz
        stro = d[d.state_group == "Stromal"].assoc_mcz
        add("genetics", f"{t}_mw_immune_vs_stromal", fp(stats.mannwhitneyu(imm, stro, alternative="greater")[1]),
            f"{lab}: MW P immune > stromal state z (one-sided)", "scdrs_state_load")
    C = read("scdrs_trait_concordance")
    for a, b in [("ra_eur_noMHC", "fg_ra_noMHC"), ("ra_eur", "fg_ra"), ("ra_eur", "ra_eur_noMHC"),
                 ("fg_ra_seropos", "fg_ra_seroneg"), ("ra_eur_noMHC", "fg_oa"), ("ra_eur_noMHC", "fg_height")]:
        r = C[(C.trait_a == a) & (C.trait_b == b)].rho
        if len(r):
            add("genetics", f"rho_{a}__{b}", round(r.iloc[0], 2), f"Spearman rho of state z: {a} vs {b}",
                "scdrs_trait_concordance")
    base = L[L.trait == "ra_eur_noMHC"].set_index("state").assoc_mcz
    for n in (500, 2000):
        t = read(f"scdrs_group_state_top{n}")
        t = t[t.trait == "ra_eur_noMHC"].set_index("state").assoc_mcz
        add("genetics", f"rho_top{n}_vs_top1000", round(stats.spearmanr(base.reindex(t.index), t)[0], 2),
            f"ra_eur_noMHC: state z with top-{n} vs top-1000 genes", f"scdrs_group_state_top{n}")


def cohorts():
    M = read("bulk_samples")
    base = M[M.timepoint == "baseline"]
    for coh, g in base.groupby("cohort"):
        r = g.response_binary.dropna()
        add("cohorts", f"{coh}_n_baseline", len(g), f"{coh}: baseline biopsies scored", "bulk_samples")
        if len(r):
            add("cohorts", f"{coh}_n_response", f"{len(r)} ({int((r == 0).sum())} non-responders)",
                f"{coh}: baseline with binary response", "bulk_samples")
    for coh, t in [("R4RA", "r4ra_salmon_qc"), ("STRAP", "strap_salmon_qc"), ("PEAC", "peac_salmon_qc")]:
        q = read(t)
        add("cohorts", f"{coh}_n_quantified", len(q), f"{coh}: runs quantified with Salmon", t)
        add("cohorts", f"{coh}_mapping_median", round(q.percent_mapped.median(), 1),
            f"{coh}: median Salmon mapping rate (%)", t)
    P = read("pathotype_assoc_state")
    for coh, g in P.groupby("cohort"):
        add("cohorts", f"{coh}_pathotype_nfdr", f"{int((g.fdr < 0.05).sum())}/{len(g)}",
            f"{coh}: states differing by pathotype (KW, FDR<0.05)", "pathotype_assoc_state")
        for comp, pt in [("Stromal", "Fibroid"), ("T/NK", "Lymphoid"), ("B/plasma", "Lymphoid")]:
            s = g[g.state.isin(read("state_signature_summary").query("state_group == @comp").state)]
            add("cohorts", f"{coh}_pathotype_top_{comp}", f"{int((s.top_pathotype == pt).sum())}/{len(s)}",
                f"{coh}: {comp} states highest in {pt}", "pathotype_assoc_state")
    S = read("strap_sample_match")
    u = S[S.used]
    add("cohorts", "strap_n_matched", f"{len(u)}/{len(S)}", "STRAP biopsies confidently linked to published arm/ACR20",
        "strap_sample_match")
    add("cohorts", "strap_matched_arms", "; ".join(f"{k} {v}" for k, v in u.treatment.value_counts().items()),
        "STRAP matched per arm", "strap_sample_match")
    add("cohorts", "strap_profile_r_median", round(u.profile_r.median(), 2),
        "median correlation of recomputed vs published AMP1 profile (matched)", "strap_sample_match")


def response():
    R = read("response_assoc_state")
    R["state_group"] = R.state_group.astype(str)
    for coh, g in R.groupby("cohort"):
        add("response", f"{coh}_n_nominal", f"{int((g.p < 0.05).sum())}/{len(g)}",
            f"{coh}: states nominally associated with non-response", "response_assoc_state")
        add("response", f"{coh}_min_fdr", round(g.fdr.min(), 3), f"{coh}: smallest FDR", "response_assoc_state")
        for c in COMP:
            add("response", f"{coh}_median_logOR_{c}", round(g[g.state_group == c].log_or.median(), 2),
                f"{coh}: median log OR non-response, {c}", "response_assoc_state")
    r4 = R[R.cohort == "R4RA"].sort_values("p").iloc[0]
    add("response", "R4RA_top_state", f"{r4.state}: OR {r4.or_nonresp:.2f} ({r4.ci_lo:.2f}-{r4.ci_hi:.2f}), "
        f"P = {r4.p:.3f}, FDR = {r4.fdr:.2f}", "R4RA strongest state", "response_assoc_state")
    G = read("gli_response")
    for _, g in G.iterrows():
        add("response", f"gli_{g.cohort}_{g.trait}", f"OR {g.or_per_sd:.2f} ({g.ci_lo:.2f}-{g.ci_hi:.2f}), "
            f"P = {g.p:.2f}, perm P = {g.p_perm_one_sided:.2f}, n = {g.n} ({g.n_nonresp} NR)",
            f"GLI vs non-response: {g.cohort}, {g.weighting}", "gli_response")
    for trait in G.trait.unique():
        tr = G[(G.trait == trait) & G.cohort.isin(["R4RA", "STRAP"])]
        b, s, p, i2 = dl(tr.log_or.to_numpy(), tr.se.to_numpy())
        add("response", f"gli_pooled_trials_{trait}", f"OR {np.exp(b):.2f} ({np.exp(b - 1.96 * s):.2f}-"
            f"{np.exp(b + 1.96 * s):.2f}), P = {p:.2f}, I2 = {100 * i2:.0f}%", f"GLI pooled R4RA+STRAP, {trait}",
            "gli_response")
        al = G[(G.trait == trait) & (G.n_nonresp >= 8) & (G.n - G.n_nonresp >= 8)]
        b, s, p, i2 = dl(al.log_or.to_numpy(), al.se.to_numpy())
        add("response", f"gli_pooled_all_{trait}", f"OR {np.exp(b):.2f} ({np.exp(b - 1.96 * s):.2f}-"
            f"{np.exp(b + 1.96 * s):.2f}), P = {p:.2f}, I2 = {100 * i2:.0f}% ({', '.join(al.cohort)})",
            f"GLI pooled all eligible cohorts, {trait}", "gli_response")
    S = read("bulk_state_scores")
    M = read("bulk_samples")
    grp = read("SELECT DISTINCT state, state_group FROM atlas_cells_state")
    grp = grp.assign(state_group=grp.state_group.astype(str)).set_index("state").state_group
    for coh in ["R4RA", "STRAP"]:
        m = M[(M.cohort == coh) & (M.timepoint == "baseline") & M.response_binary.notna()].set_index("sample")
        W = S[S.cohort == coh].pivot(index="sample", columns="state", values="score").loc[m.index]
        cm = W.T.groupby(W.columns.map(grp)).mean().T
        for c in COMP:
            p = stats.mannwhitneyu(cm.loc[m.response_binary == 0, c], cm.loc[m.response_binary == 1, c])[1]
            d = cm.loc[m.response_binary == 0, c].median() - cm.loc[m.response_binary == 1, c].median()
            add("response", f"{coh}_compartment_mw_{c}", f"median NR - R = {d:+.2f}, MW P = {p:.3f}",
                f"{coh}: compartment-mean score, non-responders vs responders", "bulk_state_scores")
    D = read("strap_drug_specific")
    add("response", "strap_drug_min_p_interaction", round(D.p_interaction.min(), 2),
        "STRAP: smallest variable x arm interaction P", "strap_drug_specific")
    g = D[D.variable == "gli"]
    add("response", "strap_drug_gli", "; ".join(f"{r.arm} OR {r.or_per_sd:.2f} ({r.ci_lo:.2f}-{r.ci_hi:.2f})"
                                                for r in g.itertuples()) + f"; interaction P = {g.p_interaction.iloc[0]:.2f}",
        "STRAP GLI by arm", "strap_drug_specific")
    A = read("activity_assoc_pooled")
    for _, a in A[A.clinical.isin(["crp", "das28", "esr", "sjc"])].iterrows():
        add("activity", f"{a.variable}_{a.clinical}", f"rho {a.rho:.2f} ({a.ci_lo:.2f}-{a.ci_hi:.2f}), "
            f"P = {fp(a.p)}, I2 = {100 * a.i2:.0f}%, n = {a.n_total}", f"pooled baseline {a.variable} vs {a.clinical}",
            "activity_assoc_pooled")
    AC = read("activity_assoc")
    add("activity", "cohorts", ", ".join(f"{c} (n = {AC[AC.cohort == c].n.max()})" for c in AC.cohort.unique()),
        "cohorts in activity analysis", "activity_assoc")


def remodeling():
    R = read("remodel_state")
    for coh, g in R.groupby("cohort"):
        add("remodel", f"{coh}_n_nominal_interaction", f"{int((g.p_interaction < 0.05).sum())}/{len(g)}",
            f"{coh}: states with nominal time x response interaction", "remodel_state")
        add("remodel", f"{coh}_min_fdr_interaction", round(g.fdr_interaction.min(), 2), f"{coh}: smallest FDR",
            "remodel_state")
    for coh, g in R.groupby("cohort"):
        add("remodel", f"{coh}_n_fdr_resp", f"{int((g.fdr_resp < 0.05).sum())}/{len(g)}",
            f"{coh}: states changing in responders (FDR<0.05)", "remodel_state")
        add("remodel", f"{coh}_n_fdr_nonresp", f"{int((g.fdr_nonresp < 0.05).sum())}/{len(g)}",
            f"{coh}: states changing in non-responders (FDR<0.05)", "remodel_state")
        med = g.assign(state_group=g.state_group.astype(str)).groupby("state_group")[["change_resp", "change_nonresp"]].median()
        add("remodel", f"{coh}_median_change", "; ".join(f"{c} R {med.loc[c].change_resp:+.2f} / NR {med.loc[c].change_nonresp:+.2f}"
                                                          for c in COMP), f"{coh}: median change by compartment", "remodel_state")
    w = R.pivot(index="state", columns="cohort", values="change_resp")
    r, p = stats.spearmanr(w["R4RA"], w["GSE198520"])
    add("remodel", "rho_resp_change_cohorts", f"{r:.2f} (P = {fp(p)})", "responder change R4RA vs GSE198520",
        "remodel_state")
    for _, g in read("remodel_gli").iterrows():
        add("remodel", f"{g.cohort}_gli_delta", f"median change responders {g.median_delta_resp:.2f} vs "
            f"non-responders {g.median_delta_nonresp:.2f}, MW P = {g.p_mw:.2f} ({g.n_resp} R / {g.n_nonresp} NR)",
            f"{g.cohort}: change in GLI", "remodel_gli")


def spatial():
    q = read("xenium_sample_qc")
    add("spatial", "n_patients", len(q), "Xenium 5k pre-treatment sections", "xenium_sample_qc")
    add("spatial", "n_cells", int(q.n_cells.sum()), "segmented cells", "xenium_sample_qc")
    n = read("xenium_niche_response")
    add("spatial", "n_niches", len(n), "k-means niches", "xenium_niche_response")
    add("spatial", "niche_min_p", round(n.p.min(), 2), "smallest remission vs non-remission niche P",
        "xenium_niche_response")
    add("spatial", "n_rem_non", f"{int(n.n_rem.iloc[0])} remission / {int(n.n_non.iloc[0])} non-remission",
        "patients by outcome", "xenium_niche_response")
    s = read("xenium_state_load_niche").iloc[0]
    add("spatial", "rho_nb_load_same", f"{s.rho:.2f} (P = {fp(s.p)})", "state load vs neighbourhood load (all)",
        "xenium_state_load_niche")
    add("spatial", "rho_nb_load_xcomp", f"{s.rho_xcomp:.2f} (P = {fp(s.p_xcomp)})",
        "state load vs cross-compartment neighbourhood load", "xenium_state_load_niche")


def targets():
    summ = read("ot_target_state_summary")
    r, p = stats.spearmanr(summ.assoc_mcz, summ.frac_genetic)
    add("targets", "rho_load_frac_genetic", f"{r:.2f} (P = {fp(p)})",
        "state genetic load vs % specific genes that are genetic RA targets", "ot_target_state_summary")
    ot = read("ot_ra_targets")
    add("targets", "n_ot_targets", len(ot), "Open Targets RA-associated targets", "ot_ra_targets")
    add("targets", "n_ot_genetic", int((ot.genetic_association >= 0.5).sum()), "genetic_association >= 0.5",
        "ot_ra_targets")
    add("targets", "n_ot_clinical", int((ot.clinical >= 0.5).sum()), "clinical >= 0.5", "ot_ra_targets")
    pb = read("state_pseudobulk_long")
    pb["state_group"] = pb.state_group.astype(str)
    spec = pb[(pb.spec_z >= 2) & (pb.pct_detected >= 0.10)].merge(ot, on="gene", how="left").fillna(
        {"genetic_association": 0.0, "clinical": 0.0})
    pg = spec.groupby("gene").agg(immune=("state_group", lambda s: s.isin(IMMUNE).any()),
                                  stromal=("state_group", lambda s: (s == "Stromal").any()),
                                  ga=("genetic_association", "first"), clin=("clinical", "first"))
    imm, stro = pg[pg.immune & ~pg.stromal], pg[pg.stromal & ~pg.immune]
    for lab, d in [("immune", imm), ("stromal", stro)]:
        k = int((d.ga >= 0.5).sum())
        add("targets", f"{lab}_specific_genetic", f"{k}/{len(d)} ({100 * k / len(d):.1f}%)",
            f"{lab}-specific genes with genetic support", "state_pseudobulk_long, ot_ra_targets")
        c = d[d.clin >= 0.5]
        add("targets", f"{lab}_specific_clinical", f"{len(c)} ({int((c.ga >= 0.5).sum())} with genetic support)",
            f"clinically pursued targets specific to {lab} states", "state_pseudobulk_long, ot_ra_targets")
    add("targets", "mw_immune_vs_stromal_ga", fp(stats.mannwhitneyu(imm.ga, stro.ga, alternative="greater")[1]),
        "MW P genetic score immune- > stromal-specific genes", "state_pseudobulk_long, ot_ra_targets")
    k1, n1, k2, n2 = int((imm.ga >= .5).sum()), len(imm), int((stro.ga >= .5).sum()), len(stro)
    add("targets", "fisher_immune_vs_stromal", fp(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]),
        "Fisher P genetic-target share immune vs stromal", "state_pseudobulk_long, ot_ra_targets")


def have_tables():
    from archive import connect
    con = connect(read_only=True)
    try:
        return set(con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()


def upgrade():
    T = have_tables()
    if "genetic_map_matrix" in T:
        g = read("genetic_map_matrix")
        for r in g.itertuples():
            add("genmap", f"{r.method}|{r.atlas}|{r.trait}|{r.compartment}",
                f"{r.n_sig}/{r.n_states}; {r.summary_type} {r.summary:.2f}",
                f"{r.method}, {r.atlas}, {r.trait}: {r.compartment}", "genetic_map_matrix")
    if "ra_causal_genes" in T:
        c = read("ra_causal_genes")
        ev = pd.Series([e for x in c.evidence for e in x.split(";")]).value_counts()
        add("genmap", "n_causal_genes", len(c), "likely causal RA genes (MHC excluded)", "ra_causal_genes")
        add("genmap", "causal_evidence", "; ".join(f"{k} {v}" for k, v in ev.items()), "evidence counts",
            "ra_causal_genes")
    if "causal_gene_state_enrichment" in T:
        e = read("causal_gene_state_enrichment")
        e = e[e.state.str.startswith("[compartment]")]
        for r in e.itertuples():
            add("genmap", f"causal|{r.atlas}|{r.gene_set}|{r.state_group}", f"z {r.enrich_z:.2f}, P = {fp(r.p)}, "
                f"n genes {r.n_genes}", "compartment enrichment of causal genes", "causal_gene_state_enrichment")
    if "gupta_state_summary" in T:
        gs = read("gupta_state_summary")
        add("genmap", "gupta_atlas", f"{int(gs.n_cells.sum())} nuclei, {len(gs)} states, 11 RA donors; "
            + "; ".join(f"{k} {v}" for k, v in gs.groupby("state_group").state.size().items()),
            "replication atlas", "gupta_state_summary")
    for tab, lab in [("scdrs_trait_concordance", "")]:
        pass
    if "activity_state_test" in T:
        for r in read("activity_state_test").itertuples():
            add("activity", f"state_test|{r.clinical}|{r.scope}", f"rho {r.rho:.2f} (P = {fp(r.p)}, {r.n_states} states)",
                "state-level genetic load vs activity association", "activity_state_test")
    if "activity_longitudinal_pooled" in T:
        for r in read("activity_longitudinal_pooled").itertuples():
            add("activity", f"longitudinal|{r.variable}|{r.clinical}", f"rho {r.rho:.2f} ({r.ci_lo:.2f}-{r.ci_hi:.2f}),"
                f" P = {fp(r.p)}, I2 = {100 * r.i2:.0f}%, n = {r.n_total}"
                + (f", perm P = {r.p_perm_weights:.2f}" if r.p_perm_weights == r.p_perm_weights else ""),
                "within-patient change association (pooled)", "activity_longitudinal_pooled")
    if "clin_activity_harmonised" in T:
        h = read("clin_activity_harmonised")
        add("activity", "das28_sources", "; ".join(f"{c}: {t}" for c, t in h.groupby("cohort").das28_type.first().items()),
            "DAS28 source per cohort", "clin_activity_harmonised")


def chromatin():
    """S-LDSC on synovial chromatin (27_sldsc_chromatin.py) and the Dupuytren control."""
    T = have_tables()
    if "sldsc_annotations" in T:
        for r in read("sldsc_annotations").itertuples():
            add("chromatin", f"annot|{r.annotation}", f"{r.n_intervals_hg19} regions, {r.mb_hg19:.1f} Mb, "
                f"median {r.median_bp:.0f} bp, {r.n_1000g_snps} 1000G SNPs", "S-LDSC annotation size (hg19)",
                "sldsc_annotations")
    if "sldsc_chromatin" in T:
        for r in read("sldsc_chromatin").itertuples():
            add("chromatin", f"sldsc|{r.annotation}|{r.trait}",
                f"enrichment {r.enrichment:.2f} (SE {r.enrichment_se:.2f}; 95% CI {r.enrichment - 1.96 * r.enrichment_se:.2f}"
                f" to {r.enrichment + 1.96 * r.enrichment_se:.2f}); prop h2 {100 * r.prop_h2:.1f}% "
                f"(SE {100 * r.prop_h2_se:.1f}) of {100 * r.prop_snps:.2f}% SNPs; tau z {r.coef_z:.2f}, one-sided "
                f"P = {fp(r.coef_p)}, FDR {r.coef_fdr:.3f}; tau* {r.tau_star:.2f}",
                f"S-LDSC {r.annotation} x {r.trait} (baselineLD v2.2 + AllPeaks control)", "sldsc_chromatin")
    if "genetic_map_concordance" in T:
        for r in read("genetic_map_concordance").itertuples():
            add("genmap", f"concordance|{r.atlas}|{r.method}|{r.trait_a}|{r.trait_b}",
                f"rho {r.rho:.2f}, P = {fp(r.p)}, {r.n_states} states", "state-level concordance",
                "genetic_map_concordance")
    if "sldsc_chromatin" in T:                                  # upper 95% bound of the proportion of h2
        for r in read("sldsc_chromatin").itertuples():
            add("chromatin", f"h2_upper|{r.annotation}|{r.trait}", f"{100 * r.prop_h2:.1f}% (upper 95% bound "
                f"{100 * (r.prop_h2 + 1.96 * r.prop_h2_se):.1f}%)", "proportion of h2 and upper bound", "sldsc_chromatin")
    if "sldsc_chromatin_r2b" in T:
        for r in read("sldsc_chromatin_r2b").itertuples():
            add("chromatin_r2b", f"{r.model}|{r.annotation}|{r.trait}",
                f"enrichment {r.enrichment:.2f} (SE {r.enrichment_se:.2f}, marginal P = {fp(r.enrichment_p)}); prop h2 "
                f"{100 * r.prop_h2:.1f}% of {100 * r.prop_snps:.2f}% SNPs (upper {100 * (r.prop_h2 + 1.96 * r.prop_h2_se):.1f}%); "
                f"tau z {r.coef_z:.2f}, one-sided P = {fp(r.coef_p)}, FDR {r.coef_fdr:.3f}",
                "S-LDSC sensitivity", "sldsc_chromatin_r2b")
    if "finemap_overlap" in T:
        for r in read("finemap_overlap").itertuples():
            add("chromatin_r2b", f"finemap|{r.annotation}", f"{100 * r.share_pip:.1f}% of PIP ({r.n_variants_in} of "
                f"{r.n_variants} variants) vs {100 * r.share_snps:.2f}% of SNPs; fold {r.fold:.1f}", "fine-mapped overlap",
                "finemap_overlap")
    if "spikein_power" in T:
        for r in read("spikein_power").itertuples():
            add("power", f"spikein|k={r.k:.2f}", f"stromal {r.n_stromal_sig}/{r.n_stromal} (median z {r.median_stromal_z:.2f}); "
                f"T/NK {r.n_TNK_sig}/{r.n_TNK} (median z {r.median_TNK_z:.2f})", "scDRS spike-in", "spikein_power")
    if "activity_partial" in T:
        for r in read("activity_partial").query("cohort == 'pooled'").itertuples():
            add("activity", f"partial|{r.variable}|adj {r.adjusted_for}|{r.clinical}",
                f"rho {r.rho:.2f} ({r.ci_lo:.2f} to {r.ci_hi:.2f}), P = {fp(r.p)}, n = {r.n}", "partial Spearman",
                "activity_partial")
    if "target_tests" in T:
        for r in read("target_tests").itertuples():
            add("targets_r2", f"{r.assignment}|{r.evidence}|{r.threshold}", f"{r.immune_hit}/{r.immune_n} vs "
                f"{r.stromal_hit}/{r.stromal_n}; OR {r.odds_ratio:.2f}; P = {fp(r.p)}", "target share tests", "target_tests")
    for tab in ("magma_genes_fg_dupuytren",):
        if tab in T:
            g = read(tab)
            add("chromatin", "dupuytren_magma_bonf", int((g.P < 0.05 / len(g)).sum()),
                "Dupuytren MAGMA genes at Bonferroni", tab)


def main():
    for f in (atlas, genetics, cohorts, response, remodeling, spatial, targets, upgrade, chromatin):
        f()
    K = pd.DataFrame(ROWS)
    write_table(K, "key_numbers", script=__file__, datasets=["all archived result tables"],
                note="numbers quoted in the manuscript")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.max_rows", 500)
    for sec, g in K.groupby("section", sort=False):
        print(f"\n## {sec}")
        print(g[["key", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
