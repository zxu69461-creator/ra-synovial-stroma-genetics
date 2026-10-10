"""34_disease_activity.py — the genetically loaded synovial programme vs disease activity, across and within patients.

Fix (2026-10-02): the ArrayExpress field "arthritis disease activity score measurement" (R4RA) / "arthritis.activity"
(STRAP) is the patient global assessment VAS (0-100), not DAS28. DAS28 is now computed from its components.

Clinical harmonisation (table clin_activity_harmonised; one row per biopsy):
  R4RA, STRAP : DAS28-ESR = 0.56*sqrt(TJC28) + 0.28*sqrt(SJC28) + 0.70*ln(ESR) + 0.014*PtGA(0-100);
                CDAI = TJC28 + SJC28 + PtGA/10 + PhGA/10
  PEAC        : DAS28 as deposited (E-MTAB-6141)
  GSE198520   : "das28 score" as deposited (pre and post)
  GSE45867    : DAS28-CRP as deposited (pre and post)
  CRP analysed as log(1 + CRP).

Genetic Load Index (GLI) per biopsy = mean_s(z_is * w_s), z within cohort over all its biopsies (baseline + post),
w = standardised scDRS load (ra_eur_noMHC; contrast fg_oa).

Pre-specified tests
  A cross-sectional (baseline biopsies): Spearman rho of GLI / compartment means vs DAS28, log CRP, ESR, SJC, CDAI
    per cohort (n >= 15), pooled by Fisher-z DerSimonian-Laird.
  A' specificity: GLI vs DAS28 / CRP / ESR against 1,000 random permutations of w (one-sided; activity_assoc_pooled
     column p_perm_weights). Result 2026-10-02: not specific (P 0.10-0.17) -> coupling is compartment-level.
  B within-patient (paired biopsies; R4RA week 16, GSE198520 week 12, GSE45867 week 12): Spearman rho of
    delta-GLI vs delta-DAS28 and delta-log CRP, per cohort and pooled; specificity: 2,000 random permutations of w
    across states (one-sided empirical P that a random weighting couples at least as strongly; pooled statistic).
DuckDB: clin_activity_harmonised, activity_assoc, activity_assoc_pooled, activity_longitudinal,
        activity_longitudinal_pooled, activity_state (per state: pooled baseline rho with DAS28 / log CRP, genetic load),
        activity_state_test (state-level Spearman of load vs activity rho: all states, and within each compartment)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_table  # noqa: E402
from cohorts import _geo_characteristics  # noqa: E402

COMP = ["T/NK", "B/plasma", "Myeloid", "Endothelial", "Stromal"]
CLIN = ["das28", "crp", "esr", "sjc", "cdai"]
GEO = ROOT / "raw" / "geo"
N_PERM = 2000


def num(x):
    return pd.to_numeric(x, errors="coerce")


def das28_esr(tjc, sjc, esr, ptga):
    return 0.56 * np.sqrt(tjc) + 0.28 * np.sqrt(sjc) + 0.70 * np.log(esr.clip(lower=1)) + 0.014 * ptga


def harmonise() -> pd.DataFrame:
    rows = []
    for coh, tab, idx in [("R4RA", "clin_r4ra", "sample_id"), ("STRAP", "clin_strap", "sampleid")]:
        c = read(tab).set_index(idx)
        pt = num(c["arthritis_disease_activity_score_measurement" if coh == "R4RA" else "arthritis_activity"])
        tjc, sjc, esr = num(c.tender_joints_counts_tjc), num(c.swollen_joints_counts_sjc), num(c.esr)
        rows.append(pd.DataFrame({"cohort": coh, "sample": c.index, "das28": das28_esr(tjc, sjc, esr, pt).values,
                                  "das28_type": "DAS28-ESR (computed)", "crp": num(c.crp).values,
                                  "esr": esr.values, "sjc": sjc.values, "tjc": tjc.values, "ptga": pt.values,
                                  "cdai": (tjc + sjc + pt / 10 + num(c.physicians_global_assessment) / 10).values}))
    c = read("clin_peac").query("organism_part == 'synovium'")
    rows.append(pd.DataFrame({"cohort": "PEAC", "sample": c.source_name.str.replace(" ", "_").values,
                              "das28": num(c.das28).values, "das28_type": "DAS28 (deposited)",
                              "crp": num(c.crp).values, "esr": num(c.esr).values, "sjc": num(c.swollen).values,
                              "tjc": num(c.tender).values, "ptga": num(c.vas).values, "cdai": np.nan}))
    ch = _geo_characteristics(GEO / "GSE198520_series_matrix.txt.gz")
    rows.append(pd.DataFrame({"cohort": "GSE198520", "sample": ch.index, "das28": num(ch["das28 score"]).values,
                              "das28_type": "DAS28 (deposited)", "crp": num(ch["crp"]).values,
                              "esr": num(ch["esr"]).values, "sjc": num(ch["swollen joint count (28)"]).values,
                              "tjc": num(ch["tender joint count (28)"]).values, "ptga": np.nan, "cdai": np.nan}))
    ch = _geo_characteristics(GEO / "GSE45867_series_matrix.txt.gz")
    rows.append(pd.DataFrame({"cohort": "GSE45867", "sample": ch.index,
                              "das28": num(ch["disease activity score (das)-28crp"]).values,
                              "das28_type": "DAS28-CRP (deposited)", "crp": np.nan, "esr": np.nan, "sjc": np.nan,
                              "tjc": np.nan, "ptga": np.nan, "cdai": np.nan}))
    H = pd.concat(rows, ignore_index=True)
    H["crp"] = np.log1p(H.crp)
    return H


def dl_fisher(r, n):
    z, v = np.arctanh(r), 1 / (n - 3)
    w = 1 / v
    fe = np.sum(w * z) / np.sum(w)
    q = np.sum(w * (z - fe) ** 2)
    df = len(z) - 1
    tau2 = max(0.0, (q - df) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if df > 0 else 0.0
    wr = 1 / (v + tau2)
    b, se = np.sum(wr * z) / np.sum(wr), np.sqrt(1 / np.sum(wr))
    i2 = max(0.0, (q - df) / q) if q > 0 and df > 0 else 0.0
    return np.tanh(b), np.tanh(b - 1.96 * se), np.tanh(b + 1.96 * se), 2 * stats.norm.sf(abs(b / se)), i2, b / se


def pool(R, keys):
    out = []
    for k, g in R.groupby(keys):
        r, lo, hi, p, i2, z = dl_fisher(g.rho.to_numpy(), g.n.to_numpy())
        out.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), n_cohorts=len(g), n_total=int(g.n.sum()),
                        rho=r, ci_lo=lo, ci_hi=hi, p=p, i2=i2, z=z))
    return pd.DataFrame(out)


def main():
    H = harmonise()
    write_table(H, "clin_activity_harmonised", script=__file__,
                datasets=["E-MTAB-11611", "E-MTAB-13733", "E-MTAB-6141", "GSE198520", "GSE45867"],
                note="DAS28-ESR computed for R4RA/STRAP (the deposited 0-100 field is patient global VAS); log1p CRP")
    S, M, load = read("bulk_state_scores"), read("bulk_samples"), read("scdrs_state_load")
    grp = read("SELECT DISTINCT state, state_group FROM atlas_cells_state")
    grp = grp.assign(state_group=grp.state_group.astype(str)).set_index("state").state_group
    w = {t: load[load.trait == t].set_index("state").assoc_mcz for t in ("ra_eur_noMHC", "fg_oa")}
    rng = np.random.default_rng(20261002)
    cross, longi, perm_inputs = [], [], []
    for coh in ["R4RA", "STRAP", "PEAC", "GSE198520", "GSE45867"]:
        m = M[M.cohort == coh].set_index("sample")
        W = S[S.cohort == coh].pivot(index="sample", columns="state", values="score").loc[m.index]
        Z = (W - W.mean()) / W.std()
        var = pd.DataFrame(index=Z.index)
        for t, lab in [("ra_eur_noMHC", "GLI (RA)"), ("fg_oa", "GLI (OA contrast)")]:
            ww = w[t].reindex(Z.columns)
            var[lab] = (Z * ((ww - ww.mean()) / ww.std())).mean(axis=1)
        cm = Z.T.groupby(Z.columns.map(grp)).mean().T
        for c in COMP:
            var[c] = cm[c]
        cl = H[H.cohort == coh].set_index("sample").reindex(Z.index)
        base = m.timepoint == "baseline"
        for v in var.columns:                                           # A: cross-sectional, baseline biopsies
            for k in CLIN:
                ok = base & cl[k].notna()
                if ok.sum() >= 15:
                    rho, p = stats.spearmanr(var.loc[ok, v], cl.loc[ok, k])
                    cross.append(dict(cohort=coh, variable=v, clinical=k, n=int(ok.sum()), rho=rho, p=p))
        if (m.timepoint == "post").any():                              # B: within-patient change
            d = pd.concat([var, cl[["das28", "crp"]], m[["patient", "timepoint"]]], axis=1)
            piv = {k: d.pivot_table(index="patient", columns="timepoint", values=k) for k in list(var.columns) + ["das28", "crp"]}
            for v in var.columns:
                for k in ["das28", "crp"]:
                    if not {"post", "baseline"} <= set(piv[k].columns):
                        continue
                    dd = pd.DataFrame({"x": piv[v].post - piv[v].baseline, "y": piv[k].post - piv[k].baseline}).dropna()
                    if len(dd) >= 10:
                        rho, p = stats.spearmanr(dd.x, dd.y)
                        longi.append(dict(cohort=coh, variable=v, clinical=k, n=len(dd), rho=rho, p=p))
            pz = Z.join(m[["patient", "timepoint"]])
            dz = (pz[pz.timepoint == "post"].set_index("patient").drop(columns="timepoint")
                  - pz[pz.timepoint == "baseline"].set_index("patient").drop(columns="timepoint")).dropna(how="all")
            dy = (piv["das28"].post - piv["das28"].baseline).reindex(dz.index)
            ok = dy.notna() & dz.notna().all(axis=1)
            perm_inputs.append((coh, dz[ok], dy[ok]))
    # per state: pooled baseline association with activity, against its genetic load
    st_rows = []
    for coh in ["R4RA", "STRAP", "PEAC", "GSE198520", "GSE45867"]:
        m = M[(M.cohort == coh) & (M.timepoint == "baseline")].set_index("sample")
        Wb = S[S.cohort == coh].pivot(index="sample", columns="state", values="score")
        Zb = ((Wb - Wb.mean()) / Wb.std()).loc[m.index]
        clb = H[H.cohort == coh].set_index("sample").reindex(Zb.index)
        for k in ["das28", "crp"]:
            ok = clb[k].notna()
            if ok.sum() >= 15:
                for st in Zb.columns:
                    st_rows.append(dict(cohort=coh, state=st, clinical=k, n=int(ok.sum()),
                                        rho=stats.spearmanr(Zb.loc[ok, st], clb.loc[ok, k])[0]))
    SP = pool(pd.DataFrame(st_rows), ["state", "clinical"])
    SP["state_group"] = SP.state.map(grp)
    SP["genetic_load"] = SP.state.map(w["ra_eur_noMHC"])
    tests = []
    for k, g in SP.groupby("clinical"):
        r, p = stats.spearmanr(g.genetic_load, g.rho)
        tests.append(dict(clinical=k, scope="all states", n_states=len(g), rho=r, p=p))
        for c, gc_ in g.groupby("state_group"):
            if len(gc_) >= 5:
                r, p = stats.spearmanr(gc_.genetic_load, gc_.rho)
                tests.append(dict(clinical=k, scope=f"within {c}", n_states=len(gc_), rho=r, p=p))
    A, B = pd.DataFrame(cross), pd.DataFrame(longi)
    AP, BP = pool(A, ["variable", "clinical"]), pool(B, ["variable", "clinical"])
    # specificity of the genetic weighting for within-patient coupling (pooled Fisher-z statistic)
    wr = w["ra_eur_noMHC"]

    def pooled_z(weights):
        r_, n_ = [], []
        for coh, dz, dy in perm_inputs:
            ww = weights.reindex(dz.columns)
            g = (dz * ((ww - ww.mean()) / ww.std()).values).mean(axis=1)
            r_.append(stats.spearmanr(g, dy)[0])
            n_.append(len(dy))
        return dl_fisher(np.array(r_), np.array(n_))[5]
    obs = pooled_z(wr)
    null = np.array([pooled_z(pd.Series(rng.permutation(wr.values), index=wr.index)) for _ in range(N_PERM)])
    # cross-sectional specificity of the genetic weighting (pooled Fisher-z over cohorts, baseline biopsies)
    base_inputs = []
    for coh in ["R4RA", "STRAP", "PEAC", "GSE198520", "GSE45867"]:
        m = M[(M.cohort == coh) & (M.timepoint == "baseline")].set_index("sample")
        Wb = S[S.cohort == coh].pivot(index="sample", columns="state", values="score")
        Zb = ((Wb - Wb.mean()) / Wb.std()).loc[m.index]
        base_inputs.append((Zb, H[H.cohort == coh].set_index("sample").reindex(Zb.index)))

    def pooled_cross(weights, k):
        r_, n_ = [], []
        for Zb, clb in base_inputs:
            ok = clb[k].notna()
            if ok.sum() < 15:
                continue
            ww = weights.reindex(Zb.columns)
            g = (Zb[ok] * ((ww - ww.mean()) / ww.std()).values).mean(axis=1)
            r_.append(stats.spearmanr(g, clb.loc[ok, k])[0])
            n_.append(int(ok.sum()))
        return dl_fisher(np.array(r_), np.array(n_))[5]
    AP["p_perm_weights"] = np.nan
    for k in ["das28", "crp", "esr"]:
        o = pooled_cross(wr, k)
        nl = np.array([pooled_cross(pd.Series(rng.permutation(wr.values), index=wr.index), k) for _ in range(1000)])
        AP.loc[(AP.variable == "GLI (RA)") & (AP.clinical == k), "p_perm_weights"] = (np.sum(nl >= o) + 1) / 1001
    BP["p_perm_weights"] = np.where((BP.variable == "GLI (RA)") & (BP.clinical == "das28"),
                                    (np.sum(null >= obs) + 1) / (N_PERM + 1), np.nan)
    ds = ["E-MTAB-11611", "E-MTAB-13733", "E-MTAB-6141", "GSE198520", "GSE45867"]
    write_table(A, "activity_assoc", script=__file__, datasets=ds, note="baseline Spearman per cohort (n >= 15)")
    write_table(AP, "activity_assoc_pooled", script=__file__, datasets=ds, note="Fisher-z DerSimonian-Laird")
    write_table(B, "activity_longitudinal", script=__file__, datasets=ds,
                note="within-patient: Spearman of post-baseline change in score vs change in DAS28 / log CRP")
    write_table(BP, "activity_longitudinal_pooled", script=__file__, datasets=ds,
                note=f"Fisher-z DL; p_perm_weights = {N_PERM} random permutations of state weights (one-sided)")
    write_table(SP, "activity_state", script=__file__, datasets=ds,
                note="per state: Fisher-z pooled baseline Spearman rho with DAS28 / log CRP; genetic_load = scDRS z")
    write_table(pd.DataFrame(tests), "activity_state_test", script=__file__, datasets=ds,
                note="state-level Spearman(genetic load, activity rho); descriptive (states correlated)")
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    pd.set_option("display.width", 220)
    print(A.pivot_table(index="variable", columns=["clinical", "cohort"], values="rho").round(2).to_string())
    print(AP.round(4).to_string(index=False))
    print(B.pivot_table(index="variable", columns=["clinical", "cohort"], values="rho").round(2).to_string())
    print(BP.round(4).to_string(index=False))
    print("H counts:", H.groupby("cohort").das28.count().to_dict())


if __name__ == "__main__":
    main()
