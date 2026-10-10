"""38_remodeling.py — Figure 5: treatment-induced remodeling of genetically loaded vs decoupled states.

Paired baseline / post-treatment biopsies: R4RA (week 16; rituximab or tocilizumab, arm not public) and GSE198520
(anti-TNF, week 12). State scores are within-cohort z across all samples of the cohort (pre and post together), so
pre/post values are on one scale.

Per state: linear mixed model  score ~ time * responder + (1 | patient)  (statsmodels MixedLM);
reported: time effect in responders and in non-responders, and the time x response interaction (BH-FDR per cohort).
Patient level: change in Genetic Load Index (GLI, 36_*) by response (Mann-Whitney).
R4RA arm inference (methodological note, not used for inference unless bimodal): week-16 minus baseline change of
the B-cell program (mean of B switched memory / B activated CD69+ / B unswitched memory scores); rituximab depletes
B cells, so a clear low mode = rituximab-treated. Reported with its distribution; used only if two modes separate.
DuckDB: remodel_state, remodel_gli, r4ra_inferred_arm
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_table  # noqa: E402

warnings.filterwarnings("ignore")
B_STATES = ["B switched memory", "B activated CD69+", "B unswitched memory"]


def paired(coh):
    S = read(f"SELECT * FROM bulk_state_scores WHERE cohort = '{coh}'")
    M = read(f"SELECT * FROM bulk_samples WHERE cohort = '{coh}'")
    M = M[M.response_binary.notna()]
    both = M.groupby("patient").timepoint.nunique()
    M = M[M.patient.isin(both[both == 2].index)]
    W = S.pivot(index="sample", columns="state", values="score").loc[M["sample"]]
    return W, M.set_index("sample")


def main():
    grp = read("SELECT DISTINCT state, state_group FROM atlas_cells_state")
    grp = grp.assign(state_group=grp.state_group.astype(str)).set_index("state").state_group
    rows, gli_rows, arm = [], [], None
    for coh in ["R4RA", "GSE198520"]:
        W, M = paired(coh)
        print(f"{coh}: {M.patient.nunique()} paired patients "
              f"({int(M.drop_duplicates('patient').response_binary.sum())} responders)")
        for st in W.columns:
            d = pd.DataFrame({"y": W[st], "time": (M.timepoint == "post").astype(int), "resp": M.response_binary,
                              "patient": M.patient})
            try:
                f = smf.mixedlm("y ~ time * resp", d, groups=d["patient"]).fit(reml=True)
                b_nr, p_nr = f.params["time"], f.pvalues["time"]
                b_int, p_int = f.params["time:resp"], f.pvalues["time:resp"]
                cov = f.cov_params()
                b_r = b_nr + b_int
                se_r = np.sqrt(cov.loc["time", "time"] + cov.loc["time:resp", "time:resp"]
                               + 2 * cov.loc["time", "time:resp"])
                p_r = 2 * stats.norm.sf(abs(b_r / se_r))
            except Exception:
                b_nr = p_nr = b_int = p_int = b_r = p_r = np.nan
            rows.append(dict(cohort=coh, state=st, state_group=grp.get(st), change_nonresp=b_nr, p_nonresp=p_nr,
                             change_resp=b_r, p_resp=p_r, interaction=b_int, p_interaction=p_int))
        gli = read(f"SELECT * FROM gli_samples WHERE cohort = '{coh}'") if coh else None
        # GLI for post samples: recompute with the same weights on this cohort's z scores
        load = read("scdrs_state_load").query("trait == 'ra_eur_noMHC'").set_index("state").assoc_mcz
        w = load.reindex(W.columns)
        w = (w - w.mean()) / w.std()
        Zall = read(f"SELECT * FROM bulk_state_scores WHERE cohort = '{coh}'").pivot(index="sample", columns="state",
                                                                                  values="score")
        Zall = (Zall - Zall.mean()) / Zall.std()
        g = (Zall.loc[W.index, W.columns] * w).mean(axis=1)
        gd = pd.DataFrame({"gli": g, "patient": M.patient, "time": M.timepoint, "resp": M.response_binary})
        wide = gd.pivot_table(index=["patient", "resp"], columns="time", values="gli").reset_index()
        wide["delta"] = wide.post - wide.baseline
        u, p = stats.mannwhitneyu(wide[wide.resp == 1].delta, wide[wide.resp == 0].delta)
        gli_rows.append(dict(cohort=coh, n_resp=int((wide.resp == 1).sum()), n_nonresp=int((wide.resp == 0).sum()),
                             median_delta_resp=wide[wide.resp == 1].delta.median(),
                             median_delta_nonresp=wide[wide.resp == 0].delta.median(), p_mw=p))
        if coh == "R4RA":
            b = W[[s for s in B_STATES if s in W.columns]].mean(axis=1)
            bw = pd.DataFrame({"b": b, "patient": M.patient, "time": M.timepoint}).pivot(
                index="patient", columns="time", values="b")
            arm = (bw.post - bw.baseline).rename("delta_B").reset_index()
    R = pd.DataFrame(rows)
    for c in ["p_resp", "p_nonresp", "p_interaction"]:
        R[c.replace("p_", "fdr_")] = R.groupby("cohort")[c].transform(
            lambda p: pd.Series(multipletests(p.fillna(1), method="fdr_bh")[1], index=p.index))
    write_table(R, "remodel_state", script=__file__, datasets=["E-MTAB-11611", "GSE198520"],
                note="MixedLM score ~ time*responder + (1|patient); change_* = post - baseline (z units)")
    G = pd.DataFrame(gli_rows)
    write_table(G, "remodel_gli", script=__file__, datasets=["E-MTAB-11611", "GSE198520"],
                note="change in GLI (post - baseline) by response; Mann-Whitney")
    write_table(arm, "r4ra_inferred_arm", script=__file__, datasets=["E-MTAB-11611"],
                note="week16 - baseline change of B-cell state scores; rituximab expected in the low mode")
    pd.set_option("display.width", 220)
    print(R.groupby(["cohort", "state_group"])[["change_resp", "change_nonresp", "interaction"]].median().round(2))
    print(R.sort_values("p_interaction").groupby("cohort").head(6)[
        ["cohort", "state", "change_resp", "change_nonresp", "interaction", "p_interaction", "fdr_interaction"]]
        .round(3).to_string(index=False))
    print(G.round(3).to_string(index=False))
    print("R4RA B-cell program change (week16 - baseline), sorted:", np.round(np.sort(arm.delta_B.values), 2))


if __name__ == "__main__":
    main()
