"""32_response_association.py — baseline state scores vs treatment response, per cohort (Fig 3/4 y-axis).

Per cohort, baseline samples only:
  primary   : logistic  non_response ~ z(score)            (responder=0 / non-responder=1; moderate excluded)
  relative  : logistic  non_response ~ z(score - mean score of the state's compartment)  (composition-aware:
              removes "more immune tissue => less stroma" coupling and biopsy-sampling depth of a compartment)
  ordinal   : Spearman  score vs response rank (good=2 > moderate=1 > non=0) where a 3-level outcome exists
  control   : Kruskal-Wallis score ~ pathotype (positive control: lymphoid states high in lymphoid pathotype,
              stromal states high in fibroid/pauci-immune)
Output: OR per SD for NON-response (OR > 1 = state marks refractoriness), BH-FDR within cohort.
DuckDB: response_assoc_state, pathotype_assoc_state
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_table  # noqa: E402

RANK = {"responder": 2, "good": 2, "good-responder": 2, "mod-responder": 1, "moderate": 1, "moderate-responder": 1,
        "non-responder": 0, "non.responder": 0, "none": 0, "poor-responder": 0}


def main():
    S = read("bulk_state_scores")
    M = read("bulk_samples")
    grp = read("SELECT DISTINCT state, state_group FROM atlas_cells_state")
    grp["state_group"] = grp.state_group.astype(str)
    comp_of = grp.set_index("state").state_group
    base = M[M.timepoint == "baseline"]
    resp_rows, path_rows = [], []
    for coh, m in base.groupby("cohort"):
        w = S[S.cohort == coh].pivot(index="sample", columns="state", values="score").loc[m["sample"]]
        m = m.set_index("sample")
        y = 1 - m.response_binary
        rank = m.response_detail.str.lower().map(RANK)
        comp_mean = w.T.groupby(comp_of.reindex(w.columns).values).mean().T      # sample x compartment
        for st in w.columns:
            x = (w[st] - w[st].mean()) / w[st].std()
            ok = y.notna()
            pt = m.pathotype.where(~m.pathotype.isin(["Ungraded", "TBC", "", None]))
            groups = [x[pt == k].values for k in pt.dropna().unique() if (pt == k).sum() >= 3]
            if len(groups) >= 2:
                h, p_kw = stats.kruskal(*groups)
                means = x.groupby(pt).mean()
                path_rows.append(dict(cohort=coh, state=st, kw_h=h, p=p_kw, top_pathotype=means.idxmax(),
                                      **{f"mean_{k}": v for k, v in means.items()}))
            if ok.sum() < 10:                          # no response annotation (e.g. PEAC): pathotype control only
                continue
            fit = sm.Logit(y[ok], sm.add_constant(x[ok])).fit(disp=0)
            lo, hi = np.exp(fit.conf_int().loc[st])
            rho, p_rho = stats.spearmanr(x[rank.notna()], rank[rank.notna()]) if rank.nunique() > 2 else (np.nan, np.nan)
            rel = w[st] - comp_mean[comp_of[st]]
            xr = ((rel - rel.mean()) / rel.std()).rename(st)
            try:
                fr = sm.Logit(y[ok], sm.add_constant(xr[ok])).fit(disp=0)
                rel_or, rel_se, rel_p = fr.params[st], fr.bse[st], fr.pvalues[st]
            except Exception:                      # e.g. perfect separation in tiny cohorts
                rel_or, rel_se, rel_p = np.nan, np.nan, np.nan
            resp_rows.append(dict(cohort=coh, state=st, n=int(ok.sum()), n_nonresp=int(y[ok].sum()),
                                  log_or=fit.params[st], se=fit.bse[st], or_nonresp=np.exp(fit.params[st]),
                                  ci_lo=lo, ci_hi=hi, p=fit.pvalues[st], rho_response_rank=rho, p_rank=p_rho,
                                  log_or_rel=rel_or, se_rel=rel_se, p_rel=rel_p))
    R = pd.DataFrame(resp_rows)
    R["fdr"] = R.groupby("cohort").p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
    R = R.merge(grp, on="state", how="left")
    P = pd.DataFrame(path_rows)
    P["fdr"] = P.groupby("cohort").p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
    P = P.merge(grp, on="state", how="left")
    ds = sorted(R.cohort.unique())
    write_table(R, "response_assoc_state", script=__file__, datasets=ds,
                note="baseline; logit(non-response ~ z score); OR per SD; moderate responders excluded")
    write_table(P, "pathotype_assoc_state", script=__file__, datasets=ds, note="Kruskal-Wallis score ~ pathotype")
    pd.set_option("display.width", 220)
    for coh, r in R.groupby("cohort"):
        print(f"== {coh}: n={r.n.iloc[0]} ({r.n_nonresp.iloc[0]} non-responders)")
        print(r.sort_values("log_or")[["state_group", "state", "or_nonresp", "ci_lo", "ci_hi", "p", "fdr",
                                       "rho_response_rank"]].round(3).to_string(index=False))
        print(r.groupby("state_group").log_or.median().round(3).to_string())
    for coh, p in P.groupby("cohort"):
        print(f"== pathotype {coh}")
        print(p.sort_values("p")[["state_group", "state", "top_pathotype", "p", "fdr"]].head(20).round(4)
              .to_string(index=False))


if __name__ == "__main__":
    main()
