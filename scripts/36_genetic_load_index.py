"""36_genetic_load_index.py — patient-level test of the thesis (Fig 4): Genetic Load Index (GLI) vs non-response.

The state-level Spearman test (35_*) treats 52 states as exchangeable, but bulk state scores are strongly correlated
within compartments, so its permutation P is optimistic. Here the unit is the patient:
  GLI_i = mean_s( z_is * w_s ),  z_is = within-cohort z of baseline state score s in sample i,
                                 w_s  = standardised scDRS genetic load of state s (ra_eur_noMHC; mean 0, SD 1)
One pre-specified test per cohort: logistic non-response ~ z(GLI)  (thesis: OR < 1).
Specificity: GLI rebuilt with 2,000 random permutations of w over states -> empirical P = share of random
weightings with an association at least as protective as the observed one (one-sided, thesis direction).
Replication weights: FinnGen RA (fg_ra_noMHC). Contrast weights: OA (fg_oa).
DuckDB: gli_samples (cohort, sample, GLI variants), gli_response (cohort x weighting: OR, CI, P, perm P)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_table  # noqa: E402

N_PERM = 2000
WEIGHTS = {"ra_eur_noMHC": "RA (Ishigaki, MHC removed)", "fg_ra_noMHC": "RA (FinnGen, MHC removed)",
           "fg_oa": "Osteoarthritis (contrast)"}


def logit_or(y, x):
    x = (x - x.mean()) / x.std()
    fit = sm.Logit(y, sm.add_constant(x.rename("gli"))).fit(disp=0)
    lo, hi = np.exp(fit.conf_int().loc["gli"])
    return fit.params["gli"], fit.bse["gli"], np.exp(fit.params["gli"]), lo, hi, fit.pvalues["gli"]


def main():
    S = read("bulk_state_scores")
    M = read("bulk_samples")
    load = read("scdrs_state_load")
    base = M[(M.timepoint == "baseline") & M.response_binary.notna()]
    rng = np.random.default_rng(20260926)
    samples, results = [], []
    for coh, m in base.groupby("cohort"):
        W = S[S.cohort == coh].pivot(index="sample", columns="state", values="score").loc[m["sample"]]
        Z = (W - W.mean()) / W.std()
        y = (1 - m.set_index("sample").response_binary).loc[Z.index]
        n_nr, n = int(y.sum()), len(y)
        out = pd.DataFrame({"cohort": coh, "sample": Z.index, "nonresponse": y.values})
        for trait, label in WEIGHTS.items():
            w = load[load.trait == trait].set_index("state").assoc_mcz.reindex(Z.columns)
            w = (w - w.mean()) / w.std()
            gli = (Z * w).mean(axis=1)
            out[f"gli_{trait}"] = gli.values
            b, se, orr, lo, hi, p = logit_or(y, gli)
            perm = np.empty(N_PERM)
            for k in range(N_PERM):
                perm[k] = logit_or(y, (Z * rng.permutation(w.values)).mean(axis=1))[0]
            p_perm = (np.sum(perm <= b) + 1) / (N_PERM + 1)
            results.append(dict(cohort=coh, weighting=label, trait=trait, n=n, n_nonresp=n_nr, log_or=b, se=se,
                                or_per_sd=orr, ci_lo=lo, ci_hi=hi, p=p, p_perm_one_sided=p_perm))
        samples.append(out)
    R = pd.DataFrame(results)
    write_table(pd.concat(samples, ignore_index=True), "gli_samples", script=__file__,
                datasets=sorted(R.cohort.unique()), note="GLI = mean(z state score * standardised scDRS load)")
    write_table(R, "gli_response", script=__file__, datasets=sorted(R.cohort.unique()),
                note=f"logit NR ~ z(GLI); perm P: {N_PERM} random state-weight permutations, one-sided (OR<1)")
    pd.set_option("display.width", 200)
    print(R[["cohort", "weighting", "n", "n_nonresp", "or_per_sd", "ci_lo", "ci_hi", "p", "p_perm_one_sided"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
