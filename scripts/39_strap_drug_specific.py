"""39_strap_drug_specific.py — exploratory supplement: drug-class specificity in STRAP (randomised: etanercept,
tocilizumab, rituximab), using the 135 confidently matched patients (37_strap_match.py).

1. GLI (RA-weighted) OR for non-response within each arm, and GLI x arm interaction (likelihood-ratio test).
2. Per compartment-mean score: OR within each arm + interaction LRT (5 tests, BH-FDR).
3. Positive control expected from trial biology: B/plasma program -> rituximab response (B-cell target).
Exploratory: arms are small (40-53); STRAP's own 18-subset analysis found no drug-specific subset (Supp Fig 3a).
DuckDB: strap_drug_specific
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_table  # noqa: E402

COMP = ["T/NK", "B/plasma", "Myeloid", "Endothelial", "Stromal"]


def lrt(full, red):
    stat = 2 * (full.llf - red.llf)
    return stats.chi2.sf(stat, full.df_model - red.df_model)


def main():
    M = read("SELECT * FROM bulk_samples WHERE cohort = 'STRAP'")
    M = M[M.response_binary.notna()].set_index("sample")
    S = read("SELECT * FROM bulk_state_scores WHERE cohort = 'STRAP'").pivot(index="sample", columns="state",
                                                                           values="score").loc[M.index]
    grp = read("SELECT DISTINCT state, state_group FROM atlas_cells_state")
    grp = grp.assign(state_group=grp.state_group.astype(str)).set_index("state").state_group
    Z = (S - S.mean()) / S.std()
    w = read("scdrs_state_load").query("trait == 'ra_eur_noMHC'").set_index("state").assoc_mcz.reindex(Z.columns)
    w = (w - w.mean()) / w.std()
    d = pd.DataFrame({"nr": 1 - M.response_binary, "arm": M.treatment, "gli": (Z * w).mean(axis=1)})
    comp = Z.T.groupby(Z.columns.map(grp)).mean().T
    for c in COMP:
        d[c.replace("/", "_")] = comp[c]
    rows = []
    for var in ["gli"] + [c.replace("/", "_") for c in COMP]:
        d["x"] = (d[var] - d[var].mean()) / d[var].std()
        full = smf.logit("nr ~ x * C(arm)", d).fit(disp=0)
        red = smf.logit("nr ~ x + C(arm)", d).fit(disp=0)
        p_int = lrt(full, red)
        for arm, g in d.groupby("arm"):
            f = smf.logit("nr ~ x", g).fit(disp=0)
            lo, hi = np.exp(f.conf_int().loc["x"])
            rows.append(dict(variable=var, arm=arm, n=len(g), n_nonresp=int(g.nr.sum()), or_per_sd=np.exp(f.params.x),
                             ci_lo=lo, ci_hi=hi, p=f.pvalues.x, p_interaction=p_int))
    R = pd.DataFrame(rows)
    comp_int = R[R.variable != "gli"].drop_duplicates("variable")[["variable", "p_interaction"]]
    comp_int["fdr_interaction"] = multipletests(comp_int.p_interaction, method="fdr_bh")[1]
    R = R.merge(comp_int[["variable", "fdr_interaction"]], on="variable", how="left")
    write_table(R, "strap_drug_specific", script=__file__, datasets=["E-MTAB-13733", "STRAP Source Data"],
                note="exploratory; confident matches only; LRT for x-by-arm interaction")
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
