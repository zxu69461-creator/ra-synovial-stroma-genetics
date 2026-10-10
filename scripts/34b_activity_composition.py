"""34b_activity_composition.py — is the inverse stromal–activity association more than dilution by immune infiltrate?

Compartment scores are compositional, so a stromal score can fall simply because immune
cells expand. Per cohort (baseline biopsies, the same compartment means as 34_disease_activity.py), the stromal score
is related to each activity measure by partial Spearman correlation adjusting for the total immune score (mean of the
T/NK, B/plasma and myeloid compartment scores): ranks of both variables are residualised on the rank of the immune
score and correlated. Pooled across cohorts by Fisher-z DerSimonian-Laird (n - 1 for the extra covariate).
The lymphocyte (T/NK) score is analysed the same way, adjusted for the stromal score.
DuckDB: activity_partial (cohort-level and pooled)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_table  # noqa: E402

spec = importlib.util.spec_from_file_location("s34", ROOT / "scripts/34_disease_activity.py")
s34 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s34)
CLIN = ["das28", "crp", "esr", "sjc", "cdai"]
IMM = ["T/NK", "B/plasma", "Myeloid"]


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    X = np.column_stack([np.ones_like(rz), rz])
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 3) / (1 - r * r))
    return r, 2 * stats.t.sf(abs(t), n - 3)


def main():
    S, M, H = read("bulk_state_scores"), read("bulk_samples"), read("clin_activity_harmonised")
    grp = read("SELECT DISTINCT state, state_group FROM atlas_cells_state")
    grp = grp.assign(state_group=grp.state_group.astype(str)).set_index("state").state_group
    rows = []
    for coh in ["R4RA", "STRAP", "PEAC", "GSE198520", "GSE45867"]:
        m = M[M.cohort == coh].set_index("sample")
        W = S[S.cohort == coh].pivot(index="sample", columns="state", values="score").loc[m.index]
        Z = (W - W.mean()) / W.std()
        cm = Z.T.groupby(Z.columns.map(grp)).mean().T
        cm["immune"] = cm[IMM].mean(axis=1)
        base = (m.timepoint == "baseline").to_numpy()
        cl = H[H.cohort == coh].set_index("sample").reindex(Z.index)
        for var, adj in [("Stromal", "immune"), ("T/NK", "Stromal")]:
            for k in CLIN:
                ok = base & cl[k].notna().to_numpy()
                if ok.sum() < 15:
                    continue
                r, p = partial_spearman(cm.loc[ok, var].to_numpy(), cl.loc[ok, k].to_numpy(), cm.loc[ok, adj].to_numpy())
                rows.append(dict(cohort=coh, variable=var, adjusted_for=adj, clinical=k, n=int(ok.sum()), rho=r, p=p))
    C = pd.DataFrame(rows)
    pooled = []
    for (v, a, k), g in C.groupby(["variable", "adjusted_for", "clinical"]):
        r, lo, hi, p, i2, z = s34.dl_fisher(g.rho.to_numpy(), g.n.to_numpy() - 1)
        pooled.append(dict(cohort="pooled", variable=v, adjusted_for=a, clinical=k, n=int(g.n.sum()), rho=r,
                           ci_lo=lo, ci_hi=hi, p=p, i2=i2, n_cohorts=len(g)))
    R = pd.concat([C, pd.DataFrame(pooled)], ignore_index=True)
    write_table(R, "activity_partial", script=__file__, datasets=["E-MTAB-11611", "E-MTAB-13733", "E-MTAB-6141",
                                                                   "GSE198520", "GSE45867"],
                note="partial Spearman of compartment score vs activity, stromal adjusted for total immune score and "
                     "T/NK adjusted for stromal score; baseline biopsies; pooled Fisher-z DL")
    pd.set_option("display.width", 200)
    print(R[R.cohort == "pooled"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
