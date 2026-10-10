"""37_strap_match.py — link STRAP ArrayExpress samples (STRAPPAT ids) to the published per-patient table
(Source Data Fig 3a: Patient_N, randomised treatment, ACR20, 18 AMP1 module scores), which carries the drug arm
and response that the ArrayExpress deposit lacks (its acr20 column is unusable).

Method: STRAP computed Seurat module scores of AMP phase-1 subsets (Zhang 2019, Supp Table 4: top-20 markers per
cluster) on bulk RNA-seq. We recompute them on our Salmon quantification (samples as 'cells'; scanpy score_genes =
AddModuleScore logic: marker mean minus expression-bin-matched control genes; log2 CPM+1), z-score each subset in
both tables, and solve the one-to-one assignment minimising Euclidean distance (Hungarian).
Confidence per match: margin = (2nd-best distance - best distance) / best distance, and leave-one-subset-out
stability (fraction of 18 re-assignments with one subset dropped that give the same partner).
Calibration (v2): z-scoring each side separately is scale-fragile when our quantified set is a subset. After the
first assignment, per-subset linear maps (published ~ ours) are fitted on confident pairs and our scores are
re-expressed on the published scale; distances are then computed on the published scale (z by published SD) and
the assignment is repeated until it stops changing (max 5 rounds).
Only matches with stability >= 0.9 and margin >= 0.2 are used downstream; the rest are reported, not used.

DuckDB: strap_sample_match (sample_id, patient, treatment, acr20, distance, margin, stability, used)
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, register_raw, write_table  # noqa: E402

AMP1 = ROOT / "refs" / "amp1_natimmunol2019_supp" / "41590_2019_378_MOESM4_ESM.xlsx"
AMP1_URL = ("https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41590-019-0378-1/"
            "MediaObjects/41590_2019_378_MOESM4_ESM.xlsx")


def amp1_markers() -> dict:
    x = pd.read_excel(AMP1, header=1)
    x["cluster"] = x["cluster"].ffill()
    x["code"] = x.cluster.str.extract(r"\((SC-[A-Z]\d)\)")[0].str.replace("-", "_")
    return {c: g.gene.astype(str).tolist() for c, g in x.dropna(subset=["code"]).groupby("code")}


def module_scores(counts: pd.DataFrame, markers: dict) -> pd.DataFrame:
    cpm = counts / counts.sum() * 1e6
    lg = np.log2(cpm[(cpm > 1).mean(axis=1) >= 0.2] + 1)
    a = ad.AnnData(lg.T.to_numpy(), obs=pd.DataFrame(index=lg.columns), var=pd.DataFrame(index=lg.index))
    for code, genes in markers.items():
        sc.tl.score_genes(a, [g for g in genes if g in a.var_names], score_name=code, ctrl_size=100, n_bins=24,
                          random_state=0)
    return a.obs[list(markers)]


def assign(A: pd.DataFrame, B: pd.DataFrame, cols, calibrated: bool = False):
    """One-to-one assignment of rows of A (ours) to rows of B (published) on the given subset columns.
    calibrated=False: each side z-scored separately; True: A is already on B's scale, both scaled by B's SD."""
    if calibrated:
        sd = B[cols].std()
        za, zb = (A[cols] - B[cols].mean()) / sd, (B[cols] - B[cols].mean()) / sd
    else:
        za = (A[cols] - A[cols].mean()) / A[cols].std()
        zb = (B[cols] - B[cols].mean()) / B[cols].std()
    D = cdist(za.to_numpy(), zb.to_numpy())
    r, c = linear_sum_assignment(D)
    out = np.full(len(A), -1, dtype=int)     # -1 = no partner (more of ours than published patients)
    out[r] = c
    return out, D


def calibrate(ours: pd.DataFrame, pub: pd.DataFrame, idx, keep, cols) -> pd.DataFrame:
    """Per-subset least-squares map of our scores onto the published scale, fitted on confident pairs."""
    keep = keep & (idx >= 0)
    cal = ours.copy()
    for c in cols:
        x, y = ours[c].to_numpy()[keep], pub[c].to_numpy()[idx[keep]]
        b, a = np.polyfit(x, y, 1)
        cal[c] = a + b * ours[c]
    return cal


def confidence(A, B, cols, idx, D, calibrated):
    has = idx >= 0
    best = np.where(has, D[np.arange(len(A)), np.where(has, idx, 0)], np.nan)
    srt = np.sort(D, axis=1)
    margin = np.where(has & np.isclose(best, srt[:, 0]), (srt[:, 1] - best) / best, 0.0)
    stab = np.zeros(len(A))
    for c in cols:                                   # leave-one-subset-out stability
        stab += (assign(A, B, [x for x in cols if x != c], calibrated)[0] == idx) & has
    return best, margin, stab / len(cols)


def main():
    register_raw(AMP1, "Zhang 2019 Nat Immunol Supp Table 4 (AMP1 cluster markers)", AMP1_URL)
    markers = amp1_markers()
    con = read('SELECT * FROM "strap_gene_counts"')
    counts = con[con.gene_type == "protein_coding"].drop(columns=["gene_id", "gene_type"]).groupby("gene_symbol").sum()
    counts = counts[[c for c in counts.columns if c.endswith("baseline")]]
    ours = module_scores(counts, markers)
    pub = read("strap_published_patient_scores").set_index("patient")
    cols = [c for c in markers if c in pub.columns]
    if len(ours) != len(pub):
        print(f"note: {len(ours)} quantified baseline samples vs {len(pub)} published patients (unequal sets; "
              "assignment is one-to-one on the smaller side)")
    idx, D = assign(ours, pub, cols)
    best, margin, stab = confidence(ours, pub, cols, idx, D, False)
    scaled, calibrated = ours, False
    for rnd in range(5):
        keep = (stab >= 0.9) & (margin >= 0.2)
        if keep.sum() < 20:
            break
        scaled, calibrated = calibrate(ours, pub, idx, keep, cols), True
        new_idx, D = assign(scaled, pub, cols, True)
        changed = int((new_idx != idx).sum())
        idx = new_idx
        best, margin, stab = confidence(scaled, pub, cols, idx, D, True)
        print(f"calibration round {rnd + 1}: {changed} assignments changed; "
              f"confident {int(((stab >= 0.9) & (margin >= 0.2)).sum())}/{len(ours)}", flush=True)
        if changed == 0:
            break
    za = (ours[cols] - ours[cols].mean()) / ours[cols].std()
    zb = (pub[cols] - pub[cols].mean()) / pub[cols].std()
    r = np.array([np.corrcoef(za.iloc[i], zb.iloc[idx[i]])[0, 1] if idx[i] >= 0 else np.nan
                  for i in range(len(ours))])
    M = pd.DataFrame({"sample_id": ours.index, "patient": [pub.index[j] if j >= 0 else None for j in idx],
                      "distance": best, "margin": margin, "stability": stab, "profile_r": r})
    M = M.merge(pub[["Treatment", "ACR20"]].rename(columns={"Treatment": "treatment", "ACR20": "acr20"}),
                left_on="patient", right_index=True, how="left")
    M["used"] = (M.stability >= 0.9) & (M.margin >= 0.2)
    write_table(M, "strap_sample_match", script=__file__, datasets=["E-MTAB-13733", "STRAP Source Data Fig 3a",
                                                                     "Zhang 2019 Supp Table 4"],
                note="Hungarian matching on z-scored AMP1 module scores; used = stability>=0.9 & margin>=0.2")
    write_table(ours.reset_index(names="sample_id"), "strap_amp1_scores_recomputed", script=__file__,
                datasets=["E-MTAB-13733"])
    print(M[["distance", "margin", "stability", "profile_r"]].describe().round(3).to_string())
    print(f"confident matches: {M.used.sum()} / {len(M)}; arms among them: "
          f"{M[M.used].treatment.value_counts().to_dict()}")
    # sanity check: recomputed vs published score correlation per subset over confident matches
    good = M[M.used]
    per = {c: np.corrcoef(ours.loc[good.sample_id, c], pub.loc[good.patient, c])[0, 1] for c in cols}
    print("per-subset r (ours vs published, confident matches):", {k: round(v, 2) for k, v in per.items()})


if __name__ == "__main__":
    main()
