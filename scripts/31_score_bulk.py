"""31_score_bulk.py — score every bulk synovial sample for every atlas state signature (Fig 3).

Score = mean z (across the cohort's samples) of log2(CPM+1) over the state's signature genes present in the cohort
(>= 5 genes required). z-scoring is within cohort, so scores are comparable across samples of one cohort only;
cross-cohort synthesis is done on effect sizes (32_response_association.py), never on pooled scores.

Usage: python scripts/31_score_bulk.py GSE198520 [R4RA ...]
DuckDB: bulk_state_scores (cohort, sample, state, score, n_genes_used), bulk_samples (sample metadata)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import connect, read, write_table  # noqa: E402
from cohorts import LOADERS, LOG_SCALE  # noqa: E402

MIN_GENES = 5


def score(counts: pd.DataFrame, sig: pd.DataFrame, log_scale: bool = False):
    if log_scale:                                                 # microarray: already log2 (possibly centred)
        lg = counts[counts.std(axis=1) > 0]
    else:
        keep = counts.sum(axis=1) > 0
        cpm = counts[keep] / counts[keep].sum() * 1e6
        lg = np.log2(cpm + 1)
        lg = lg[(cpm > 1).mean(axis=1) >= 0.2]                    # expressed in >= 20% of samples
    z = lg.sub(lg.mean(axis=1), axis=0).div(lg.std(axis=1) + 1e-6, axis=0)
    out = []
    for st, g in sig.groupby("state"):
        genes = [x for x in g.gene if x in z.index]
        if len(genes) >= MIN_GENES:
            out.append(pd.DataFrame({"sample": z.columns, "state": st, "score": z.loc[genes].mean().values,
                                     "n_genes_used": len(genes)}))
    return pd.concat(out, ignore_index=True)


def main():
    sig = read("state_signatures")
    usable = read("state_signature_summary").query("usable").state
    sig = sig[sig.state.isin(usable)]
    all_scores, all_meta = [], []
    for name in sys.argv[1:]:
        counts, meta = LOADERS[name]()
        s = score(counts, sig, log_scale=name in LOG_SCALE)
        s.insert(0, "cohort", name)
        all_scores.append(s)
        all_meta.append(meta.reset_index(names="sample"))
        print(f"{name}: {counts.shape[1]} samples, {s.state.nunique()} states scored "
              f"(median genes/state {int(s.groupby('state').n_genes_used.first().median())})", flush=True)
    S, M = pd.concat(all_scores, ignore_index=True), pd.concat(all_meta, ignore_index=True)
    # replace only the cohorts scored in this run, keep others already archived
    con = connect()
    try:
        tables = set(con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()
    if "bulk_state_scores" in tables:
        old_s, old_m = read("bulk_state_scores"), read("bulk_samples")
        S = pd.concat([old_s[~old_s.cohort.isin(S.cohort.unique())], S], ignore_index=True)
        M = pd.concat([old_m[~old_m.cohort.isin(M.cohort.unique())], M], ignore_index=True)
    write_table(S, "bulk_state_scores", script=__file__, datasets=sorted(S.cohort.unique()),
                note="mean within-cohort z of log2CPM over state signature genes (>=5 genes)")
    write_table(M, "bulk_samples", script=__file__, datasets=sorted(M.cohort.unique()))


if __name__ == "__main__":
    main()
