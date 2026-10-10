"""73_target_revision.py — target-landscape analyses: approved targets, clinical and genetic shares, and their sensitivity.

Gene-to-compartment assignment (discovery-atlas state pseudobulks, state_pseudobulk_long):
  'exclusive'  : specific (spec_z >= 2, detected >= 10%) in >= 1 state of the compartment and in no state of the
                 other compartment class (immune = T/NK, B/plasma, myeloid vs stromal) — the definition of Fig 6b/c;
  'exclusive4' : the same with spec_z >= 4 (stricter specificity);
  'maxstate'   : every expressed gene assigned to the compartment of its single most specific state.
Tests:
  (1) genetic support (Open Targets genetic_association >= 0.1 / 0.3 / 0.5): immune- vs stromal-specific, Fisher;
  (2) clinically pursued (Open Targets clinical >= 0.5), normalised to the gene denominators, Fisher;
  (3) approved biologic / targeted-synthetic DMARD targets for RA from ChEMBL (max phase 4 for RA; mechanisms
      TNF, IL6R, MS4A1, CD80/CD86, IL1R1, JAK1/2/3, TYK2 — abatacept's targets are CD80/CD86, not CTLA4): compartment
      of the most specific state, binomial test against the base rate of 'maxstate' immune assignment;
  (4) stroma-specific genes with any genetic evidence (genetic_association >= 0.1), for the supplement.
Also records the Open Targets and ChEMBL releases currently served (the data were pulled on 2026-09-24).
DuckDB: target_tests, approved_target_compartment, stromal_genetic_candidates, target_sources
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_table  # noqa: E402

IMMUNE = {"T/NK", "B/plasma", "Myeloid"}
BTS = {"TNF-alpha inhibitor": "TNF", "Interleukin-6 receptor alpha subunit inhibitor": "IL6R",
       "Interleukin-6 receptor alpha subunit antagonist": "IL6R", "B-lymphocyte antigen CD20 binding agent": "MS4A1",
       "T-lymphocyte activation antigen CD80 inhibitor": "CD80", "T-lymphocyte activation antigen CD86 inhibitor": "CD86",
       "Interleukin-1 receptor antagonist": "IL1R1", "Janus Kinase (JAK) inhibitor": None,
       "Tyrosine-protein kinase JAK1 inhibitor": "JAK1", "Tyrosine-protein kinase JAK2 inhibitor": "JAK2",
       "Tyrosine-protein kinase JAK3 inhibitor": "JAK3", "Tyrosine-protein kinase TYK2 inhibitor": "TYK2"}


def assign(pb):
    pb = pb.assign(cls=np.where(pb.state_group.isin(IMMUNE), "immune",
                                np.where(pb.state_group == "Stromal", "stromal", "other")))
    out = {}
    for name, zmin in (("exclusive", 2), ("exclusive4", 4)):
        s = pb[(pb.spec_z >= zmin) & (pb.pct_detected >= 0.10)]
        g = s.groupby("gene").cls.agg(lambda x: set(x))
        out[name] = pd.Series({k: ("immune" if "immune" in v and "stromal" not in v else
                                   "stromal" if "stromal" in v and "immune" not in v else "both/other")
                               for k, v in g.items()})
    expressed = pb[pb.pct_detected >= 0.10].gene.unique()
    top = pb[pb.gene.isin(expressed)].sort_values("spec_z", ascending=False).drop_duplicates("gene").set_index("gene")
    out["maxstate"] = top.cls.where(top.spec_z >= 2, "unspecific")
    return out, top


def fisher(a_hit, a_n, b_hit, b_n):
    odds, p = stats.fisher_exact([[a_hit, a_n - a_hit], [b_hit, b_n - b_hit]])
    return odds, p


def sources():
    rows = []
    try:
        q = "query { meta { dataVersion { year month iteration } apiVersion { x y z } } }"
        m = requests.post("https://api.platform.opentargets.org/api/v4/graphql", json={"query": q}, timeout=60).json()
        dv = m["data"]["meta"]["dataVersion"]
        rows.append(dict(source="Open Targets Platform", release=f"{dv['year']}.{dv['month']}"
                         + (f".{dv.get('iteration')}" if dv.get("iteration") else ""),
                         note="release served on 2026-10-07; RA associations were pulled on 2026-09-24"))
    except Exception as e:                                  # noqa: BLE001
        rows.append(dict(source="Open Targets Platform", release="unknown", note=f"meta query failed: {e}"))
    try:
        st = requests.get("https://www.ebi.ac.uk/chembl/api/data/status.json", timeout=60).json()
        rows.append(dict(source="ChEMBL", release=str(st.get("chembl_db_version")),
                         note="release served on 2026-10-07; RA drug mechanisms were pulled on 2026-09-24"))
    except Exception as e:                                  # noqa: BLE001
        rows.append(dict(source="ChEMBL", release="unknown", note=f"status query failed: {e}"))
    return pd.DataFrame(rows)


def main():
    pb = read("state_pseudobulk_long")
    pb["state_group"] = pb.state_group.astype(str)
    ot = read("ot_ra_targets").groupby("gene")[["genetic_association", "clinical"]].max()   # symbols with >1 Ensembl id
    ga = ot.genetic_association.reindex(pb.gene.unique()).fillna(0.0)
    cl = ot.clinical.reindex(pb.gene.unique()).fillna(0.0)
    A, top = assign(pb)
    rows = []
    for name, lab in A.items():
        imm, stro = lab[lab == "immune"].index, lab[lab == "stromal"].index
        for thr in (0.1, 0.3, 0.5):
            a, b = int((ga[imm] >= thr).sum()), int((ga[stro] >= thr).sum())
            o, p = fisher(a, len(imm), b, len(stro))
            rows.append(dict(assignment=name, evidence="genetic", threshold=thr, immune_hit=a, immune_n=len(imm),
                             stromal_hit=b, stromal_n=len(stro), odds_ratio=o, p=p))
        a, b = int((cl[imm] >= 0.5).sum()), int((cl[stro] >= 0.5).sum())
        o, p = fisher(a, len(imm), b, len(stro))
        rows.append(dict(assignment=name, evidence="clinical", threshold=0.5, immune_hit=a, immune_n=len(imm),
                         stromal_hit=b, stromal_n=len(stro), odds_ratio=o, p=p))
    T = pd.DataFrame(rows)
    # approved b/ts-DMARD targets (ChEMBL phase 4 for RA)
    dt = read("ra_drug_targets")
    dt = dt[(dt.max_phase_ra >= 4) & dt.mechanism.isin(BTS)]
    genes = sorted(set(g for g in dt.gene.dropna() if g in set(BTS.values())))
    base = float((top.cls[top.spec_z >= 2] == "immune").mean())
    ap = top.loc[[g for g in genes if g in top.index], ["state", "state_group", "spec_z", "cls"]].reset_index()
    ap["drugs"] = ap.gene.map(dt.groupby("gene").drug_name.agg(lambda x: ", ".join(sorted(set(map(str, x))))))
    k = int((ap.cls == "immune").sum())
    bt = stats.binomtest(k, len(ap), base, alternative="greater")
    T = pd.concat([T, pd.DataFrame([dict(assignment="maxstate", evidence="approved b/tsDMARD targets (ChEMBL phase 4)",
                                         threshold=np.nan, immune_hit=k, immune_n=len(ap), stromal_hit=int((ap.cls == "stromal").sum()),
                                         stromal_n=len(ap), odds_ratio=np.nan, p=bt.pvalue, base_rate=base)])],
                  ignore_index=True)
    # stroma-specific genes with any genetic evidence
    stro = A["exclusive"][A["exclusive"] == "stromal"].index
    cand = pd.DataFrame({"gene": stro, "genetic_association": ga[stro].values, "clinical": cl[stro].values})
    cand = cand[cand.genetic_association >= 0.1].merge(top[["state", "spec_z"]], left_on="gene", right_index=True)
    cand = cand.sort_values("genetic_association", ascending=False)
    write_table(T, "target_tests", script=__file__, datasets=["Open Targets (2026-09-24)", "ChEMBL (2026-09-24)", "E-MTAB-11791"],
                note="genetic (OT >= 0.1/0.3/0.5) and clinical (>= 0.5) target shares, immune- vs stromal-specific genes, "
                     "three assignments; approved b/tsDMARD targets vs base rate")
    write_table(ap, "approved_target_compartment", script=__file__, datasets=["ChEMBL"],
                note="ChEMBL phase-4 RA biologic/targeted-synthetic DMARD targets and their most specific state")
    write_table(cand, "stromal_genetic_candidates", script=__file__, datasets=["Open Targets", "E-MTAB-11791"],
                note="stroma-specific genes (exclusive) with OT genetic association >= 0.1")
    write_table(sources(), "target_sources", script=__file__, datasets=["Open Targets", "ChEMBL"],
                note="data releases currently served; RA data pulled 2026-09-24")
    pd.set_option("display.width", 220)
    print(T.round(4).to_string(index=False))
    print(f"base rate (maxstate immune among specific genes) = {base:.3f}")
    print(ap.to_string(index=False))
    print(cand.head(25).to_string(index=False))
    print(read("target_sources").to_string(index=False))


if __name__ == "__main__":
    main()
