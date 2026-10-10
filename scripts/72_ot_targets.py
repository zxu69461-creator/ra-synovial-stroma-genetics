"""72_ot_targets.py — Open Targets RA target evidence (genetic vs clinical) mapped onto synovial states (Fig 7).

Open Targets Platform GraphQL, disease MONDO_0008383 (rheumatoid arthritis): every associated target with its
datatype scores (genetic_association, clinical, literature, ...). "Genetically prioritised" = genetic_association
>= 0.5; "clinically pursued" = clinical >= 0.5 (drugs in trials/approved for RA).

State mapping: target specific to a state when spec_z >= 2 and detected >= 10% (state_pseudobulk_long).
Tests: (i) per state, n genetic targets vs scDRS genetic load (Spearman, 54 states);
       (ii) targets specific to stromal vs immune states: genetic_association score (Mann-Whitney);
       (iii) share of clinically pursued targets that lack genetic support, by compartment.
DuckDB: ot_ra_targets, ot_target_state_summary
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

API = "https://api.platform.opentargets.org/api/v4/graphql"
Q = """query($i:Int!,$n:Int!){ disease(efoId:"MONDO_0008383"){ associatedTargets(page:{index:$i,size:$n}){
  count rows{ score target{ id approvedSymbol } datatypeScores{ id score } } } } }"""
IMMUNE = {"T/NK", "B/plasma", "Myeloid"}


def fetch() -> pd.DataFrame:
    rows, i, n = [], 0, 1000
    while True:
        r = requests.post(API, json={"query": Q, "variables": {"i": i, "n": n}}, timeout=120)
        r.raise_for_status()
        at = r.json()["data"]["disease"]["associatedTargets"]
        for x in at["rows"]:
            d = {"gene": x["target"]["approvedSymbol"], "ensembl_id": x["target"]["id"], "overall": x["score"]}
            d.update({s["id"]: s["score"] for s in x["datatypeScores"]})
            rows.append(d)
        i += 1
        if i * n >= at["count"]:
            break
    return pd.DataFrame(rows).fillna(0.0)


def main():
    ot = fetch()
    write_table(ot, "ot_ra_targets", script=__file__, datasets=["Open Targets Platform (live API)", "MONDO_0008383"],
                note="all RA-associated targets with datatype scores")
    ot["genetic"] = ot.genetic_association >= 0.5
    ot["clinical_pursued"] = ot.get("clinical", 0) >= 0.5
    pb = read("state_pseudobulk_long")
    spec = pb[(pb.spec_z >= 2) & (pb.pct_detected >= 0.10)].merge(ot, on="gene")

    load = read("scdrs_state_load").query("trait == 'ra_eur_noMHC'")[["state", "state_group", "assoc_mcz"]]
    load["n_genetic_targets"] = load.state.map(spec[spec.genetic].groupby("state").gene.nunique()).fillna(0)
    load["n_clinical_targets"] = load.state.map(spec[spec.clinical_pursued].groupby("state").gene.nunique()).fillna(0)
    load["n_specific_genes"] = load.state.map(pb[(pb.spec_z >= 2) & (pb.pct_detected >= 0.10)]
                                              .groupby("state").gene.nunique()).fillna(0)
    load["frac_genetic"] = load.n_genetic_targets / load.n_specific_genes.clip(lower=1)
    rho, p = stats.spearmanr(load.assoc_mcz, load.frac_genetic)
    load["rho_load_vs_frac_genetic"], load["p"] = rho, p
    write_table(load, "ot_target_state_summary", script=__file__, datasets=["Open Targets", "E-MTAB-11791"],
                note="state-specific genes (spec_z>=2, >=10% detected) that are OT genetic / clinical RA targets")

    tgt = spec.assign(immune=spec.state_group.isin(IMMUNE))
    per_gene = tgt.groupby("gene").agg(immune=("immune", "max"), stromal=("state_group", lambda s: (s == "Stromal").any()),
                                       ga=("genetic_association", "first"), clin=("clinical_pursued", "first"))
    imm, stro = per_gene[per_gene.immune & ~per_gene.stromal], per_gene[per_gene.stromal & ~per_gene.immune]
    u, p_mw = stats.mannwhitneyu(imm.ga, stro.ga, alternative="greater")
    pd.set_option("display.width", 200)
    print(f"{len(ot)} OT targets; genetic>=0.5: {ot.genetic.sum()}; clinical>=0.5: {ot.clinical_pursued.sum()}")
    print(f"state-level: genetic load vs fraction of specific genes that are genetic targets: rho={rho:.2f} p={p:.2g}")
    print(f"immune-specific genes (n={len(imm)}) genetic score mean {imm.ga.mean():.3f}, "
          f"frac>=0.5 {(imm.ga >= .5).mean():.1%}; stromal-specific (n={len(stro)}) mean {stro.ga.mean():.3f}, "
          f"frac>=0.5 {(stro.ga >= .5).mean():.1%}; MW p={p_mw:.2g}")
    for name, d in [("immune", imm), ("stromal", stro)]:
        c = d[d.clin]
        print(f"{name}-specific clinically pursued targets: {len(c)}; lacking genetic support: "
              f"{(c.ga < 0.5).sum()} ({', '.join(c[c.ga < .5].index[:12])})")
    print(load.groupby("state_group")[["frac_genetic", "n_genetic_targets", "n_clinical_targets"]].mean().round(3))


if __name__ == "__main__":
    main()
