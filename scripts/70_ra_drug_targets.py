"""70_ra_drug_targets.py — targets of drugs tested/approved for RA (ChEMBL), for the Fig 7 target-landscape axis.

ChEMBL REST: drug_indication (efo_id EFO_0000685 rheumatoid arthritis, incl. child terms via efo_term match)
-> molecule mechanisms -> target components -> gene symbols. Keeps the max clinical phase for RA per drug.
Parent/child salt forms are collapsed to the parent molecule.

DuckDB: ra_drug_targets (drug_chembl_id, drug_name, max_phase_ra, mechanism, action_type, target_chembl_id, gene)
"""
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import write_table  # noqa: E402

API = "https://www.ebi.ac.uk/chembl/api/data"


def pages(endpoint, **params):
    params = {"format": "json", "limit": 1000, **params}
    url = f"{API}/{endpoint}.json"
    while url:
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        j = r.json()
        key = [k for k in j if k != "page_meta"][0]
        yield from j[key]
        nxt = j["page_meta"]["next"]
        url, params = (f"https://www.ebi.ac.uk{nxt}", None) if nxt else (None, None)


def main():
    ind = pd.DataFrame(pages("drug_indication", efo_id="EFO:0000685"))
    ind = ind.groupby("molecule_chembl_id").max_phase_for_ind.max().astype(float).rename("max_phase_ra").reset_index()
    print(f"{len(ind)} molecules with an RA indication", flush=True)

    rows = []
    ids = ind.molecule_chembl_id.tolist()
    for i in range(0, len(ids), 50):
        for m in pages("mechanism", molecule_chembl_id__in=",".join(ids[i:i + 50])):
            rows.append(dict(drug_chembl_id=m["molecule_chembl_id"], parent=m.get("parent_molecule_chembl_id"),
                             mechanism=m["mechanism_of_action"], action_type=m["action_type"],
                             target_chembl_id=m["target_chembl_id"]))
    mech = pd.DataFrame(rows).dropna(subset=["target_chembl_id"])

    tg = []
    tids = mech.target_chembl_id.unique().tolist()
    for i in range(0, len(tids), 50):
        for t in pages("target", target_chembl_id__in=",".join(tids[i:i + 50])):
            for comp in t.get("target_components", []):
                syms = [s["component_synonym"] for s in comp.get("target_component_synonyms", [])
                        if s["syn_type"] == "GENE_SYMBOL"]
                if syms and t.get("organism") == "Homo sapiens":
                    tg.append(dict(target_chembl_id=t["target_chembl_id"], target_type=t["target_type"],
                                   gene=syms[0]))
    tg = pd.DataFrame(tg).drop_duplicates()

    names = {}
    mids = mech.drug_chembl_id.unique().tolist()
    for i in range(0, len(mids), 50):
        for m in pages("molecule", molecule_chembl_id__in=",".join(mids[i:i + 50])):
            names[m["molecule_chembl_id"]] = m.get("pref_name")
    out = mech.merge(ind.rename(columns={"molecule_chembl_id": "drug_chembl_id"}), on="drug_chembl_id")
    out = out.merge(tg, on="target_chembl_id", how="inner")
    out["drug_name"] = out.drug_chembl_id.map(names)
    out = out.drop_duplicates(["drug_chembl_id", "gene"])
    write_table(out, "ra_drug_targets", script=__file__, datasets=["ChEMBL (live API)", "EFO_0000685"],
                note="drugs with an RA indication (any phase) -> mechanism targets -> human gene symbols")
    pd.set_option("display.width", 200)
    s = out.groupby("gene").agg(n_drugs=("drug_chembl_id", "nunique"), max_phase=("max_phase_ra", "max"))
    print(s.sort_values(["max_phase", "n_drugs"], ascending=False).head(40).to_string())
    print(f"{out.drug_chembl_id.nunique()} drugs, {out.gene.nunique()} target genes "
          f"({(s.max_phase >= 4).sum()} with an approved RA drug)")


if __name__ == "__main__":
    main()
