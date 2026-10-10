"""00_register_clinical.py — archive cohort clinical annotations + published per-patient tables into DuckDB.

Tables written:
  clin_r4ra        E-MTAB-11611 SDRF, one row per sample (baseline + week 16), CDAI response (drug arm NOT public)
  clin_strap       E-MTAB-13733 SDRF, one row per sample (acr20 column in the deposit is unreliable: 5/208 responders)
  clin_peac        E-MTAB-6141 SDRF, one row per sample
  strap_published_patient_scores   STRAP Nat Commun 2025 Source Data 'Figure 3a': treatment arm, ACR20,
                                   18 AMP-phase-1 cell-subset module scores (Patient_N ids, not linked to STRAPPAT ids)
  strap_published_polar            STRAP Source Data 'Supp. Fig. 3a': subset x drug association (all ns)
  clin_xenium (--xenium)           Zenodo 15058262 sample metadata: pathotype, DAS28, 6-month remission
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import write_table, register_raw  # noqa: E402

META = ROOT / "raw" / "meta"
STRAP_SD = ROOT / "refs" / "strap_natcommun2025_supp" / "41467_2025_60987_MOESM10_ESM.xlsx"
SDRF_URL = "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/{0}/{1}/Files/{1}.sdrf.txt"
STRAP_SD_URL = ("https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2F"
                "s41467-025-60987-9/MediaObjects/41467_2025_60987_MOESM10_ESM.xlsx")


def tidy_sdrf(acc: str) -> pd.DataFrame:
    p = META / f"{acc}.sdrf.txt"
    register_raw(p, acc, SDRF_URL.format(acc[-3:], acc))
    d = pd.read_csv(p, sep="\t")
    keep = [c for c in d.columns if c.startswith(("Characteristics[", "Factor Value[")) or c in
            ("Source Name", "Comment[ENA_RUN]", "Comment[ENA_SAMPLE]")]
    d = d[keep].drop_duplicates("Comment[ENA_RUN]")
    d.columns = [c.split("[", 1)[-1].rstrip("]").replace(" ", "_").replace(".", "_").lower()
                 if "[" in c else c.replace(" ", "_").lower() for c in d.columns]
    d = d.loc[:, ~d.columns.duplicated()]
    return d.reset_index(drop=True)


def main():
    for acc, name in [("E-MTAB-11611", "clin_r4ra"), ("E-MTAB-13733", "clin_strap"), ("E-MTAB-6141", "clin_peac")]:
        d = tidy_sdrf(acc)
        write_table(d, name, script=__file__, datasets=[acc], inputs=[META / f"{acc}.sdrf.txt"],
                    note="ArrayExpress SDRF, one row per sequencing run")
        print(name, d.shape)

    register_raw(STRAP_SD, "STRAP_NatCommun2025_SourceData", STRAP_SD_URL)
    s = pd.read_excel(STRAP_SD, sheet_name="Figure 3a").rename(columns={"Unnamed: 0": "patient"})
    write_table(s, "strap_published_patient_scores", script=__file__,
                datasets=["STRAP", "doi:10.1038/s41467-025-60987-9"], inputs=[STRAP_SD],
                note="Source Data Fig 3a: arm + ACR20 + 18 AMP1 module scores; Patient_N not linked to STRAPPAT ids")
    p = pd.read_excel(STRAP_SD, sheet_name="Supp. Fig. 3a").rename(columns={"Unnamed: 0": "subset"})
    write_table(p, "strap_published_polar", script=__file__,
                datasets=["STRAP", "doi:10.1038/s41467-025-60987-9"], inputs=[STRAP_SD],
                note="published 3-drug polar association per AMP1 subset; all p>0.26")
    print("strap tables", s.shape, p.shape)


def register_xenium_meta():
    """Xenium cohort (Zenodo 15058262): 17 pre-treatment synovia, pathotype, DAS28 baseline/6 mo, remission status."""
    p = ROOT / "raw" / "xenium" / "zenodo15058262" / "samples_metadata.xlsx"
    register_raw(p, "Zenodo 15058262 (Xenium RA synovium)", "https://zenodo.org/records/15058262")
    x = pd.read_excel(p)
    x.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in x.columns]
    write_table(x, "clin_xenium", script=__file__, datasets=["Zenodo 15058262"], inputs=[p],
                note="17 patients; response = remission / borderline / non-remission at 6 months")
    print("clin_xenium", x.shape)


if __name__ == "__main__":
    register_xenium_meta() if "--xenium" in sys.argv else main()
