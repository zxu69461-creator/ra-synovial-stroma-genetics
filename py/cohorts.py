"""cohorts.py — uniform loaders for bulk synovial cohorts: counts (genes x samples, gene symbols) + sample metadata.

Every loader returns (counts: DataFrame[gene_symbol x sample], meta: DataFrame indexed by sample) with meta columns
    cohort, patient, timepoint ('baseline'|'post'), treatment, response_binary (1 = responder, 0 = non-responder,
    NaN = not applicable), response_detail, pathotype
so downstream scoring / association code is cohort-agnostic.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "raw" / "geo"


def _geo_characteristics(series_matrix: Path) -> pd.DataFrame:
    rows, titles, gsm = [], None, None
    with gzip.open(series_matrix, "rt") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                gsm = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                rows.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
    d = {}
    for r in rows:
        key = r[0].split(":", 1)[0].strip() if r and ":" in r[0] else f"char{len(d)}"
        d[key] = [x.split(":", 1)[1].strip() if ":" in x else x for x in r]
    return pd.DataFrame(d, index=titles).assign(gsm=gsm)


def gse198520():
    """Anti-TNF (certolizumab/etanercept/...) synovium, pre and 12 weeks post; EULAR status. ~1M reads/sample."""
    c = pd.read_csv(GEO / "GSE198520_Raw_gene_count_matrix.txt.gz", sep="\t", index_col=0)
    c.columns = c.columns.str.replace("_bx$", "", regex=True)
    c = c.groupby(level=0).sum()
    ch = _geo_characteristics(GEO / "GSE198520_series_matrix.txt.gz")
    m = pd.DataFrame(index=ch.index)
    m["cohort"] = "GSE198520"
    m["patient"] = ch.index.str.replace(r"_(pre|post)$", "", regex=True)
    m["timepoint"] = ch["timepoint"].str.lower().map({"pre": "baseline", "post": "post"})
    m["treatment"] = ch["biologic"]
    m["response_detail"] = ch["status"]
    m["response_binary"] = ch["status"].map(lambda s: 1.0 if s.lower() == "responder" else
                                            (0.0 if "non" in s.lower() else np.nan))
    m["pathotype"] = ch["pathotype"]
    m["rin"] = pd.to_numeric(ch["rin"], errors="coerce")
    return c[m.index.intersection(c.columns)], m.loc[m.index.intersection(c.columns)]


def r4ra():
    """R4RA (E-MTAB-11611): Salmon re-quantified; CDAI50 response (drug arm not public)."""
    import duckdb
    con = duckdb.connect(str(ROOT / "db" / "ra_rx.duckdb"), read_only=True)
    c = con.execute('SELECT * FROM "r4ra_gene_counts"').df()
    clin = con.execute('SELECT * FROM "clin_r4ra"').df()
    con.close()
    c = c[c.gene_type == "protein_coding"].drop(columns=["gene_id", "gene_type"]).groupby("gene_symbol").sum()
    clin = clin.set_index("sample_id")
    m = pd.DataFrame(index=clin.index)
    m["cohort"] = "R4RA"
    m["patient"] = clin["individual"]
    m["timepoint"] = clin["visit"].astype(int).map({0: "baseline", 16: "post"})
    m["treatment"] = "rituximab_or_tocilizumab"
    m["response_detail"] = clin["cdai_response_status_v7"]
    m["response_binary"] = (clin["cdai_response_status_v7"] == "Responder").astype(float)
    m["pathotype"] = clin["pathotype"]
    keep = m.index.intersection(c.columns)
    return c[keep], m.loc[keep]


def _gpl570_matrix(series_matrix: Path) -> pd.DataFrame:
    """RMA log2 series matrix -> gene symbols (probe with the highest mean expression per gene)."""
    x = pd.read_csv(series_matrix, sep="\t", comment="!", index_col=0).dropna(how="all")
    ann = pd.read_csv(GEO / "GPL570.annot.gz", sep="\t", skiprows=27, usecols=["ID", "Gene symbol"],
                      dtype=str).dropna()
    ann = ann[~ann["Gene symbol"].str.contains("///")].set_index("ID")["Gene symbol"]
    x = x.loc[x.index.intersection(ann.index)]
    if x.max().max() > 100:                     # MAS5 / linear-scale deposits -> log2
        x = np.log2(x.clip(lower=1))
    x["gene"] = ann.reindex(x.index).values
    x["_m"] = x.drop(columns="gene").mean(axis=1)
    return x.sort_values("_m", ascending=False).drop_duplicates("gene").set_index("gene").drop(columns="_m")


def _eular(das0: float, das1: float) -> str:
    """EULAR response from DAS28 before/after (van Gestel 1996)."""
    d = das0 - das1
    if d > 1.2:
        return "good" if das1 <= 3.2 else "moderate"
    if d > 0.6:
        return "none" if das1 > 5.1 else "moderate"
    return "none"


def gse24742():
    """Rituximab, synovium T0/T12, EULAR good/moderate/non (Affymetrix GPL570, RMA log2)."""
    x = _gpl570_matrix(GEO / "GSE24742_series_matrix.txt.gz")
    ch = _geo_characteristics(GEO / "GSE24742_series_matrix.txt.gz")
    x.columns = ch.index[[list(ch.gsm).index(c) for c in x.columns]]
    m = pd.DataFrame(index=ch.index)
    m["cohort"], m["treatment"] = "GSE24742", "rituximab"
    m["patient"] = ch.index.str.split().str[0]
    m["timepoint"] = ch["treatment"].map(lambda t: "baseline" if "baseline" in t else "post")
    m["response_detail"] = ch["response"].str.replace("EULAR ", "")
    m["response_binary"] = m.response_detail.str.lower().map(
        lambda r: 1.0 if r.startswith("good") else (0.0 if r.startswith(("non", "poor")) else np.nan))
    m["pathotype"] = np.nan
    return x[m.index], m


def gse45867():
    """Tocilizumab (n=12) vs MTX (n=8) in early RA, synovium pre/post 12 wk; EULAR derived from DAS28-CRP."""
    x = _gpl570_matrix(GEO / "GSE45867_series_matrix.txt.gz")
    ch = _geo_characteristics(GEO / "GSE45867_series_matrix.txt.gz")
    x.columns = ch.index[[list(ch.gsm).index(c) for c in x.columns]]
    m = pd.DataFrame(index=ch.index)
    m["cohort"] = "GSE45867"
    m["treatment"] = np.where(ch.index.str.contains("Tocilizumab"), "tocilizumab", "methotrexate")
    m["timepoint"] = np.where(ch.index.str.contains("prior"), "baseline", "post")
    m["patient"] = m.treatment + "_" + ch.index.str.extract(r"repl (\d+)", expand=False).values
    das = pd.to_numeric(ch["disease activity score (das)-28crp"], errors="coerce")
    wide = pd.DataFrame({"das": das.values, "patient": m.patient.values, "tp": m.timepoint.values}).pivot(
        index="patient", columns="tp", values="das")
    resp = {p: _eular(r.baseline, r.post) for p, r in wide.iterrows() if r.notna().all()}
    m["response_detail"] = m.patient.map(resp)
    m["response_binary"] = m.response_detail.map({"good": 1.0, "moderate": np.nan, "none": 0.0})
    m["pathotype"] = np.nan
    return x[m.index], m


def strap():
    """STRAP (E-MTAB-13733): Salmon re-quantified baseline biopsies; treatment arm + ACR20 only for samples
    confidently matched to the published per-patient table (37_strap_match.py, used = True)."""
    import duckdb
    con = duckdb.connect(str(ROOT / "db" / "ra_rx.duckdb"), read_only=True)
    c = con.execute('SELECT * FROM "strap_gene_counts"').df()
    clin = con.execute('SELECT * FROM "clin_strap"').df()
    match = con.execute('SELECT * FROM "strap_sample_match"').df()
    con.close()
    c = c[c.gene_type == "protein_coding"].drop(columns=["gene_id", "gene_type"]).groupby("gene_symbol").sum()
    clin = clin.set_index("sampleid")
    match = match[match.used].set_index("sample_id")
    m = pd.DataFrame(index=[x for x in c.columns if x.endswith("baseline")])
    m["cohort"] = "STRAP"
    m["patient"] = clin.reindex(m.index)["individual"].values
    m["timepoint"] = "baseline"
    m["treatment"] = match.reindex(m.index)["treatment"].str.lower().values
    m["response_binary"] = match.reindex(m.index)["acr20"].astype(float).values
    m["response_detail"] = m.response_binary.map({1.0: "responder", 0.0: "non-responder"})
    m["pathotype"] = clin.reindex(m.index)["pathotype"].values
    return c[m.index], m


def peac():
    """PEAC (E-MTAB-6141): early treatment-naive RA synovium, Salmon re-quantified; no treatment response in the
    public annotation -> response_binary NaN; used for pathotype / disease-activity (DAS28) controls."""
    import duckdb
    con = duckdb.connect(str(ROOT / "db" / "ra_rx.duckdb"), read_only=True)
    c = con.execute('SELECT * FROM "peac_gene_counts"').df()
    clin = con.execute('SELECT * FROM "clin_peac"').df()
    con.close()
    c = c[c.gene_type == "protein_coding"].drop(columns=["gene_id", "gene_type"]).groupby("gene_symbol").sum()
    clin = clin[clin.organism_part == "synovium"].copy()
    clin.index = clin.source_name.str.replace(" ", "_")
    m = pd.DataFrame(index=clin.index)
    m["cohort"] = "PEAC"
    m["patient"] = clin["individual"].astype(str)
    m["timepoint"] = "baseline"
    m["treatment"] = "treatment-naive"
    m["response_binary"] = np.nan
    m["response_detail"] = np.nan
    m["pathotype"] = clin["pathotype"].str.capitalize().replace({"Ungraded": np.nan})
    for k in ["das28", "crp", "esr", "inflammatory_score", "ccp", "rf"]:
        m[k] = pd.to_numeric(clin[k], errors="coerce")
    keep = m.index.intersection(c.columns)
    return c[keep], m.loc[keep]


LOADERS = {"GSE198520": gse198520, "STRAP": strap, "R4RA": r4ra, "GSE24742": gse24742, "GSE45867": gse45867, "PEAC": peac}
LOG_SCALE = {"GSE24742", "GSE45867"}  # arrays: already log2, no CPM
