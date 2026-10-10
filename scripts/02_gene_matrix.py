"""02_gene_matrix.py — aggregate Salmon transcript quantifications to gene level and archive as Parquet views.

Salmon index is built WITHOUT --gencode, so transcript names keep the full GENCODE header
(ENST|ENSG|OTTHUMG|OTTHUMT|tx_name|gene_symbol|length|biotype|); genes are summed from that directly
(equivalent to tximport countsFromAbundance="no").

Usage: python scripts/02_gene_matrix.py E-MTAB-11611 [--prefix r4ra]
Views: <prefix>_gene_counts, <prefix>_gene_tpm  (columns: gene_id, gene_symbol, gene_type, <sample ids...>)
Table: <prefix>_salmon_qc  (mapping rate, library type, reads processed per sample)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import write_parquet_view, write_table  # noqa: E402


def load(qdir: Path):
    q = pd.read_csv(qdir / "quant.sf", sep="\t", usecols=["Name", "TPM", "NumReads"])
    parts = q.Name.str.split("|", expand=True)
    q["gene_id"], q["gene_symbol"], q["gene_type"] = parts[1].str.split(".").str[0], parts[5], parts[7]
    g = q.groupby(["gene_id", "gene_symbol", "gene_type"], sort=True)[["NumReads", "TPM"]].sum()
    meta = json.loads((qdir / "aux_info" / "meta_info.json").read_text())
    lib = json.loads((qdir / "lib_format_counts.json").read_text())
    qc = {"sample_id": qdir.name, "num_processed": meta["num_processed"], "num_mapped": meta["num_mapped"],
          "percent_mapped": meta["percent_mapped"], "library_types": ",".join(meta["library_types"]),
          "expected_format": lib["expected_format"], "salmon_version": meta["salmon_version"]}
    return g, qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accession")
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    base = ROOT / "db" / "objects" / "salmon" / a.accession
    dirs = sorted(d for d in base.iterdir() if (d / "quant.sf").exists() and not d.name.endswith(".partial"))
    counts, tpm, qcs = {}, {}, []
    for d in dirs:
        g, qc = load(d)
        counts[d.name], tpm[d.name] = g.NumReads, g.TPM
        qcs.append(qc)
    C = pd.DataFrame(counts).round(3).reset_index()
    T = pd.DataFrame(tpm).round(4).reset_index()
    ds = [a.accession, "GENCODE_v46"]
    write_parquet_view(C, f"{a.prefix}_gene_counts", script=__file__, datasets=ds,
                       note=f"Salmon NumReads summed per gene; {len(dirs)} samples; wide (genes x samples)")
    write_parquet_view(T, f"{a.prefix}_gene_tpm", script=__file__, datasets=ds,
                       note=f"Salmon TPM summed per gene; {len(dirs)} samples")
    write_table(pd.DataFrame(qcs), f"{a.prefix}_salmon_qc", script=__file__, datasets=ds)
    print(f"{a.prefix}: {C.shape[0]} genes x {len(dirs)} samples")


if __name__ == "__main__":
    main()
