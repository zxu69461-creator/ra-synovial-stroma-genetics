"""20_magma.py — MAGMA gene-level analysis for every GWAS used in the paper (adapted from ad_mg/scripts/02_magma.py).

Runs (tag -> DuckDB table magma_genes_<tag>, raw file db/objects/magma/<tag>.genes.raw):
  ra_eur            RA, Ishigaki 2022 EUR (GCST90132223; 22,350 cases / 74,823 controls)        primary
  ra_eur_seropos    seropositive RA, Ishigaki 2022 EUR (GCST90132226; 17,221 / 74,823)         sensitivity
  fg_ra             FinnGen R13 M13_RHEUMA (16,775 / 308,847)                                  replication
  fg_ra_seropos     FinnGen R13 RHEUMA_SEROPOS_STRICT (6,358 / 488,366)                        replication
  fg_ra_seroneg     FinnGen R13 RHEUMA_SERONEG (5,137 / 368,412)                               serotype contrast
  fg_oa             FinnGen R13 M13_ARTHROSIS (106,674 / 308,847)                              contrast trait (stromal)
  fg_height         FinnGen R13 HEIGHT_IRN (N 364,515)                                         non-immune negative control
  fg_dupuytren      FinnGen R13 M13_DUPUTRYEN (6,972 / 350,578)                              fibroblast-disease positive control
  ra_eas(_noMHC)    RA, Ishigaki 2022 EAS (GCST90132224; 11,025 / 162,608; LD ref 1000G EAS)    cross-ancestry replication

All hg38; window 10 kb up / 1.5 kb down (scDRS default); LD ref 1000G EUR (rsID-matched). MHC is kept for the
primary run and removed (chr6:25-34 Mb) in *_noMHC variants for the RA traits, since MHC dominates RA signal.
Intermediate MAGMA files live in a temp dir that is deleted after archiving.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import OBJECTS, register_raw, write_object, write_table  # noqa: E402

MAGMA = ROOT / "tools/magma/magma"
REF = ROOT / "tools/magma/g1000_eur"
GENELOC = ROOT / "tools/magma/NCBI38.gene.loc"
MHC_HG38 = ("6", 25_000_000, 34_000_000)
EBI = "http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000"
FG = "https://storage.googleapis.com/finngen-public-data-r13/summary_stats"


def read_gwascat(path, n):
    df = pd.read_csv(path, sep="\t", usecols=["chromosome", "base_pair_location", "p_value", "rsid"],
                     dtype={"chromosome": str})
    df = df.rename(columns={"rsid": "SNP", "chromosome": "CHR", "base_pair_location": "BP", "p_value": "P"})
    df["N"] = n
    return df[["SNP", "CHR", "BP", "P", "N"]]


def read_finngen(path, n):
    df = pd.read_csv(path, sep="\t", usecols=["#chrom", "pos", "rsids", "pval"], dtype={"#chrom": str})
    df = df.rename(columns={"#chrom": "CHR", "pos": "BP", "pval": "P"})
    df["SNP"] = df.rsids.astype(str).str.split(",").str[0]
    df["CHR"] = df.CHR.replace({"23": "X"})
    df["N"] = n
    return df[["SNP", "CHR", "BP", "P", "N"]]


def gc(g, tag, n, **kw):
    return dict(tag=tag, reader=read_gwascat, n=n, path=ROOT / f"raw/gwas/Ishigaki2022_{g}.h.tsv.gz",
                dataset=f"{g} (Ishigaki 2022 RA GWAS)", url=f"{EBI}/{g}/harmonised/{g}.h.tsv.gz", **kw)


def fg(e, tag, n, **kw):
    return dict(tag=tag, reader=read_finngen, n=n, path=ROOT / f"raw/gwas/finngen_R13_{e}.gz",
                dataset=f"FinnGen R13 {e}", url=f"{FG}/finngen_R13_{e}.gz", **kw)


GWAS = [
    gc("GCST90132223", "ra_eur", 22_350 + 74_823),
    gc("GCST90132223", "ra_eur_noMHC", 22_350 + 74_823, drop_mhc=True),
    gc("GCST90132226", "ra_eur_seropos", 17_221 + 74_823),
    fg("M13_RHEUMA", "fg_ra", 16_775 + 308_847),
    fg("M13_RHEUMA", "fg_ra_noMHC", 16_775 + 308_847, drop_mhc=True),
    fg("RHEUMA_SEROPOS_STRICT", "fg_ra_seropos", 6_358 + 488_366),
    fg("RHEUMA_SERONEG", "fg_ra_seroneg", 5_137 + 368_412),
    fg("M13_ARTHROSIS", "fg_oa", 106_674 + 308_847),
    fg("HEIGHT_IRN", "fg_height", 364_515),
    # fibroblast-disease positive control for the stromal compartment (Dupuytren's contracture)
    fg("M13_DUPUTRYEN", "fg_dupuytren", 6_972 + 350_578),
    # cross-ancestry replication: EAS meta-analysis, 1000G EAS LD reference
    gc("GCST90132224", "ra_eas", 11_025 + 162_608, ref=ROOT / "tools/magma_eas/g1000_eas"),
    gc("GCST90132224", "ra_eas_noMHC", 11_025 + 162_608, drop_mhc=True, ref=ROOT / "tools/magma_eas/g1000_eas"),
]


def run(cmd):
    r = subprocess.run(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[1]} failed ({r.returncode}): {r.stdout[-2000:]}")


def magma_one(g, tmp: Path, only: set | None):
    if only and g["tag"] not in only:
        return
    register_raw(g["path"], g["dataset"], g["url"])
    df = g["reader"](g["path"], g["n"])
    df = df[df.SNP.astype(str).str.startswith("rs") & df.P.notna() & df.BP.notna()].drop_duplicates("SNP")
    df["BP"], df["N"] = df.BP.astype(int), df.N.astype(int)
    df["P"] = df.P.clip(lower=1e-300)  # MHC p-values underflow to denormals MAGMA cannot parse
    if g.get("drop_mhc"):
        df = df[~((df.CHR == MHC_HG38[0]) & df.BP.between(MHC_HG38[1], MHC_HG38[2]))]
    tag, loc, pval, out = g["tag"], tmp / f"{g['tag']}.snploc", tmp / f"{g['tag']}.pval", tmp / g["tag"]
    df[["SNP", "CHR", "BP"]].to_csv(loc, sep="\t", header=False, index=False)
    df[["SNP", "P", "N"]].to_csv(pval, sep="\t", index=False)
    run([MAGMA, "--annotate", "window=10,1.5", "--snp-loc", loc, "--gene-loc", GENELOC, "--out", out])
    run([MAGMA, "--bfile", g.get("ref", REF), "--pval", pval, "ncol=N", "--gene-annot", f"{out}.genes.annot", "--out", out])

    raw_dst = OBJECTS / "magma" / f"{tag}.genes.raw"
    raw_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(f"{out}.genes.raw", raw_dst)
    write_object(raw_dst, f"magma_raw_{tag}", script=__file__, datasets=[g["dataset"]], inputs=[g["path"]],
                 note="MAGMA gene-level raw file for --gene-covar / --set-annot")

    res = pd.read_csv(f"{out}.genes.out", sep=r"\s+")
    sym = pd.read_csv(GENELOC, sep="\t", header=None, usecols=[0, 5], names=["GENE", "SYMBOL"])
    res = res.merge(sym, on="GENE", how="left")
    write_table(res, f"magma_genes_{tag}", script=__file__, datasets=[g["dataset"], f"1000G {Path(str(g.get('ref', REF))).name} (MAGMA ref)"],
                inputs=[g["path"]], note=f"MAGMA v1.10, window 10/1.5 kb, hg38; {len(df):,} input SNPs")
    nsig = (res.P < 0.05 / len(res)).sum()
    print(f"[{tag}] SNPs {len(df):,} | genes {len(res):,} | Bonferroni-sig {nsig}", flush=True)
    print(res.sort_values("P").head(10)[["SYMBOL", "CHR", "NSNPS", "ZSTAT", "P"]].to_string(index=False), flush=True)


def main():
    only = set(sys.argv[1:]) or None
    tmp = Path(tempfile.mkdtemp(prefix="magma_", dir=os.environ.get("SCRATCH")))
    try:
        for g in GWAS:
            magma_one(g, tmp, only)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
