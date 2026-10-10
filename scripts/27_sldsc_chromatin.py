"""27_sldsc_chromatin.py — chromatin-level test of the stromal "blind spot": stratified LD score regression (S-LDSC).

The gene-level map (scDRS, MAGMA gene-property, causal genes) found no RA signal in stromal states,
whereas Ge 2021 (Genome Biol) attributed "up to 24%" of RA heritability to cultured-FLS active chromatin and Tsuchiya
2021 (ARD) found RA loci in cytokine-induced FLS super-enhancers. This script asks the same question at the level of
open chromatin, with the standard heritability-enrichment framework:

  in vivo   GSE243917 single-nucleus multiome (11 RA donors, same nuclei as the replication atlas): donor x compartment
            pseudobulk accessibility; per compartment, the top 10% of peaks by Welch t (compartment vs the other
            compartments, donor pseudobulks as units) = compartment-specific chromatin (LDSC-SEG logic, Finucane 2018).
  in vitro  GSE163548 (Ge 2021) cultured RA FLS ATAC-seq: merged basal peaks, merged TNF-stimulated peaks, and
            TNF-gained peaks (stimulated peaks that overlap no basal peak).
            fls_active = union over the 7 FLS samples of H3K27ac, H3K4me1 and H3K4me3 peaks, i.e. Ge 2021's own
            definition of "active regulatory regions" (their "up to 24% of RA heritability"), to compare proportion
            of heritability with enrichment on the same footing.
  control   AllPeaks = union of every ATAC peak above (so each test is conditional on being open chromatin at all).

Each tested annotation is fitted separately on top of baselineLD v2.2 + AllPeaks (Finucane 2018 / Gupta 2023 design);
the primary statistic is the one-sided P of the annotation's regression coefficient (tau), with enrichment
(proportion of h2 / proportion of SNPs) and tau* reported alongside. Traits: RA (Ishigaki 2022 EUR), FinnGen RA,
osteoarthritis, height and Dupuytren's contracture (fibroblast-disease positive control).

Stages (run in order; each is idempotent):
  python scripts/27_sldsc_chromatin.py munge     HapMap3 LDSC sumstats per trait (store.cached, shared across projects)
  python scripts/27_sldsc_chromatin.py annot     peak sets -> hg19 -> 1000G SNP annotations (db/objects/sldsc/annot)
  python scripts/27_sldsc_chromatin.py l2        LD scores for all annotation columns (PolyFun ldsc.py, 1 cM window)
  python scripts/27_sldsc_chromatin.py h2        one regression per annotation x trait
  python scripts/27_sldsc_chromatin.py collect   DuckDB: sldsc_annotations, sldsc_chromatin

LDSC runs in the dedicated `sldsc` env (PolyFun's python-3 LDSC needs pandas < 2.2); this script itself runs in rarx.
"""
import gzip
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import OBJECTS, read, register_raw, write_table  # noqa: E402
import store  # noqa: E402  (py/store.py: download + content-addressed cache)

LDSC_PY = os.environ.get("LDSC_PYTHON", sys.executable)   # Python 3 environment with PolyFun requirements
LDSC = ROOT / "tools/polyfun/ldsc.py"
REF_TGZ = ROOT / "raw/ldsc"                                   # Zenodo 10515792 archives (scripts/00_download_sldsc.sh)
REF = Path(os.environ.get("LDSC_REF_DIR", ROOT / "refs/ldsc"))   # extracted Zenodo 10515792 archives
HM3 = Path(os.environ.get("HM3_SNPLIST", ROOT / "refs/ldsc/w_hm3.snplist"))
CHAIN = Path(os.environ.get("LIFTOVER_CHAIN", ROOT / "refs/hg38ToHg19.over.chain.gz"))
ATAC = ROOT / "raw/sc/gupta2023/GSE243917_peaksXcells_ALL_raw.h5ad.gz"
FLS = {"fls_basal": ROOT / "raw/geo/GSE163548_Basal.merged_peaks.narrowPeak.gz",
       "fls_stim": ROOT / "raw/geo/GSE163548_Stim.merged_peaks.narrowPeak.gz"}
FLS_RAW = ROOT / "raw/geo/GSE163548_RAW.tar"          # per-sample histone peaks (Ge 2021; GRCh38)
ATLAS = ROOT / "db/objects/atlas/gupta2023_states.h5ad"
OUT = OBJECTS / "sldsc"
COMPARTMENTS = {"T/NK": "iv_TNK", "B/plasma": "iv_Bplasma", "Myeloid": "iv_Myeloid", "Endothelial": "iv_Endothelial",
                "Stromal": "iv_Stromal"}
EXT = 500   # pre-specified sensitivity: in vivo compartment peaks extended by 500 bp each side (control AllPeaks_ext)
TESTS = (list(COMPARTMENTS.values()) + ["fls_basal", "fls_stim", "fls_tnfgained", "fls_active"]
         + [f"{v}_ext" for v in COMPARTMENTS.values()])
MHC_HG38 = (6, 25_000_000, 34_000_000)
TOP = 0.10
MIN_CELLS = 20

GC = "http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000"
FG = "https://storage.googleapis.com/finngen-public-data-r13/summary_stats"
TRAITS = {  # tag: (kind, local copy or None -> store.fetch(url), url, n_case, n_control or None if quantitative)
    "ra_eur": ("gwascat", ROOT / "raw/gwas/Ishigaki2022_GCST90132223.h.tsv.gz",
               f"{GC}/GCST90132223/harmonised/GCST90132223.h.tsv.gz", 22_350, 74_823),
    "fg_ra": ("finngen", ROOT / "raw/gwas/finngen_R13_M13_RHEUMA.gz", f"{FG}/finngen_R13_M13_RHEUMA.gz", 16_775, 308_847),
    "fg_oa": ("finngen", ROOT / "raw/gwas/finngen_R13_M13_ARTHROSIS.gz", f"{FG}/finngen_R13_M13_ARTHROSIS.gz", 106_674, 308_847),
    "fg_height": ("finngen", None, f"{FG}/finngen_R13_HEIGHT_IRN.gz", 364_515, None),
    "fg_dupuytren": ("finngen", None, f"{FG}/finngen_R13_M13_DUPUTRYEN.gz", 6_972, 350_578),
}


def neff(nc, nn):
    return nc if nn is None else 4.0 / (1.0 / nc + 1.0 / nn)


THREADS1 = {**os.environ, **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def run(cmd, log=None):
    """LDSC subprocesses are pinned to one BLAS thread each (they run in parallel; oversubscription stalls them)."""
    r = subprocess.run(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=THREADS1)
    if log:
        Path(log).write_text(r.stdout)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd[:3]))} failed ({r.returncode}): {r.stdout[-3000:]}")
    return r.stdout


# ----------------------------------------------------------------------------------------------------------- munge
def reduce_to_hm3(kind, gz, n, out_dir):
    """rsID + allele match to the HapMap3 list, Z = beta / se signed on A1, MHC (hg38 chr6:25-34 Mb) removed."""
    import duckdb
    con = duckdb.connect()
    src = f"read_csv('{gz}', delim='\t', header=true, compression='gzip', all_varchar=true)"
    if kind == "parquet":  # FinnGen summary statistics stored as Parquet: alt = effect allele, typed columns
        src = f"(SELECT CAST(chrom AS VARCHAR) AS chrom, CAST(pos AS VARCHAR) AS pos, ref, alt, rsid, CAST(beta AS VARCHAR) AS beta, CAST(se AS VARCHAR) AS se FROM read_parquet('{gz}'))"
        ea, oa, ch, bp, b, se = "alt", "ref", "chrom", "pos", "beta", "se"
        rsid = "s.rsid"
    elif kind == "gwascat":
        rs, ea, oa, ch, bp, b, se = "rsid", "effect_allele", "other_allele", "chromosome", "base_pair_location", "beta", "standard_error"
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
        if "hm_rsid" in cols:
            rs, ea, oa, ch, bp, b = "hm_rsid", "hm_effect_allele", "hm_other_allele", "hm_chrom", "hm_pos", "hm_beta"
        rsid = f"s.{rs}"
    else:  # FinnGen: alt is the effect allele, rsids may hold several ids
        ea, oa, ch, bp, b, se = "alt", "ref", '"#chrom"', "pos", "beta", "sebeta"
        rsid = "split_part(s.rsids, ',', 1)"
    con.execute(f"CREATE TABLE h AS SELECT * FROM read_csv('{HM3}', delim='\t', header=true)")
    d = con.execute(f"""
        SELECT {rsid} AS SNP, upper(s.{ea}) AS A1, upper(s.{oa}) AS A2,
               try_cast(s.{b} AS DOUBLE) / try_cast(s.{se} AS DOUBLE) AS Z, {n} AS N
        FROM {src} s JOIN h ON {rsid} = h.SNP
        WHERE ((upper(s.{ea}) = h.A1 AND upper(s.{oa}) = h.A2) OR (upper(s.{ea}) = h.A2 AND upper(s.{oa}) = h.A1))
          AND NOT (try_cast(s.{ch} AS INTEGER) = {MHC_HG38[0]}
                   AND try_cast(s.{bp} AS BIGINT) BETWEEN {MHC_HG38[1]} AND {MHC_HG38[2]})
        QUALIFY row_number() OVER (PARTITION BY {rsid} ORDER BY 1) = 1""").fetchdf().dropna()
    con.close()
    d = d[d.Z.abs() < 1e3]
    with gzip.open(out_dir / "trait.sumstats.gz", "wt") as fh:
        d.to_csv(fh, sep="\t", index=False)
    return len(d)


def munged(tag):
    kind, path, url, nc, nn = TRAITS[tag]
    acc = Path(url).name.replace(".h.tsv.gz", "").replace(".gz", "")
    gz = path if path is not None and Path(path).exists() else Path(store.fetch(url, acc=acc))
    if kind != "parquet":
        register_raw(gz, f"{tag} GWAS summary statistics", url)
    n = round(neff(nc, nn), 2)
    out = store.cached("munge_hm3_ldsc", lambda o: reduce_to_hm3(kind, gz, n, o), inputs=[gz],
                       params={"N": n, "hm3": "w_hm3.snplist", "mhc": "chr6:25-34Mb (hg38)",
                               "reduction": f"ra_rx 27 {kind}: rsID+allele match, Z=beta/se"},
                       version="1", note=f"{tag} HapMap3 LDSC sumstats", script=__file__)
    return Path(out) / "trait.sumstats.gz"


def stage_munge():
    for t in TRAITS:
        f = munged(t)
        n = sum(1 for _ in gzip.open(f, "rt")) - 1
        print(f"{t}: {n:,} HapMap3 SNPs -> {f}", flush=True)


# ----------------------------------------------------------------------------------------------------------- annot
def extract_refs():
    REF.mkdir(parents=True, exist_ok=True)
    for name in ["1000G_Phase3_baselineLD_v2.2_ldscores.tgz", "1000G_Phase3_plinkfiles.tgz",
                 "1000G_Phase3_weights_hm3_no_MHC.tgz", "1000G_Phase3_frq.tgz"]:
        flag = REF / f".{name}.done"
        if not flag.exists():
            with tarfile.open(REF_TGZ / name) as tf:
                tf.extractall(REF)
            flag.write_text("ok")
    plink = next(REF.rglob("1000G.EUR.QC.22.bim")).parent
    base = next(REF.rglob("baselineLD.22.l2.ldscore.gz")).parent
    w = next(REF.rglob("weights.hm3_noMHC.22.l2.ldscore.gz")).parent
    frq = next(REF.rglob("1000G.EUR.QC.22.frq")).parent
    return dict(plink=plink, base=base, w=w, frq=frq)


def parse_peak_names(names):
    p = pd.Series(names).str.extract(r"^(chr[0-9XYM]+)[:\-_](\d+)[\-_](\d+)$")
    p.columns = ["chrom", "start", "end"]
    p["start"], p["end"] = p.start.astype(float), p.end.astype(float)
    return p


def invivo_peaks():
    """Donor x compartment pseudobulk of the snATAC peak counts; compartment-specific = top 10% Welch t."""
    import anndata as ad
    import scipy.sparse as sp
    from scipy import stats
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCRATCH")) as td:
        h5 = Path(td) / "atac.h5ad"
        subprocess.run(f"gzip -dc '{ATAC}' > '{h5}'", shell=True, check=True)
        a = ad.read_h5ad(h5)
    lab = ad.read_h5ad(ATLAS, backed="r").obs[["sample", "state_group"]].copy()
    common = a.obs_names.intersection(lab.index)
    if len(common) < 0.5 * len(lab):  # fall back to matching on barcode + sample if the prefixes differ
        raise RuntimeError(f"only {len(common)} of {len(lab)} atlas nuclei matched ATAC barcodes; "
                           f"examples ATAC {list(a.obs_names[:3])} vs RNA {list(lab.index[:3])}")
    a = a[common]
    lab = lab.loc[common]
    X = sp.csr_matrix(a.X)
    keys = lab["sample"].astype(str) + "|" + lab["state_group"].astype(str)
    groups = keys.value_counts()
    groups = groups[groups >= MIN_CELLS].index
    codes = np.asarray(pd.Categorical(keys, categories=groups).codes)
    mask = codes >= 0
    G = sp.csr_matrix((np.ones(mask.sum()), (codes[mask], np.arange(mask.sum()))), shape=(len(groups), mask.sum()))
    PB = np.asarray((G @ X[mask]).todense(), dtype=float)
    lib = PB.sum(axis=1, keepdims=True)
    L = np.log2(PB / lib * 1e6 + 1)
    comp = pd.Index(groups).str.split("|").str[1]
    peaks = parse_peak_names(a.var_names)
    rows, spec = [], {}
    for c, tag in COMPARTMENTS.items():
        inn, out = (comp == c), (comp != c)
        t = stats.ttest_ind(L[inn], L[out], axis=0, equal_var=False).statistic
        t = np.nan_to_num(t, nan=-np.inf)
        k = int(round(TOP * len(t)))
        idx = np.argsort(-t)[:k]
        spec[tag] = peaks.iloc[idx].assign(t=t[idx])
        rows.append(dict(annotation=tag, source="GSE243917 snATAC (11 RA donors)", n_pseudobulks=int(inn.sum()),
                         n_peaks=k, t_min=float(t[idx].min())))
    spec["iv_all"] = peaks.assign(t=np.nan)
    return spec, pd.DataFrame(rows), dict(n_nuclei=int(len(common)), n_peaks=int(a.n_vars), n_pseudobulks=int(len(groups)))


def fls_peaks():
    out = {}
    for tag, f in FLS.items():
        register_raw(f, "GSE163548 (Ge 2021 FLS ATAC-seq, merged peaks)",
                     f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE163nnn/GSE163548/suppl/{f.name}")
        d = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
        d["chrom"] = np.where(d.chrom.astype(str).str.startswith("chr"), d.chrom.astype(str), "chr" + d.chrom.astype(str))
        out[tag] = d.astype({"start": float, "end": float})
    b = merge(out["fls_basal"])
    gained = []
    for ch, g in out["fls_stim"].groupby("chrom"):
        bb = b[b.chrom == ch]
        s, e = bb.start.to_numpy(), bb.end.to_numpy()
        i = np.searchsorted(s, g.end.to_numpy(), side="left") - 1  # last basal interval starting before peak end
        hit = (i >= 0) & (e[np.clip(i, 0, None)] > g.start.to_numpy()) if len(s) else np.zeros(len(g), bool)
        gained.append(g[~hit])
    out["fls_tnfgained"] = pd.concat(gained, ignore_index=True)
    register_raw(FLS_RAW, "GSE163548 (Ge 2021) per-sample ChIP/ATAC peaks",
                 "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE163nnn/GSE163548/suppl/GSE163548_RAW.tar")
    act = []
    with tarfile.open(FLS_RAW) as tf:
        for m in tf.getmembers():
            if re.search(r"_(H3K27ac|H3K4me1|H3K4me3)_.*Peak\.gz$", m.name):
                d = pd.read_csv(tf.extractfile(m), sep="\t", header=None, usecols=[0, 1, 2], compression="gzip",
                                names=["chrom", "start", "end"])
                act.append(d)
    a = pd.concat(act, ignore_index=True)
    a["chrom"] = np.where(a.chrom.astype(str).str.startswith("chr"), a.chrom.astype(str), "chr" + a.chrom.astype(str))
    out["fls_active"] = merge(a.astype({"start": float, "end": float}))
    print(f"fls_active: {len(act)} histone peak files (H3K27ac/H3K4me1/H3K4me3) -> {len(out['fls_active']):,} regions",
          flush=True)
    return out


def merge(d):
    d = d.sort_values(["chrom", "start"]).reset_index(drop=True)
    res = []
    for ch, g in d.groupby("chrom", sort=False):
        s, e = g.start.to_numpy(), g.end.to_numpy()
        cs, ce = s[0], e[0]
        for a, b in zip(s[1:], e[1:]):
            if a <= ce:
                ce = max(ce, b)
            else:
                res.append((ch, cs, ce)); cs, ce = a, b
        res.append((ch, cs, ce))
    return pd.DataFrame(res, columns=["chrom", "start", "end"])


def lift(d):
    from pyliftover import LiftOver
    lo = LiftOver(str(CHAIN))
    keep = []
    for ch, s, e in d[["chrom", "start", "end"]].itertuples(index=False):
        a, b = lo.convert_coordinate(ch, int(s)), lo.convert_coordinate(ch, int(e) - 1)
        if a and b and a[0][0] == ch and b[0][0] == ch:
            s19, e19 = a[0][1], b[0][1] + 1
            if 0 < e19 - s19 <= 2 * (e - s) + 100:
                keep.append((ch, s19, e19))
    return merge(pd.DataFrame(keep, columns=["chrom", "start", "end"]))


def stage_annot():
    refs = extract_refs()
    iv, iv_tab, iv_meta = invivo_peaks()
    fls = fls_peaks()
    sets = {**{k: v for k, v in iv.items()}, **fls}
    hg19 = {k: lift(v) for k, v in sets.items()}
    hg19["AllPeaks"] = merge(pd.concat([hg19["iv_all"], hg19["fls_basal"], hg19["fls_stim"]]))
    ext = lambda d: merge(d.assign(start=(d.start - EXT).clip(lower=0), end=d.end + EXT))
    for v in list(COMPARTMENTS.values()) + ["AllPeaks"]:
        hg19[f"{v}_ext"] = ext(hg19[v])
    cols = ["AllPeaks", "AllPeaks_ext"] + TESTS
    adir = OUT / "annot"
    adir.mkdir(parents=True, exist_ok=True)
    nsnp = {c: 0 for c in cols}
    for chrn in range(1, 23):
        bim = pd.read_csv(refs["plink"] / f"1000G.EUR.QC.{chrn}.bim", sep="\t", header=None,
                          names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
        A = bim[["CHR", "BP", "SNP", "CM"]].copy()
        for c in cols:
            iv_ = hg19[c][hg19[c].chrom == f"chr{chrn}"]
            s, e = iv_.start.to_numpy(), iv_.end.to_numpy()
            i = np.searchsorted(s, bim.BP.to_numpy(), side="right") - 1
            hit = (i >= 0) & (e[np.clip(i, 0, None)] > bim.BP.to_numpy()) if len(s) else np.zeros(len(bim), bool)
            A[c] = hit.astype(int)
            nsnp[c] += int(hit.sum())
        A.to_csv(adir / f"ra_rx.{chrn}.annot.gz", sep="\t", index=False, compression="gzip")
    meta = pd.DataFrame([dict(annotation=c, n_intervals_hg19=len(hg19[c]),
                              mb_hg19=float((hg19[c].end - hg19[c].start).sum() / 1e6), n_1000g_snps=nsnp[c],
                              median_bp=float((hg19[c].end - hg19[c].start).median()),
                              n_peaks_hg38=len(sets[c]) if c in sets else None) for c in cols])
    meta = meta.merge(iv_tab, on="annotation", how="left")
    (OUT / "annot_meta.json").write_text(json.dumps(dict(iv=iv_meta), indent=1))
    meta.to_csv(OUT / "annot_meta.tsv", sep="\t", index=False)
    print(meta.to_string(index=False))
    print(iv_meta)


# ----------------------------------------------------------------------------------------------------------- l2
def stage_l2(workers=8):
    refs = extract_refs()
    ldir = OUT / "ldscore"
    ldir.mkdir(parents=True, exist_ok=True)
    snps = OUT / "hm3_snps.txt"
    pd.read_csv(HM3, sep="\t").SNP.to_csv(snps, index=False, header=False)

    def one(c):
        if (ldir / f"ra_rx.{c}.l2.ldscore.gz").exists():
            return c
        run([LDSC_PY, LDSC, "--l2", "--bfile", refs["plink"] / f"1000G.EUR.QC.{c}", "--ld-wind-cm", 1,
             "--annot", OUT / f"annot/ra_rx.{c}.annot.gz", "--out", ldir / f"ra_rx.{c}", "--print-snps", snps],
            log=ldir / f"ra_rx.{c}.l2.stdout")
        return c
    with ThreadPoolExecutor(workers) as ex:
        for c in ex.map(one, range(22, 0, -1)):
            print(f"l2 chr{c} done", flush=True)


# ----------------------------------------------------------------------------------------------------------- h2
def _subset_chr(c, tests):
    """One chromosome: read the full LD-score / annot / M files once, write the AllPeaks(+_ext) + test subset for
    every pending test (fast gzip: these files are intermediates)."""
    src = OUT / "ldscore" / f"ra_rx.{c}"
    L = pd.read_csv(f"{src}.l2.ldscore.gz", sep="\t")
    names = [n for n in L.columns if n not in ("CHR", "SNP", "BP")]
    M = {suf: Path(f"{src}.{suf}").read_text().split() for suf in ("l2.M", "l2.M_5_50")}
    A = pd.read_csv(OUT / f"annot/ra_rx.{c}.annot.gz", sep="\t")
    gz = {"method": "gzip", "compresslevel": 1}
    for test in tests:
        sdir = OUT / "sets" / test
        keep = ["AllPeaks_ext" if test.endswith("_ext") else "AllPeaks", test]
        idx = [names.index(f"{k}L2") if f"{k}L2" in names else names.index(k) for k in keep]
        L[["CHR", "SNP", "BP"] + [names[i] for i in idx]].to_csv(f"{sdir}/ra_rx.{c}.l2.ldscore.gz", sep="\t",
                                                                   index=False, compression=gz)
        A[["CHR", "BP", "SNP", "CM"] + keep].to_csv(f"{sdir}/ra_rx.{c}.annot.gz", sep="\t", index=False,
                                                     compression=gz)
        for suf, m in M.items():                       # M last: its presence marks the chromosome as complete
            Path(f"{sdir}/ra_rx.{c}.{suf}").write_text("\t".join(m[i] for i in idx) + "\n")
    return c


def build_sets(tests, workers=4):
    """LD-score / annot / M files with only AllPeaks + one tested annotation, so each test is fitted on its own."""
    from concurrent.futures import ProcessPoolExecutor
    pending = {}
    for test in tests:
        sdir = OUT / "sets" / test
        sdir.mkdir(parents=True, exist_ok=True)
        for c in range(1, 23):
            if not (sdir / f"ra_rx.{c}.l2.M_5_50").exists():
                pending.setdefault(c, []).append(test)
    with ProcessPoolExecutor(workers) as ex:
        for c in ex.map(_subset_chr, list(pending), list(pending.values())):
            print(f"sets chr{c} done ({len(pending[c])} tests)", flush=True)
    return {t: OUT / "sets" / t for t in tests}


def _pq_chr(c, base, dst):
    """baselineLD chromosome c: LD scores and annotations as Parquet (PolyFun reads <prefix>.parquet before .gz);
    M files are symlinked. Written to a temp name and renamed, so a reader never sees a partial file."""
    for kind, src in (("l2.ldscore", base / f"baselineLD.{c}.l2.ldscore.gz"), ("annot", base / f"baselineLD.{c}.annot.gz")):
        out = dst / f"baselineLD.{c}.{kind}.parquet"
        if not out.exists():
            tmp = out.with_suffix(".tmp")
            pd.read_csv(src, sep="\t").to_parquet(tmp, index=False)
            tmp.rename(out)
    for suf in ("l2.M", "l2.M_5_50"):
        link = dst / f"baselineLD.{c}.{suf}"
        if not link.exists():
            link.symlink_to(base / f"baselineLD.{c}.{suf}")
    return c


def baseline_parquet(base, workers=4):
    """Project-local Parquet copy of baselineLD v2.2 (identical values; reading 97-column gz text dominated each
    regression)."""
    from concurrent.futures import ProcessPoolExecutor
    dst = OUT / "baseline_pq"
    dst.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_pq_chr, range(1, 23), [base] * 22, [dst] * 22))
    return dst


def polyfun_sumstats(tag, wdir):
    """PolyFun's LDSC parser also requires CHR and BP: add the hg19 positions of the regression SNPs (from the LDSC
    weight files) to the shared munged file; the shared cached file itself is left in the standard 5-column form."""
    out = OUT / "sumstats" / f"{tag}.sumstats.gz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    pos = pd.concat([pd.read_csv(wdir / f"weights.hm3_noMHC.{c}.l2.ldscore.gz", sep="\t", usecols=["CHR", "SNP", "BP"])
                     for c in range(1, 23)])
    d = pd.read_csv(munged(tag), sep="\t").merge(pos, on="SNP", how="inner")
    d[["SNP", "CHR", "BP", "A1", "A2", "Z", "N"]].to_csv(out, sep="\t", index=False, compression="gzip")
    return out


def stage_h2(workers=4):
    refs = extract_refs()
    rdir = OUT / "h2"
    rdir.mkdir(parents=True, exist_ok=True)
    sets = build_sets(TESTS)
    bpq = baseline_parquet(refs["base"])
    for t in TRAITS:                                   # written once here, not concurrently inside the workers
        polyfun_sumstats(t, refs["w"])
    rank = {t: i for i, t in enumerate(["ra_eur", "fg_dupuytren", "fg_height", "fg_ra", "fg_oa"])}
    jobs = sorted([(test, t, sets[test]) for test in TESTS for t in TRAITS],     # primary tests and key traits first
                  key=lambda j: (j[0].endswith("_ext"), rank[j[1]], TESTS.index(j[0])))

    def one(job):
        test, t, sdir = job
        out = rdir / f"{test}__{t}"
        if Path(f"{out}.results").exists():
            return out
        run([LDSC_PY, LDSC, "--h2", polyfun_sumstats(t, refs["w"]), "--ref-ld-chr", f"{bpq}/baselineLD.,{sdir}/ra_rx.",
             "--w-ld-chr", f"{refs['w']}/weights.hm3_noMHC.", "--overlap-annot", "--frqfile-chr",
             f"{refs['frq']}/1000G.EUR.QC.", "--print-coefficients", "--out", out], log=f"{out}.stdout")
        return out
    with ThreadPoolExecutor(workers) as ex:
        for o in ex.map(one, jobs):
            print(f"h2 {o.name}", flush=True)


# ----------------------------------------------------------------------------------------------------------- collect
def stage_collect():
    from scipy.stats import norm
    meta = pd.read_csv(OUT / "annot_meta.tsv", sep="\t")
    base = extract_refs()["base"]
    M_total = sum(float((base / f"baselineLD.{c}.l2.M_5_50").read_text().split()[0]) for c in range(1, 23))
    rows = []
    for test in TESTS:
        for t in TRAITS:
            f = OUT / "h2" / f"{test}__{t}"
            R = pd.read_csv(f"{f}.results", sep="\t")
            r = R[R.Category.str.replace(r"L2_\d+$", "", regex=True).str.replace("L2$", "", regex=True) == test]
            if r.empty:
                r = R.iloc[[-1]]
            r = r.iloc[0]
            log = Path(f"{f}.log").read_text() if Path(f"{f}.log").exists() else Path(f"{f}.stdout").read_text()
            h2 = float(re.search(r"Total Observed scale h2: ([\-0-9.e]+)", log).group(1))
            p = float(r["Prop._SNPs"])
            tau_star = float(r["Coefficient"]) * np.sqrt(p * (1 - p)) / (h2 / M_total) if h2 > 0 else np.nan
            z = float(r["Coefficient_z-score"])
            rows.append(dict(annotation=test, trait=t, prop_snps=p, prop_h2=float(r["Prop._h2"]),
                             prop_h2_se=float(r["Prop._h2_std_error"]), enrichment=float(r["Enrichment"]),
                             enrichment_se=float(r["Enrichment_std_error"]), enrichment_p=float(r["Enrichment_p"]),
                             coef=float(r["Coefficient"]), coef_se=float(r["Coefficient_std_error"]), coef_z=z,
                             coef_p=float(norm.sf(z)), tau_star=tau_star, h2_obs=h2))
    D = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    D["coef_fdr"] = D.groupby("trait").coef_p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
    datasets = ["GSE243917", "GSE163548", "Zenodo 10515792 (baselineLD v2.2, 1000G EUR)"] + [Path(TRAITS[t][2]).name for t in TRAITS]
    write_table(meta, "sldsc_annotations", script=__file__, datasets=datasets[:3],
                note="S-LDSC annotations: in vivo compartment-specific snATAC peaks (top 10% Welch t, donor pseudobulk), "
                     "Ge 2021 FLS basal / TNF / TNF-gained peaks, AllPeaks union control; hg38->hg19")
    write_table(D, "sldsc_chromatin", script=__file__, datasets=datasets,
                note="S-LDSC, baselineLD v2.2 + AllPeaks + one annotation; coef_p one-sided; FDR within trait; "
                     "tau* = coef*sd/(h2/M_5_50)")
    pd.set_option("display.width", 220)
    print(D.pivot_table(index="annotation", columns="trait", values="coef_p").reindex(TESTS).map(lambda x: f"{x:.2g}"))
    print(D.pivot_table(index="annotation", columns="trait", values="enrichment").reindex(TESTS).round(2))
    print(D[D.annotation == "fls_basal"][["trait", "prop_snps", "prop_h2", "prop_h2_se"]])


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"munge": stage_munge, "annot": stage_annot, "l2": stage_l2, "pq": lambda: print(baseline_parquet(extract_refs()["base"])),
     "h2": stage_h2, "collect": stage_collect}[stage]()
