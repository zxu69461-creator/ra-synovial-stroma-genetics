"""27b_sldsc_sensitivity.py — chromatin sensitivity and comparator analyses (extends 27_sldsc_chromatin.py).

New annotations (hg19), each tested on top of baselineLD v2.2 + the same all-open-chromatin control (AllPeaks):
  fls_active_<line>   Ge 2021 active regions per FLS line (union of H3K27ac, H3K4me1, H3K4me3 peaks of ONE line), as
                      in Ge et al. (who reported 12-24% of RA h2 per line); 7 lines.
  rm_T_active,        same construction in Roadmap primary T cells (E034) and B cells (E032): an RA-relevant comparator
  rm_B_active         built exactly like the FLS regions, to show that the model is sensitive for RA.
  iv_<comp>_acc       "accessible" chromatin of each in vivo compartment: the half of all snATAC peaks with the highest
                      pooled CPM in that compartment (absolute, not specificity-based; cf. Gupta 2023).
  iv_fib_lining,      sub-compartment-specific snATAC peaks (top 10% Welch t of donor pseudobulks vs every other
  iv_fib_sublining,   pseudobulk, other stromal groups included): lining fibroblasts (F-0, F-1), sublining fibroblasts
  iv_mural            (F-2 to F-7), mural cells (Mu-0).
Baseline-only sensitivity: the nine primary annotations of 27_sldsc_chromatin.py refitted without the AllPeaks control.
Fine-mapping overlap: PIP-weighted share of Ishigaki 2022 fine-mapped variants (ST-6; PIP > 0.1) in each annotation.

Stages: annot | l2 | h2 | finemap | collect      (traits: RA Ishigaki EUR, height, Dupuytren's contracture)
DuckDB: sldsc_chromatin_r2b (model = 'AllPeaks control' | 'baseline only'), sldsc_annotations_r2b, finemap_overlap
"""
import gzip
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
from archive import OBJECTS, register_raw, write_table  # noqa: E402

LDSC_PY = os.environ.get("LDSC_PYTHON", sys.executable)
LDSC = ROOT / "tools/polyfun/ldsc.py"
REF = Path(os.environ.get("LDSC_REF_DIR", ROOT / "refs/ldsc"))
PLINK, W, FRQ = REF / "1000G_EUR_Phase3_plink", REF / "1000G_Phase3_weights_hm3_no_MHC", REF / "1000G_Phase3_frq"
CHAIN = Path(os.environ.get("LIFTOVER_CHAIN", ROOT / "refs/hg38ToHg19.over.chain.gz"))
ATAC = ROOT / "raw/sc/gupta2023/GSE243917_peaksXcells_ALL_raw.h5ad.gz"
ATLAS = ROOT / "db/objects/atlas/gupta2023_states.h5ad"
FLS_RAW = ROOT / "raw/geo/GSE163548_RAW.tar"
ISHIGAKI = ROOT / "refs/ishigaki2022_supp/41588_2022_1213_MOESM4_ESM.xlsx"
R1 = OBJECTS / "sldsc"                    # outputs of 27_sldsc_chromatin.py (annot, ldscore, sets, sumstats, baseline_pq)
OUT = OBJECTS / "sldsc_r2b"
ROADMAP = "https://egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/narrowPeak"
COMP = {"T/NK": "TNK", "B/plasma": "Bplasma", "Myeloid": "Myeloid", "Endothelial": "Endothelial", "Stromal": "Stromal"}
SUB = {"iv_fib_lining": ["F-0", "F-1"], "iv_fib_sublining": ["F-2", "F-3", "F-4", "F-5", "F-6", "F-7"],
       "iv_mural": ["Mu-0"]}
PRIMARY = ["iv_TNK", "iv_Bplasma", "iv_Myeloid", "iv_Endothelial", "iv_Stromal", "fls_basal", "fls_stim",
           "fls_tnfgained", "fls_active"]
TRAITS = ["ra_eur", "fg_height", "fg_dupuytren"]
MIN_CELLS, TOP = 20, 0.10
THREADS1 = {**os.environ, **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def run(cmd, log=None):
    r = subprocess.run(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=THREADS1)
    if log:
        Path(log).write_text(r.stdout)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd[:3]))} failed ({r.returncode}): {r.stdout[-3000:]}")
    return r.stdout


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


def read_bed(fh, compression="gzip"):
    d = pd.read_csv(fh, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], compression=compression)
    d["chrom"] = np.where(d.chrom.astype(str).str.startswith("chr"), d.chrom.astype(str), "chr" + d.chrom.astype(str))
    return d.astype({"start": float, "end": float})


# ------------------------------------------------------------------------------------------------ annotations
def fls_per_line():
    marks = {}
    with tarfile.open(FLS_RAW) as tf:
        for m in tf.getmembers():
            g = re.search(r"GSM\d+_(\d+)_(H3K27ac|H3K4me1|H3K4me3)_(\w+?)_peaks\.\w+Peak\.gz$", m.name)
            if g:
                line = f"{g.group(1)}_{g.group(3)}"
                marks.setdefault(line, []).append(read_bed(tf.extractfile(m)))
    return {f"fls_active_{k}": lift(merge(pd.concat(v))) for k, v in marks.items() if len(v) == 3}


def roadmap():
    import store
    out = {}
    for eid, tag in (("E034", "rm_T_active"), ("E032", "rm_B_active")):
        parts = []
        for m in ("H3K27ac", "H3K4me1", "H3K4me3"):
            url = f"{ROADMAP}/{eid}-{m}.narrowPeak.gz"
            f = Path(store.fetch(url, acc=f"roadmap_{eid}", link=ROOT / "raw/geo"))
            register_raw(f, f"Roadmap {eid} {m} consolidated narrowPeak (hg19)", url)
            parts.append(read_bed(f))
        out[tag] = merge(pd.concat(parts))                       # Roadmap peaks are hg19 already
    return out


def invivo():
    import anndata as ad
    import scipy.sparse as sp
    from scipy import stats
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCRATCH")) as td:
        h5 = Path(td) / "atac.h5ad"
        subprocess.run(f"gzip -dc '{ATAC}' > '{h5}'", shell=True, check=True)
        a = ad.read_h5ad(h5)
    lab = ad.read_h5ad(ATLAS, backed="r").obs[["sample", "state_group", "state"]].copy()
    common = a.obs_names.intersection(lab.index)
    a, lab = a[common], lab.loc[common]
    X = sp.csr_matrix(a.X)
    peaks = pd.Series(a.var_names).str.extract(r"^(chr[0-9XYM]+)[:\-_](\d+)[\-_](\d+)$")
    peaks.columns = ["chrom", "start", "end"]
    peaks = peaks.astype({"start": float, "end": float})
    out = {}
    # (1) accessible half per compartment: pooled CPM over all nuclei of the compartment
    for c, tag in COMP.items():
        m = (lab.state_group.astype(str) == c).to_numpy()
        cnt = np.asarray(X[m].sum(axis=0)).ravel()
        cpm = cnt / cnt.sum() * 1e6
        keep = cpm >= np.median(cpm)
        out[f"iv_{tag}_acc"] = peaks[keep].copy()
    # (2) sub-compartment-specific peaks: donor pseudobulks of each group vs every other donor pseudobulk
    code = lab.state.astype(str).str.split(":").str[0]
    grp = lab.state_group.astype(str).copy()
    for g, members in SUB.items():
        grp[code.isin(members).to_numpy()] = g
    grp[(lab.state_group.astype(str) == "Stromal").to_numpy() & ~grp.isin(list(SUB)).to_numpy()] = "stromal_other"
    keys = lab["sample"].astype(str) + "|" + grp
    counts = keys.value_counts()
    groups = counts[counts >= MIN_CELLS].index
    codes = np.asarray(pd.Categorical(keys, categories=groups).codes)
    mask = codes >= 0
    G = sp.csr_matrix((np.ones(mask.sum()), (codes[mask], np.arange(mask.sum()))), shape=(len(groups), mask.sum()))
    PB = np.asarray((G @ X[mask]).todense(), dtype=float)
    L = np.log2(PB / PB.sum(axis=1, keepdims=True) * 1e6 + 1)
    gname = pd.Index(groups).str.split("|").str[1]
    for g in SUB:
        inn = (gname == g)
        t = np.nan_to_num(stats.ttest_ind(L[inn], L[~inn], axis=0, equal_var=False).statistic, nan=-np.inf)
        idx = np.argsort(-t)[: int(round(TOP * len(t)))]
        out[g] = peaks.iloc[idx].copy()
        print(f"{g}: {int(inn.sum())} donor pseudobulks", flush=True)
    return {k: lift(v) for k, v in out.items()}


def stage_annot():
    sets = {**fls_per_line(), **roadmap(), **invivo()}
    cols = ["AllPeaks"] + list(sets)
    adir = OUT / "annot"
    adir.mkdir(parents=True, exist_ok=True)
    nsnp = {c: 0 for c in cols}
    for c in range(1, 23):
        A = pd.read_csv(R1 / f"annot/ra_rx.{c}.annot.gz", sep="\t", usecols=["CHR", "BP", "SNP", "CM", "AllPeaks"])
        nsnp["AllPeaks"] += int(A.AllPeaks.sum())
        for k, iv in sets.items():
            iv_ = iv[iv.chrom == f"chr{c}"]
            s, e = iv_.start.to_numpy(), iv_.end.to_numpy()
            i = np.searchsorted(s, A.BP.to_numpy(), side="right") - 1
            hit = (i >= 0) & (e[np.clip(i, 0, None)] > A.BP.to_numpy()) if len(s) else np.zeros(len(A), bool)
            A[k] = hit.astype(int)
            nsnp[k] += int(hit.sum())
        A.to_csv(adir / f"ra_rx2.{c}.annot.gz", sep="\t", index=False, compression="gzip")
    meta = pd.DataFrame([dict(annotation=k, n_intervals_hg19=len(v), mb_hg19=float((v.end - v.start).sum() / 1e6),
                              median_bp=float((v.end - v.start).median()), n_1000g_snps=nsnp[k])
                         for k, v in sets.items()])
    meta.to_csv(OUT / "annot_meta.tsv", sep="\t", index=False)
    print(meta.to_string(index=False))


def stage_l2(workers=6):
    ldir = OUT / "ldscore"
    ldir.mkdir(parents=True, exist_ok=True)

    def one(c):
        if not (ldir / f"ra_rx2.{c}.l2.ldscore.gz").exists():
            run([LDSC_PY, LDSC, "--l2", "--bfile", PLINK / f"1000G.EUR.QC.{c}", "--ld-wind-cm", 1,
                 "--annot", OUT / f"annot/ra_rx2.{c}.annot.gz", "--out", ldir / f"ra_rx2.{c}",
                 "--print-snps", R1 / "hm3_snps.txt"], log=ldir / f"ra_rx2.{c}.l2.stdout")
        return c
    with ThreadPoolExecutor(workers) as ex:
        for c in ex.map(one, range(22, 0, -1)):
            print(f"l2 chr{c} done", flush=True)


# ------------------------------------------------------------------------------------------------ regressions
def subset(prefix_ld, prefix_annot, keep, sdir):
    """One LD-score/annot/M subset per chromosome (written once; M files last mark completion)."""
    sdir.mkdir(parents=True, exist_ok=True)
    for c in range(1, 23):
        if (sdir / f"s.{c}.l2.M_5_50").exists():
            continue
        L = pd.read_csv(f"{prefix_ld}.{c}.l2.ldscore.gz", sep="\t")
        names = [n for n in L.columns if n not in ("CHR", "SNP", "BP")]
        idx = [names.index(f"{k}L2") if f"{k}L2" in names else names.index(k) for k in keep]
        gz = {"method": "gzip", "compresslevel": 1}
        L[["CHR", "SNP", "BP"] + [names[i] for i in idx]].to_csv(sdir / f"s.{c}.l2.ldscore.gz", sep="\t", index=False,
                                                                   compression=gz)
        A = pd.read_csv(f"{prefix_annot}.{c}.annot.gz", sep="\t", usecols=["CHR", "BP", "SNP", "CM"] + keep)
        A.to_csv(sdir / f"s.{c}.annot.gz", sep="\t", index=False, compression=gz)
        for suf in ("l2.M", "l2.M_5_50"):
            m = Path(f"{prefix_ld}.{c}.{suf}").read_text().split()
            (sdir / f"s.{c}.{suf}").write_text("\t".join(m[i] for i in idx) + "\n")


def jobs():
    meta = pd.read_csv(OUT / "annot_meta.tsv", sep="\t")
    J = []
    for a in meta.annotation:                                     # new annotations, AllPeaks control
        J.append(("AllPeaks control", a, OUT / "ldscore/ra_rx2", OUT / "annot/ra_rx2", ["AllPeaks", a]))
    for a in PRIMARY:                                             # primary annotations, baseline only
        J.append(("baseline only", a, R1 / "ldscore/ra_rx", R1 / "annot/ra_rx", [a]))
    return J


def stage_h2(workers=int(os.environ.get("H2_WORKERS", 4))):   # ~3.5 GB RAM per job; lower when memory is shared
    bpq = R1 / "baseline_pq"
    tasks = []
    for model, a, pld, pan, keep in jobs():
        sdir = OUT / "sets" / f"{a}__{'ctrl' if model.startswith('AllPeaks') else 'base'}"
        subset(pld, pan, keep, sdir)
        for t in TRAITS:
            tasks.append((model, a, t, sdir))
    rdir = OUT / "h2"
    rdir.mkdir(parents=True, exist_ok=True)

    def one(job):
        model, a, t, sdir = job
        out = rdir / f"{sdir.name}__{t}"
        if not Path(f"{out}.results").exists():
            run([LDSC_PY, LDSC, "--h2", R1 / f"sumstats/{t}.sumstats.gz", "--ref-ld-chr", f"{bpq}/baselineLD.,{sdir}/s.",
                 "--w-ld-chr", f"{W}/weights.hm3_noMHC.", "--overlap-annot", "--frqfile-chr", f"{FRQ}/1000G.EUR.QC.",
                 "--print-coefficients", "--out", out], log=f"{out}.stdout")
        return out
    rank = {t: i for i, t in enumerate(TRAITS)}
    tasks.sort(key=lambda j: (rank[j[2]], j[0] != "AllPeaks control"))
    with ThreadPoolExecutor(workers) as ex:
        for o in ex.map(one, tasks):
            print(f"h2 {o.name}", flush=True)


def stage_h2rev(workers=int(os.environ.get("H2_WORKERS", 2))):
    """Second runner for stage_h2 that works from the end of the same task list, so that two processes share the
    queue; a task whose log already exists (started by the other runner) is skipped."""
    bpq = R1 / "baseline_pq"
    rdir = OUT / "h2"
    tasks = []
    for model, a, pld, pan, keep in jobs():
        sdir = OUT / "sets" / f"{a}__{'ctrl' if model.startswith('AllPeaks') else 'base'}"
        for t in TRAITS:
            tasks.append((model, a, t, sdir))
    rank = {t: i for i, t in enumerate(TRAITS)}
    tasks.sort(key=lambda j: (rank[j[2]], j[0] != "AllPeaks control"), reverse=True)

    def one(job):
        model, a, t, sdir = job
        out = rdir / f"{sdir.name}__{t}"
        if Path(f"{out}.results").exists() or Path(f"{out}.stdout").exists() or Path(f"{out}.log").exists():
            return None
        run([LDSC_PY, LDSC, "--h2", R1 / f"sumstats/{t}.sumstats.gz", "--ref-ld-chr", f"{bpq}/baselineLD.,{sdir}/s.",
             "--w-ld-chr", f"{W}/weights.hm3_noMHC.", "--overlap-annot", "--frqfile-chr", f"{FRQ}/1000G.EUR.QC.",
             "--print-coefficients", "--out", out], log=f"{out}.stdout")
        return out
    with ThreadPoolExecutor(workers) as ex:
        for o in ex.map(one, tasks):
            if o is not None:
                print(f"h2rev {o.name}", flush=True)


def stage_h2extra(workers=int(os.environ.get("H2_WORKERS", 2))):
    """Reconciliation with Gupta et al. 2023 (cell-state-invariant fibroblast peaks enriched for RA, conditional on the
    baseline model only): the accessible half of each in vivo compartment refitted on baselineLD v2.2 alone, for RA and
    height (Gupta's invariant peaks were enriched for height too). Same h2/ directory, model = 'baseline only'."""
    bpq = R1 / "baseline_pq"
    tasks = []
    for c in ("TNK", "Bplasma", "Myeloid", "Endothelial", "Stromal"):
        a = f"iv_{c}_acc"
        sdir = OUT / "sets" / f"{a}__base"
        subset(OUT / "ldscore/ra_rx2", OUT / "annot/ra_rx2", [a], sdir)
        tasks += [(a, t, sdir) for t in ("ra_eur", "fg_height")]
    rdir = OUT / "h2"

    def one(job):
        a, t, sdir = job
        out = rdir / f"{sdir.name}__{t}"
        if not Path(f"{out}.results").exists():
            run([LDSC_PY, LDSC, "--h2", R1 / f"sumstats/{t}.sumstats.gz", "--ref-ld-chr", f"{bpq}/baselineLD.,{sdir}/s.",
                 "--w-ld-chr", f"{W}/weights.hm3_noMHC.", "--overlap-annot", "--frqfile-chr", f"{FRQ}/1000G.EUR.QC.",
                 "--print-coefficients", "--out", out], log=f"{out}.stdout")
        return out
    tasks.sort(key=lambda j: j[1] != "ra_eur")
    with ThreadPoolExecutor(workers) as ex:
        for o in ex.map(one, tasks):
            print(f"h2 {o.name}", flush=True)


# ------------------------------------------------------------------------------------------------ fine-mapping
def stage_finemap():
    """PIP-weighted share of fine-mapped RA variants (Ishigaki 2022 ST-6, hg19) inside each annotation."""
    d = pd.read_excel(ISHIGAKI, sheet_name="ST-6", header=5)
    d = d[pd.to_numeric(d["PIP"], errors="coerce") > 0.1].copy()       # credible-set variants: 'Variant ID.1'
    pos = d["Variant ID.1"].astype(str).str.extract(r"^(\d+):(\d+):")
    d["CHR"], d["BP"] = pd.to_numeric(pos[0]), pd.to_numeric(pos[1])   # GRCh37, as the 1000G reference
    d = d.dropna(subset=["CHR", "BP"])
    rows = []
    for c, g in d.groupby("CHR"):
        c = int(c)
        A1 = pd.read_csv(R1 / f"annot/ra_rx.{c}.annot.gz", sep="\t")
        A2 = pd.read_csv(OUT / f"annot/ra_rx2.{c}.annot.gz", sep="\t").drop(columns=["CHR", "BP", "CM", "AllPeaks"])
        A = A1.merge(A2, on="SNP", how="left")
        by_rs, by_bp = A.drop_duplicates("SNP").set_index("SNP"), A.drop_duplicates("BP").set_index("BP")
        cols = [k for k in A.columns if k not in ("CHR", "BP", "SNP", "CM")]
        for _, v in g.iterrows():
            rs, bp = str(v["Rs ID.1"]), int(v["BP"])
            h = by_rs.loc[rs] if rs in by_rs.index else (by_bp.loc[bp] if bp in by_bp.index else None)
            rows.append(dict(variant=v["Variant ID.1"], rsid=rs, pip=float(v["PIP"]), in_1000g=h is not None,
                             **{k: (float(h[k]) if h is not None else np.nan) for k in cols}))
    F = pd.DataFrame(rows)
    tot = {}
    for c in range(1, 23):
        A1 = pd.read_csv(R1 / f"annot/ra_rx.{c}.annot.gz", sep="\t")
        A2 = pd.read_csv(OUT / f"annot/ra_rx2.{c}.annot.gz", sep="\t")
        for k in A1.columns[4:]:
            tot[k] = tot.get(k, [0, 0]); tot[k][0] += int(A1[k].sum()); tot[k][1] += len(A1)
        for k in A2.columns[5:]:
            tot[k] = tot.get(k, [0, 0]); tot[k][0] += int(A2[k].sum()); tot[k][1] += len(A2)
    m = F[F.in_1000g]
    S = []
    for k, (n1, n) in tot.items():
        if k not in m.columns:
            continue
        w = m.pip.sum()
        S.append(dict(annotation=k, n_variants=len(m), pip_total=w, pip_in=float((m.pip * m[k].fillna(0)).sum()),
                      share_pip=float((m.pip * m[k].fillna(0)).sum() / w), share_snps=n1 / n,
                      n_variants_in=int(m[k].fillna(0).sum())))
    S = pd.DataFrame(S)
    S["fold"] = S.share_pip / S.share_snps
    S.to_csv(OUT / "finemap_overlap.tsv", sep="\t", index=False)
    F.to_csv(OUT / "finemap_variants.tsv", sep="\t", index=False)
    print(f"{len(F)} variants PIP > 0.1 ({F.in_1000g.sum()} in 1000G EUR)")
    print(S.round(4).to_string(index=False))


def stage_collect():
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    rows = []
    for f in sorted((OUT / "h2").glob("*.results")):
        name, t = f.stem.rsplit("__", 1)
        a, model = name.rsplit("__", 1)
        R = pd.read_csv(f, sep="\t")
        r = R[R.Category.str.replace(r"L2_\d+$", "", regex=True) == a].iloc[0]
        z = float(r["Coefficient_z-score"])
        rows.append(dict(annotation=a, model="AllPeaks control" if model == "ctrl" else "baseline only", trait=t,
                         prop_snps=float(r["Prop._SNPs"]), prop_h2=float(r["Prop._h2"]),
                         prop_h2_se=float(r["Prop._h2_std_error"]), enrichment=float(r["Enrichment"]),
                         enrichment_se=float(r["Enrichment_std_error"]), enrichment_p=float(r["Enrichment_p"]),
                         coef=float(r["Coefficient"]), coef_se=float(r["Coefficient_std_error"]), coef_z=z,
                         coef_p=float(norm.sf(z))))
    D = pd.DataFrame(rows)
    D["coef_fdr"] = D.groupby(["model", "trait"]).coef_p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
    ds = ["GSE243917", "GSE163548", "Roadmap E034/E032", "Zenodo 10515792", "Ishigaki 2022", "FinnGen R13"]
    write_table(pd.read_csv(OUT / "annot_meta.tsv", sep="\t"), "sldsc_annotations_r2b", script=__file__, datasets=ds[:3],
                note="S-LDSC sensitivity annotations: FLS active per line, Roadmap T/B active, in vivo accessible half, "
                     "fibroblast sub-compartments")
    write_table(D, "sldsc_chromatin_r2b", script=__file__, datasets=ds,
                note="S-LDSC sensitivity: new annotations with AllPeaks control; primary annotations with baseline only; "
                     "one-sided coefficient P, BH-FDR within model x trait")
    if (OUT / "finemap_overlap.tsv").exists():
        write_table(pd.read_csv(OUT / "finemap_overlap.tsv", sep="\t"), "finemap_overlap", script=__file__,
                    datasets=["Ishigaki 2022 ST-6", "GSE243917", "GSE163548", "Roadmap"],
                    note="PIP-weighted share of fine-mapped RA variants (PIP > 0.1, 1000G EUR) in each annotation")
    pd.set_option("display.width", 220)
    print(D.pivot_table(index=["model", "annotation"], columns="trait", values="coef_p").map(lambda x: f"{x:.2g}"))
    print(D.pivot_table(index=["model", "annotation"], columns="trait", values="enrichment").round(2))


if __name__ == "__main__":
    {"annot": stage_annot, "l2": stage_l2, "h2": stage_h2, "h2extra": stage_h2extra, "h2rev": stage_h2rev, "finemap": stage_finemap,
     "collect": stage_collect}[sys.argv[1]]()
