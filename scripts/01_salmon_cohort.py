"""01_salmon_cohort.py — stream-quantify an ArrayExpress bulk RNA-seq cohort with Salmon.

For each run: download reads -> salmon quant -> keep quant.sf -> delete reads.
  --source ena : paired FASTQ.gz from ENA
  --source aws : NCBI SRA Open Data (.sra on S3, without any HTTP proxy) -> fasterq-dump
                 --split-3
  --source sra-stream: first --max-pairs spots via fastq-dump -X from NCBI (direct), plain FASTQ. Same subsample
                 definition as ena-sub (first N spots of the run), so routes can be mixed via --shard. Throughput
                 knobs: --shard k/n (parallel download streams; throttling is per connection), --salmon-slots and
                 --threads (CPU), --ready-cap (transient disk).
  --source ena-sub (default): only the first --max-pairs read pairs (20M) of each ENA FASTQ.gz, fetched as byte
                 ranges in parallel 64 MB chunks (robust to per-connection throttling). R1/R2 are in identical order, so the first N records of each stay paired.
                 Samples with <= N pairs are taken whole. Declared in Methods as "first 20 million read pairs".
A downloader thread keeps at most LOOKAHEAD samples on disk so download and quantification overlap.
Resumable: runs whose quant.sf already exists are skipped.

Usage:
    python scripts/01_salmon_cohort.py E-MTAB-11611 [--threads 8] [--lookahead 2] [--limit N]

Outputs: db/objects/salmon/<accession>/<sample_id>/quant.sf (+ logs, lib_format_counts.json)
Gene-level aggregation + DuckDB registration happen in 02_tximport_register.py.
"""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import traceback
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.executable).parent
INDEX = ROOT / "db" / "objects" / "salmon_index_gencode_v46"
FASTQ_TMP = ROOT / "raw" / "fastq_tmp"


def run_table(acc: str, tissue: str | None = None) -> pd.DataFrame:
    sdrf = pd.read_csv(ROOT / "raw" / "meta" / f"{acc}.sdrf.txt", sep="\t")
    if tissue:                                   # e.g. PEAC deposits synovium + blood under one accession
        sdrf = sdrf[sdrf["Characteristics[organism part]"].str.contains(tissue, case=False, na=False)]
    sid_col = next(c for c in ["Characteristics[sample ID]", "Characteristics[sampleid]", "Source Name"] if c in sdrf)
    sdrf = sdrf.drop_duplicates("Comment[ENA_RUN]")[[sid_col, "Comment[ENA_RUN]"]]
    sdrf.columns = ["sample_id", "run"]
    ena = pd.read_csv(ROOT / "raw" / "meta" / f"{acc}.ena_runs.tsv", sep="\t")
    ena = ena.rename(columns={"run_accession": "run"})
    # a few runs have no ENA-processed FASTQ; fall back to the submitter's files (no md5 check then)
    miss = ena.fastq_ftp.isna()
    ena.loc[miss, "fastq_ftp"] = ena.loc[miss, "submitted_ftp"]
    ena.loc[miss, "fastq_md5"] = ""
    ena.loc[miss, "fastq_bytes"] = ena.loc[miss, "submitted_bytes"] if "submitted_bytes" in ena else None
    df = sdrf.merge(ena[["run", "fastq_ftp", "fastq_md5", "fastq_bytes", "read_count"]], on="run", how="left")
    if df.fastq_ftp.isna().any():
        raise SystemExit(f"{df.fastq_ftp.isna().sum()} runs lack any FASTQ link")
    df["sample_id"] = df.sample_id.astype(str).str.replace(" ", "_")
    return df


PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy")


def aria2(url: str, dest: Path, out: str, md5: str = "", direct: bool = False) -> Path:
    cmd = ["aria2c", "-c", "-x", "16", "-s", "16", "-k", "4M", "--file-allocation=none",
           "--max-tries=50", "--retry-wait=5", "--lowest-speed-limit=1M", "--console-log-level=error",
           "--summary-interval=0", "--download-result=hide", *([f"--checksum=md5={md5}"] if md5 else []),
           "-d", str(dest), "-o", out, url]
    # --lowest-speed-limit aborts stalled transfers (exit 5); resume with fresh connections instead of giving up
    # direct=True ignores HTTP proxy variables for this process only
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS} if direct else None
    for _ in range(30):
        if subprocess.run(cmd, env=env).returncode == 0:
            return dest / out
        time.sleep(10)
    raise subprocess.CalledProcessError(1, cmd)


def download_aws(row, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    sra = aria2(f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{row.run}/{row.run}", dest, f"{row.run}.sra",
                direct=True)
    subprocess.run([str(BIN / "fasterq-dump"), str(sra), "--split-3", "-e", "6", "-O", str(dest),
                    "-t", str(dest / "tmp"), "--skip-technical"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sra.unlink()
    shutil.rmtree(dest / "tmp", ignore_errors=True)
    fq = [dest / f"{row.run}_1.fastq", dest / f"{row.run}_2.fastq"]
    if not all(f.exists() for f in fq):
        raise subprocess.CalledProcessError(1, "fasterq-dump: missing paired output")
    return fq


CHUNK = 64 << 20


def _range_chunk(url: str, a: int, b: int, part: Path) -> None:
    """Fetch bytes [a, b] into part, resuming on stalls (curl aborts < 50 kB/s for 30 s)."""
    for _ in range(60):
        have = part.stat().st_size if part.exists() else 0
        if have >= b - a + 1:
            return
        r = subprocess.run(["curl", "-sS", "--fail", "-r", f"{a + have}-{b}", "--speed-limit", "50000",
                            "--speed-time", "30", "--max-time", "600", url],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        with open(part, "ab") as fh:
            fh.write(r.stdout)
        if r.returncode not in (0, 28, 18, 56):
            time.sleep(5)
    raise subprocess.CalledProcessError(1, f"range {a}-{b} of {url}")


def fetch_range(url: str, start: int, end: int, out: Path, workers: int = 6) -> None:
    """Append bytes [start, end] of url to out, fetched as parallel 64 MB chunks."""
    from concurrent.futures import ThreadPoolExecutor
    spans = [(a, min(a + CHUNK - 1, end)) for a in range(start, end + 1, CHUNK)]
    parts = [out.with_name(f"{out.name}.part{i:04d}") for i in range(len(spans))]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda x: _range_chunk(url, *x[0], x[1]), zip(spans, parts)))
    with open(out, "ab") as fo:
        for p in parts:
            fo.write(p.read_bytes())
            p.unlink()


def _records(gz: Path, max_pairs: int, fq: Path) -> int:
    # truncated gzip ends with an error by design; keep at most 4 * max_pairs complete lines
    subprocess.run(f"gzip -dc '{gz}' 2>/dev/null | head -n {4 * max_pairs} > '{fq}'", shell=True)
    n = sum(1 for _ in open(fq))
    return n // 4


def download_ena_sub(row, dest: Path, max_pairs: int) -> list[Path]:
    """First max_pairs records of R1 and R2 (whole files when the run is smaller). The byte prefix is grown until it
    decompresses to >= max_pairs records, so no assumption about compressed bytes per read is needed."""
    dest.mkdir(parents=True, exist_ok=True)
    urls = ["https://" + u for u in row.fastq_ftp.split(";")]
    sizes = [int(float(x)) for x in str(row.fastq_bytes).split(";")]
    fqs, got = [], []
    for i, (u, size) in enumerate(zip(urls, sizes), start=1):
        gz, fq = dest / f"R{i}.prefix.gz", dest / f"R{i}.fastq"
        gz.unlink(missing_ok=True)
        have, want = 0, min(size, 800 << 20)
        while True:
            fetch_range(u, have, want - 1, gz)
            have = want
            n = _records(gz, max_pairs, fq)
            if n >= max_pairs or have >= size:
                break
            want = min(size, int(have * max_pairs / max(n, 1) * 1.05) + (16 << 20))
        gz.unlink()
        fqs.append(fq)
        got.append(n)
    # the last record of a truncated prefix may be partial; trim both mates to the same complete count
    k = min(got) - 1 if min(got) < max_pairs else max_pairs
    for fq in fqs:
        subprocess.run(f"head -n {4 * k} '{fq}' > '{fq}.tmp' && mv '{fq}.tmp' '{fq}'", shell=True, check=True)
    return fqs


def _n_lines(f: Path) -> int:
    cmd = f"gzip -dc '{f}' | wc -l" if f.suffix == ".gz" else f"wc -l < '{f}'"
    return int(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout or 0)


def _complete_pair(dest: Path, run: str, want: int):
    """Return an existing, complete R1/R2 pair (plain or gzip) left by an earlier run, else None."""
    for ext in (".fastq", ".fastq.gz"):
        fq = [dest / f"{run}_1{ext}", dest / f"{run}_2{ext}"]
        if (dest / "READY").exists() and all(f.exists() for f in fq):
            return fq
        if all(f.exists() for f in fq) and not any(dest.glob("*.tmp*")):
            n1, n2 = _n_lines(fq[0]), _n_lines(fq[1])
            if n1 == n2 and n1 >= 4 * want * 0.999:
                (dest / "READY").touch()
                return fq
    return None


def download_sra_stream(row, dest: Path, max_pairs: int, gzip: bool = False) -> list[Path]:
    """First max_pairs spots streamed from NCBI SRA with fastq-dump -X (without any HTTP proxy).
    Plain FASTQ by default (gzip costs ~35% CPU per stream and Salmon must decompress again); transient disk is
    bounded by wait_for_queue_room(). A READY marker flags a validated, complete pair for reuse after restarts."""
    dest.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    want = min(max_pairs, int(row.read_count))
    if (fq := _complete_pair(dest, row.run, want)):
        return fq
    for _ in range(5):
        for f in dest.glob("*.fastq*"):
            f.unlink()
        subprocess.run([str(BIN / "fastq-dump"), "-X", str(max_pairs), "--split-files", "--skip-technical",
                        *(["--gzip"] if gzip else []), "-O", str(dest), row.run],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if (fq := _complete_pair(dest, row.run, want)):
            return fq
        time.sleep(30)
    raise subprocess.CalledProcessError(1, f"fastq-dump -X {max_pairs} {row.run}")


def wait_for_queue_room(acc: str, cap: int) -> None:
    """Machine-wide cap on downloaded-but-not-yet-quantified samples (bounds transient disk across shards)."""
    while len(list((FASTQ_TMP / acc).glob("*/READY"))) >= cap:
        time.sleep(30)


def download(row, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    urls = ["https://" + u for u in row.fastq_ftp.split(";")]
    md5s = str(row.fastq_md5 or "").split(";") if row.fastq_md5 else [""] * len(urls)
    files = []
    for u, m in zip(urls, md5s):
        files.append(aria2(u, dest, u.rsplit("/", 1)[1], m))
    return files


SLOTS = ROOT / "db" / "objects" / "salmon" / ".slots"


def acquire_slot(n_slots: int):
    """Machine-wide limit on concurrent Salmon runs across all shard processes (flock on slot files)."""
    import fcntl
    SLOTS.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(n_slots):
            fh = open(SLOTS / f"slot{i}.lock", "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fh                      # released when closed (or when the process exits)
            except BlockingIOError:
                fh.close()
        time.sleep(15)


def quant(row, fq: list[Path], outdir: Path, threads: int, n_slots: int = 3) -> None:
    slot = acquire_slot(n_slots)
    try:
        _quant(fq, outdir, threads)
    finally:
        slot.close()


def _quant(fq: list[Path], outdir: Path, threads: int) -> None:
    tmp = outdir.with_name(outdir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run([str(BIN / "salmon"), "quant", "-i", str(INDEX), "-l", "A",
                    "-1", str(fq[0]), "-2", str(fq[1]), "-p", str(threads),
                    "--validateMappings", "--gcBias", "--seqBias", "-o", str(tmp)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for junk in ["aux_info/bootstrap", "libParams"]:
        shutil.rmtree(tmp / junk, ignore_errors=True)
    tmp.rename(outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accession")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", choices=["ena-sub", "sra-stream", "aws", "ena"], default="ena-sub")
    ap.add_argument("--shard", default="0/1", help="k/n: process only runs with index %% n == k (parallel routes)")
    ap.add_argument("--max-pairs", type=int, default=20_000_000)
    ap.add_argument("--salmon-slots", type=int, default=3, help="max concurrent Salmon runs across all shards")
    ap.add_argument("--tissue", default=None, help="keep runs whose SDRF organism part matches (e.g. synovi)")
    ap.add_argument("--priority", default=None,
                    help="regex on sample_id; matching runs are processed first (e.g. 'week0$' = baseline biopsies)")
    ap.add_argument("--claim", action="store_true",
                    help="dynamic work queue: each worker claims the next unclaimed run (mkdir lock); run many workers")
    ap.add_argument("--ready-cap", type=int, default=4,
                    help="max downloaded samples waiting for Salmon, machine-wide (bounds transient disk)")
    a = ap.parse_args()

    out_root = ROOT / "db" / "objects" / "salmon" / a.accession
    out_root.mkdir(parents=True, exist_ok=True)
    df = run_table(a.accession, a.tissue)
    if a.priority:
        df = pd.concat([df[df.sample_id.str.contains(a.priority)], df[~df.sample_id.str.contains(a.priority)]])
    k, n = map(int, a.shard.split("/"))
    todo = [r for i, r in enumerate(df.itertuples())
            if i % n == k and not (out_root / r.sample_id / "quant.sf").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{a.accession}: {len(df)} runs, {len(todo)} to do", flush=True)

    q: queue.Queue = queue.Queue(maxsize=a.lookahead)

    claims = FASTQ_TMP / a.accession / "_claims"
    claims.mkdir(parents=True, exist_ok=True)

    def claim(r) -> bool:
        if not a.claim:
            return True
        if (out_root / r.sample_id / "quant.sf").exists():
            return False
        try:
            (claims / r.run).mkdir()
            return True
        except FileExistsError:
            return False

    def producer():
        try:
            _produce()
        except BaseException:                  # never leave the consumer blocked on q.get() (stalled 6 h, 09-27)
            traceback.print_exc()
            sys.stdout.flush()
        finally:
            q.put(None)

    def _produce():
        for r in todo:
            if not claim(r):
                continue
            try:
                d = FASTQ_TMP / a.accession / r.run
                if a.source == "sra-stream" and not (d / "READY").exists():
                    wait_for_queue_room(a.accession, a.ready_cap)
                fq = (download_ena_sub(r, d, a.max_pairs) if a.source == "ena-sub" else
                      download_sra_stream(r, d, a.max_pairs) if a.source == "sra-stream" else
                      download_aws(r, d) if a.source == "aws" else download(r, d))
                q.put((r, fq))
            except Exception as e:                                   # incl. OSError / network, not just exit codes
                print(f"DOWNLOAD FAIL {r.run} {r.sample_id}: {type(e).__name__}: {e}", flush=True)
                shutil.rmtree(FASTQ_TMP / a.accession / r.run, ignore_errors=True)
                shutil.rmtree(claims / r.run, ignore_errors=True)      # let another worker retry it

    threading.Thread(target=producer, daemon=True).start()
    done = 0
    while (item := q.get()) is not None:
        r, fq = item
        try:
            quant(r, fq, out_root / r.sample_id, a.threads, a.salmon_slots)
            done += 1
            print(f"[{done}/{len(todo)}] {r.sample_id} ({r.run}) ok", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"QUANT FAIL {r.run} {r.sample_id}: {e}", flush=True)
        finally:
            shutil.rmtree(FASTQ_TMP / a.accession / r.run, ignore_errors=True)
    print(f"ALLDONE {done}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
