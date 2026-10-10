"""store.py — download and cache helpers used by the S-LDSC scripts.

fetch(url, acc=None, name=None, md5=None, link=None)
    Download `url` once into raw/<acc>/<name> (or into `link` if given), resumably with curl, and return the
    local path. An existing non-empty file is reused. `md5`, if given, is verified.

cached(step, build, inputs=(), params=None, version="1")
    Content-addressed cache for deterministic intermediate steps (for example GWAS summary statistics reduced to
    HapMap3 SNPs). The key combines the step name, the size and modification time of every input file, the
    parameters and a version string. `build(outdir)` writes its files into a fresh directory, which is returned;
    later calls with the same key return it without rebuilding.

Locations can be changed with the environment variables RA_RX_RAW and RA_RX_CACHE.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("RA_RX_RAW", ROOT / "raw"))
CACHE = Path(os.environ.get("RA_RX_CACHE", ROOT / "cache"))


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, acc=None, name=None, md5=None, link=None, **_):
    name = name or os.path.basename(urllib.parse.urlparse(url).path) or "download"
    folder = Path(link) if link else RAW / (acc or urllib.parse.urlparse(url).netloc)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    if not (dest.exists() and dest.stat().st_size > 0):
        part = dest.with_name(dest.name + ".part")
        subprocess.run(["curl", "-sS", "-L", "--retry", "5", "-C", "-", "-o", str(part), url], check=True)
        part.rename(dest)
    if md5 and _md5(dest) != md5:
        raise ValueError(f"MD5 mismatch for {dest}")
    return str(dest)


def _key(step, inputs, params, version):
    parts = []
    for i in inputs:
        p = Path(i)
        parts.append([p.name, p.stat().st_size, int(p.stat().st_mtime)] if p.exists() else str(i))
    blob = json.dumps([step, parts, params or {}, version], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cached(step, build, inputs=(), params=None, version="1", **_):
    out = CACHE / step / _key(step, inputs, params, version)
    if (out / ".done").exists():
        return out
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    build(tmp)
    (tmp / ".done").write_text("")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out
