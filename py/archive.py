"""ra_rx local archive: every result goes into db/ra_rx.duckdb with provenance.

Usage (from any script under ra_rx/scripts):
    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "py"))
    from archive import write_table, write_parquet_view, register_raw, manifest

    write_table(df, "scdrs_state", script=__file__, datasets=["E-MTAB-11611", "GCST90132222"], inputs=[path1, path2])
    write_parquet_view(big_df, "r4ra_salmon_gene_counts", script=__file__, datasets=["E-MTAB-11611"], inputs=[h5ad])
    manifest()          # list every archived asset
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db" / "ra_rx.duckdb"
PARQUET = ROOT / "db" / "parquet"
OBJECTS = ROOT / "db" / "objects"   # derived AnnData/Seurat objects (regenerable, registered in _provenance)

_DDL = """
CREATE TABLE IF NOT EXISTS _provenance (
    asset       VARCHAR PRIMARY KEY,
    kind        VARCHAR,          -- table | view | object
    script      VARCHAR,
    created_at  TIMESTAMP,
    n_rows      BIGINT,
    datasets    VARCHAR,          -- JSON list of accession / dataset ids
    inputs      VARCHAR,          -- JSON list of {path, sha256}
    note        VARCHAR
);
CREATE TABLE IF NOT EXISTS _raw_files (
    path        VARCHAR PRIMARY KEY,   -- relative to project root
    dataset     VARCHAR,
    url         VARCHAR,
    size_bytes  BIGINT,
    mtime       DOUBLE,
    sha256      VARCHAR,
    registered_at TIMESTAMP
);
"""


def connect(read_only: bool = False, wait_s: int = 600) -> duckdb.DuckDBPyConnection:
    """Open the project DB. Several background jobs share it and DuckDB allows one writer process at a time,
    so a held lock is retried (connections here are short-lived) instead of failing a multi-hour job."""
    import time
    DB.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            con = duckdb.connect(str(DB), read_only=read_only)
            break
        except duckdb.IOException as e:
            if "lock" not in str(e).lower() or time.time() - t0 > wait_s:
                raise
            time.sleep(3)
    if not read_only:
        con.execute(_DDL)
    return con


def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256(path, con=None) -> str:
    """SHA256 of a file, cached in _raw_files by (size, mtime) so multi-GB inputs hash once."""
    p = Path(path).resolve()
    st = p.stat()
    own = con is None
    con = con or connect()
    try:
        row = con.execute(
            "SELECT sha256 FROM _raw_files WHERE path=? AND size_bytes=? AND mtime=?",
            [_rel(p), st.st_size, st.st_mtime],
        ).fetchone()
        if row and row[0]:
            return row[0]
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        digest = h.hexdigest()
        con.execute(
            """INSERT INTO _raw_files (path, size_bytes, mtime, sha256, registered_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (path) DO UPDATE SET size_bytes=excluded.size_bytes,
                   mtime=excluded.mtime, sha256=excluded.sha256, registered_at=excluded.registered_at""",
            [_rel(p), st.st_size, st.st_mtime, digest, _dt.datetime.now()],
        )
        return digest
    finally:
        if own:
            con.close()


def register_raw(path, dataset: str, url: str | None = None) -> str:
    """Record a downloaded raw file (dataset id, source URL, SHA256)."""
    con = connect()
    try:
        digest = sha256(path, con)
        con.execute("UPDATE _raw_files SET dataset=?, url=? WHERE path=?", [dataset, url, _rel(path)])
        return digest
    finally:
        con.close()


def retire_raw(path, dataset: str | None = None, url: str | None = None, reason: str = "") -> None:
    """Free disk: record SHA256 (+ dataset/url if given), mark the raw file deleted with a reason, then delete it.
    The _raw_files row stays, so manifest(raw=True) still shows what was used and where to re-download it."""
    p = Path(path)
    if not p.exists():
        return
    con = connect()
    try:
        con.execute("ALTER TABLE _raw_files ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP")
        con.execute("ALTER TABLE _raw_files ADD COLUMN IF NOT EXISTS deleted_reason VARCHAR")
        sha256(p, con)
        if dataset or url:
            con.execute("UPDATE _raw_files SET dataset=coalesce(?, dataset), url=coalesce(?, url) WHERE path=?",
                        [dataset, url, _rel(p)])
        con.execute("UPDATE _raw_files SET deleted_at=?, deleted_reason=? WHERE path=?",
                    [_dt.datetime.now(), reason, _rel(p)])
    finally:
        con.close()
    p.unlink()


def _record(con, asset, kind, script, n_rows, datasets, inputs, note):
    inp = [{"path": _rel(p), "sha256": sha256(p, con)} for p in (inputs or [])]
    con.execute(
        """INSERT INTO _provenance VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (asset) DO UPDATE SET kind=excluded.kind, script=excluded.script,
               created_at=excluded.created_at, n_rows=excluded.n_rows, datasets=excluded.datasets,
               inputs=excluded.inputs, note=excluded.note""",
        [asset, kind, _rel(script), _dt.datetime.now(), n_rows,
         json.dumps(list(datasets or [])), json.dumps(inp), note],
    )


def write_table(df: pd.DataFrame, name: str, *, script, datasets=(), inputs=(), note: str = "") -> None:
    """Store a result table inside DuckDB (replaces an existing table of the same name)."""
    con = connect()
    try:
        con.register("_df", df)
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _df')
        con.unregister("_df")
        _record(con, name, "table", script, len(df), datasets, inputs, note)
    finally:
        con.close()


def write_parquet_view(df: pd.DataFrame | None, name: str, *, script, datasets=(), inputs=(),
                       note: str = "", parquet_path=None) -> Path:
    """Store a large matrix as Parquet under db/parquet and expose it as a DuckDB view.

    Pass df=None with parquet_path to register a Parquet file that was written elsewhere.
    """
    PARQUET.mkdir(parents=True, exist_ok=True)
    pq = Path(parquet_path) if parquet_path else PARQUET / f"{name}.parquet"
    if df is not None:
        df.to_parquet(pq, index=False)
    con = connect()
    try:
        con.execute(f"CREATE OR REPLACE VIEW \"{name}\" AS SELECT * FROM read_parquet('{pq.resolve()}')")
        n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        _record(con, name, "view", script, n, datasets, inputs, note)
    finally:
        con.close()
    return pq


def write_object(obj_path, name: str, *, script, datasets=(), inputs=(), note: str = "", n_rows=None) -> Path:
    """Register a derived analysis object (e.g. processed .h5ad) saved under db/objects/.

    The caller writes the file; this only checks it lives in db/objects and records provenance.
    """
    p = Path(obj_path).resolve()
    if OBJECTS not in p.parents:
        raise ValueError(f"objects must be saved under {OBJECTS}, got {p}")
    con = connect()
    try:
        _record(con, name, "object", script, n_rows, datasets, inputs, f"{_rel(p)} | {note}")
    finally:
        con.close()
    return p


def read(name_or_sql: str) -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        q = name_or_sql if " " in name_or_sql.strip() else f'SELECT * FROM "{name_or_sql}"'
        return con.execute(q).df()
    finally:
        con.close()


def manifest(raw: bool = False) -> pd.DataFrame:
    """List every archived asset (or raw files with raw=True)."""
    con = connect(read_only=DB.exists())
    try:
        if raw:
            cols = {r[0] for r in con.execute("DESCRIBE _raw_files").fetchall()}
            extra = ", deleted_at, deleted_reason" if "deleted_at" in cols else ""
            return con.execute(f"SELECT path, dataset, size_bytes, sha256, registered_at{extra} FROM _raw_files ORDER BY path").df()
        return con.execute(
            "SELECT asset, kind, n_rows, script, created_at, datasets, note FROM _provenance ORDER BY created_at"
        ).df()
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60)
    print(manifest(raw="--raw" in sys.argv).to_string())
