#!/usr/bin/env bash
# 00_download_refs.sh — public reference inputs (GWAS, scDRS gene sets, GENCODE) via aria2c.
# SHA256 + dataset ids are recorded in db/ra_rx.duckdb by scripts/00_register_raw.py afterwards.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIST="$(mktemp)"
add(){ printf '%s\n  dir=%s\n  out=%s\n' "$1" "$2" "$3" >> "$LIST"; }

# RA GWAS: Ishigaki et al. 2022 Nat Genet (harmonised GRCh38). 222 all-ancestry, 223 EUR, 224 EAS, 225-227 seropositive (all/EUR/EAS)
EBI="http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000"
for g in GCST90132222 GCST90132223 GCST90132224 GCST90132225 GCST90132226 GCST90132227; do
  add "$EBI/$g/harmonised/$g.h.tsv.gz" "$ROOT/raw/gwas" "Ishigaki2022_$g.h.tsv.gz"
  add "$EBI/$g/harmonised/md5sum.txt" "$ROOT/raw/gwas" "Ishigaki2022_${g}_md5sum.txt"
done
# FinnGen R13 (GRCh38): RA replication endpoints + OA contrast trait + height (non-immune negative control)
FG="https://storage.googleapis.com/finngen-public-data-r13/summary_stats"
for e in M13_RHEUMA RHEUMA_SEROPOS_STRICT RHEUMA_SERONEG M13_ARTHROSIS HEIGHT_IRN; do add "$FG/finngen_R13_$e.gz" "$ROOT/raw/gwas" "finngen_R13_$e.gz"; done
# scDRS data release (Zhang et al. 2022 Nat Genet; figshare 19312583): MAGMA gene sets incl. PASS_Rheumatoid_Arthritis
# figshare redirects to a signed S3 URL; resolve it first so aria2c can open parallel connections
fs(){ curl -sI "https://ndownloader.figshare.com/files/$1" | grep -i '^location' | tr -d '\r' | cut -d' ' -f2; }
add "$(fs 34300898)" "$ROOT/raw/scdrs" gs_file.zip
add "$(fs 34300892)" "$ROOT/raw/scdrs" gene_annotation.zip
# GENCODE v46 transcriptome + annotation for Salmon
GC="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46"
add "$GC/gencode.v46.transcripts.fa.gz" "$ROOT/raw/gencode" gencode.v46.transcripts.fa.gz
add "$GC/gencode.v46.annotation.gtf.gz" "$ROOT/raw/gencode" gencode.v46.annotation.gtf.gz

aria2c -i "$LIST" -c -j 4 -x 16 -s 16 -k 4M --auto-file-renaming=false --allow-overwrite=false \
       --file-allocation=none --retry-wait=10 --max-tries=20 --summary-interval=0 --console-log-level=warn
rc=$?; rm -f "$LIST"; echo "ALLDONE rc=$rc"
