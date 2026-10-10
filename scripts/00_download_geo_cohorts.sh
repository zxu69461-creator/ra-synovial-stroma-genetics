#!/usr/bin/env bash
# 00_download_geo_cohorts.sh — GEO synovial biopsy cohorts used as deposited (read by py/cohorts.py):
#   GSE198520 (anti-TNF; count matrix), GSE45867 (tocilizumab / methotrexate; GPL570), GSE24742 (rituximab; GPL570),
#   and the GPL570 annotation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; mkdir -p "$ROOT/raw/geo"; cd "$ROOT/raw/geo"
G=https://ftp.ncbi.nlm.nih.gov/geo
for u in $G/series/GSE198nnn/GSE198520/suppl/GSE198520_Raw_gene_count_matrix.txt.gz \
         $G/series/GSE198nnn/GSE198520/matrix/GSE198520_series_matrix.txt.gz \
         $G/series/GSE45nnn/GSE45867/matrix/GSE45867_series_matrix.txt.gz \
         $G/series/GSE24nnn/GSE24742/matrix/GSE24742_series_matrix.txt.gz \
         $G/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz; do
  f=$(basename "$u"); [ -s "$f" ] || curl -sS -L --retry 5 -C - -o "$f" "$u"; echo "ok $f"
done
