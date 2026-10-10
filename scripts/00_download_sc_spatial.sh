#!/usr/bin/env bash
# 00_download_sc_spatial.sh — open single-cell reference + Xenium spatial data (aria2c, resumable).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIST="$(mktemp)"
add(){ printf '%s\n  dir=%s\n  out=%s\n' "$1" "$2" "$3" >> "$LIST"; }

# Micheroli et al. 2024 iScience — RA synovium scRNA-seq atlas (E-MTAB-11791), filtered count matrix
B=https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/791/E-MTAB-11791/Files
for f in count_matrix_filtered.mtx barcodes_filtered.txt genes_filtered.txt E-MTAB-11791.sdrf.txt E-MTAB-11791.idf.txt; do
  add "$B/$f" "$ROOT/raw/sc/micheroli2024" "$f"; done

# Xenium 5k, 17 pre-treatment RA synovia (Zenodo 15058262, CC-BY-4.0; remission vs non-remission)
curl -s https://zenodo.org/api/records/15058262 | python3 -c "
import json,sys
for f in json.load(sys.stdin)['files']: print(f['links']['self'], f['key'], f['checksum'].split(':')[1])" |
while read url key md5; do printf '%s\n  dir=%s\n  out=%s\n  checksum=md5=%s\n' "$url" "$ROOT/raw/xenium/zenodo15058262" "$key" "$md5" >> "$LIST"; done

aria2c -i "$LIST" -c -j 3 -x 16 -s 16 -k 4M --auto-file-renaming=false --allow-overwrite=false \
       --file-allocation=none --retry-wait=10 --max-tries=20 --summary-interval=0 --console-log-level=warn
rc=$?; rm -f "$LIST"; echo "ALLDONE rc=$rc"
