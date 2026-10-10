#!/usr/bin/env bash
# 00_download_sldsc.sh — inputs of the chromatin analyses (scripts 27 and 27b):
#   LD score references (Zenodo 10515792; 1000G Phase 3 EUR, hg19; MD5-verified), the HapMap3 SNP list, the UCSC
#   hg38->hg19 liftOver chain, cultured-FLS chromatin of Ge et al. 2021 (GSE163548), the snATAC peaks x nuclei matrix
#   of GSE243917, and PolyFun (Python 3 implementation of LDSC). Downloads are resumable; existing files are kept.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fetch() {  # url dest_dir [md5]
  local url=$1 dir="$ROOT/$2" md5=${3:-} name
  name=$(basename "${url%/content}"); mkdir -p "$dir"
  [ -s "$dir/$name" ] || curl -sS -L --retry 5 -C - -o "$dir/$name" "$url"
  if [ -n "$md5" ]; then
    python3 -c "import hashlib,sys; h=hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest(); sys.exit(h!=sys.argv[2])" \
      "$dir/$name" "$md5" || { echo "MD5 mismatch: $dir/$name" >&2; exit 1; }
  fi
  echo "ok $name"
}
Z=https://zenodo.org/api/records/10515792/files
fetch $Z/1000G_Phase3_weights_hm3_no_MHC.tgz/content raw/ldsc a98ac0f089ee285177544a3e6e721ca3
fetch $Z/hm3_no_MHC.list.txt/content raw/ldsc 65a34c68833eb4a764d0707b5505b508
fetch $Z/1000G_Phase3_frq.tgz/content raw/ldsc ac29686ffd5b6378789857a522ebca77
fetch $Z/1000G_Phase3_plinkfiles.tgz/content raw/ldsc a7773ab485827b533cb300c76356d76b
fetch $Z/1000G_Phase3_baselineLD_v2.2_ldscores.tgz/content raw/ldsc b261e0caf06a003e7522938e01b3d349
# HapMap3 SNP list (LDSC) and UCSC liftOver chain
fetch https://storage.googleapis.com/broad-alkesgroup-public/LDSCORE/w_hm3.snplist.bz2 refs/ldsc
[ -s "$ROOT/refs/ldsc/w_hm3.snplist" ] || bunzip2 -k "$ROOT/refs/ldsc/w_hm3.snplist.bz2"
fetch https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz refs
# Ge et al. 2021 cultured FLS: merged ATAC peaks and per-sample histone peaks (GEO RAW archive)
G=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE163nnn/GSE163548/suppl
fetch $G/GSE163548_Basal.merged_peaks.narrowPeak.gz raw/geo
fetch $G/GSE163548_Stim.merged_peaks.narrowPeak.gz raw/geo
[ -s "$ROOT/raw/geo/GSE163548_RAW.tar" ] || curl -sS -L --retry 5 -o "$ROOT/raw/geo/GSE163548_RAW.tar" \
  "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE163548&format=file"
# GSE243917 snATAC peaks x nuclei (Gupta et al. 2023 / Weinand et al. 2024)
fetch https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243917/suppl/GSE243917_peaksXcells_ALL_raw.h5ad.gz raw/sc/gupta2023
# PolyFun (commit used in the analysis)
if [ ! -d "$ROOT/tools/polyfun" ]; then
  git clone https://github.com/omerwe/polyfun.git "$ROOT/tools/polyfun"
  git -C "$ROOT/tools/polyfun" checkout 8b17b4ec87fde11f8d216c049c08d1420cd25de6
fi
echo ALL_DONE
