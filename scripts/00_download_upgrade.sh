#!/usr/bin/env bash
# Additional downloads: EAS GWAS + MAGMA EAS LD reference (cross-ancestry
# replication), GSE243917 (Gupta 2023 RA synovium multiome; RNA only) for atlas replication, Ishigaki 2022 supp.
cd "$(dirname "$0")/.."
mkdir -p raw/sc/gupta2023 raw/gwas tools/magma_eas refs/ishigaki2022_supp
set -x
G=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243917/suppl
for f in genesXcells_ALL_raw T_RNA_processed B_RNA_processed myeloid_RNA_processed fibroblast_RNA_processed endothelial_RNA_processed; do
  [ -s raw/sc/gupta2023/GSE243917_$f.h5ad.gz ] || curl -sS -L --retry 5 -C - -o raw/sc/gupta2023/GSE243917_$f.h5ad.gz $G/GSE243917_$f.h5ad.gz &
done
E=https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000/GCST90132224/harmonised
[ -s raw/gwas/Ishigaki2022_GCST90132224.h.tsv.gz ] || curl -sS -L --retry 5 -C - -o raw/gwas/Ishigaki2022_GCST90132224.h.tsv.gz $E/GCST90132224.h.tsv.gz &
[ -s tools/magma_eas/g1000_eas.zip ] || curl -sS -L --retry 5 -o tools/magma_eas/g1000_eas.zip "https://vu.data.surfsara.nl/index.php/s/dz6PYdKOi3xVqHn/download" &
S="https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-022-01213-w/MediaObjects"
for k in 1 2 3 4; do curl -sS -L -o refs/ishigaki2022_supp/41588_2022_1213_MOESM${k}_ESM.xlsx "$S/41588_2022_1213_MOESM${k}_ESM.xlsx" & done
wait
cd tools/magma_eas && unzip -o -q g1000_eas.zip && rm -f g1000_eas.zip
echo DOWNLOADS_DONE
