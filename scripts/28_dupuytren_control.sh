#!/bin/bash
# Dupuytren's contracture (FinnGen R13 M13_DUPUTRYEN) as a fibroblast-disease positive control for the
# stromal compartment, run through the same gene-level pipeline as RA. scDRS goes to separate tagged tables (dup,
# gupta_dup) so the primary scDRS tables are untouched; magma_celltype is re-run with every default trait plus Dupuytren
# because 23_magma_celltype.py replaces all rows of the atlas it runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY scripts/20_magma.py fg_dupuytren
$PY scripts/21_scdrs.py --h5ad db/objects/atlas/micheroli_states.h5ad --state-col state --tag dup --traits fg_dupuytren --datasets E-MTAB-11791
$PY scripts/21_scdrs.py --h5ad db/objects/atlas/gupta2023_states.h5ad --state-col state --tag gupta_dup --traits fg_dupuytren --datasets GSE243917
for A in micheroli gupta; do
  $PY scripts/23_magma_celltype.py --atlas $A --traits ra_eur_noMHC ra_eur fg_ra_noMHC ra_eas_noMHC fg_oa fg_height fg_dupuytren
done
echo DUPUYTREN_DONE
