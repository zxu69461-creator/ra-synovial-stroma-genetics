#!/usr/bin/env bash
# Cross-ancestry replication: EAS MAGMA -> scDRS (both atlases) -> MAGMA cell-type -> matrix.
cd "$(dirname "$0")/.."
set -e
P=${PYTHON:-python}
$P scripts/20_magma.py ra_eas ra_eas_noMHC
$P scripts/21_scdrs.py --h5ad db/objects/atlas/micheroli_states.h5ad --state-col state --tag eas \
   --datasets E-MTAB-11791 --traits ra_eas_noMHC ra_eas
$P scripts/21_scdrs.py --h5ad db/objects/atlas/gupta2023_states.h5ad --state-col state --tag gupta_eas \
   --datasets GSE243917 --traits ra_eas_noMHC ra_eas
$P scripts/23_magma_celltype.py --atlas micheroli
$P scripts/23_magma_celltype.py --atlas gupta
$P scripts/25_genetic_map_summary.py
echo EAS_PIPELINE_DONE
