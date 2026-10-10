#!/usr/bin/env bash
# 27b_sldsc_sensitivity.py end to end, after 27_run_all.sh; logs in logs/27b_*.log.
# H2_WORKERS sets the number of parallel LDSC regressions (about 3.5 GB of memory each).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SCRATCH=${SCRATCH:-$PWD/tmp}; mkdir -p "$SCRATCH" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for st in annot l2 h2 h2extra finemap collect; do
  $PY scripts/27b_sldsc_sensitivity.py $st > logs/27b_$st.log 2>&1; rc=$?
  echo "stage $st exit=$rc"
  [ $rc -ne 0 ] && { tail -20 logs/27b_$st.log; exit $rc; }
done
echo SLDSC_SENSITIVITY_DONE
