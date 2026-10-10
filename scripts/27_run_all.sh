#!/usr/bin/env bash
# 27_sldsc_chromatin.py end to end (annot -> l2 -> munge -> h2 -> collect); logs in logs/27_*.log.
# PYTHON: interpreter of the main environment; LDSC_PYTHON: interpreter of the LDSC environment (environment_ldsc.yml).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SCRATCH=${SCRATCH:-$PWD/tmp}; mkdir -p "$SCRATCH" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for st in annot l2 munge h2 collect; do
  $PY scripts/27_sldsc_chromatin.py $st > logs/27_$st.log 2>&1; rc=$?
  echo "stage $st exit=$rc"
  [ $rc -ne 0 ] && { tail -20 logs/27_$st.log; exit $rc; }
done
echo SLDSC_ALL_DONE
