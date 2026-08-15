#!/usr/bin/env bash
# NB05 — downstream economics. Consumes the NB03 and NB04 .npz packs, so it must
# run LAST, once both are finished.
#
#   ssh lame25
#   cd ~/thesis && tmux new -s nb05
#   ./run_nb05.sh

set -u
cd "$(dirname "$0")" || exit 1
source ./env_thesis.sh

export CUDA_VISIBLE_DEVICES=""          # pure CPU notebook
export THESIS_FIG_PREFIX="nb05"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# ---- gate: NB05 sections skip silently on missing inputs, which would produce a
# ---- notebook full of "skipped" instead of an error. Check loudly instead.
n3=$(ls -1 "$THESIS_OUT_DIR/nb03_surfaces"/*.npz 2>/dev/null | wc -l)
n4=$(ls -1 "$THESIS_OUT_DIR/nb04_surfaces"/*.npz 2>/dev/null | wc -l)
echo "NB03 packs: $n3 | NB04 packs: $n4"
if [ "$n3" -eq 0 ] || [ "$n4" -eq 0 ]; then
  echo "REFUSING TO RUN: one of the pack directories is empty."
  echo "NB05 would 'succeed' with every section skipped. Finish NB03/NB04 first."
  exit 1
fi
if [ ! -f "$THESIS_VIX_RAW" ]; then
  echo "WARNING: $THESIS_VIX_RAW missing — the VIX replication section will skip."
fi

# ---- production settings -----------------------------------------------------
export NB05_F1_MAX_DAYS=100000     # scan every day, not a 250-day sample
export NB05_F2_HORIZONS="1,3,5,10"
export NB05_G_NREP=500             # detection frontier: tighter Monte Carlo
export NB05_MIN_SIGNAL_NAMES=30

echo "=============================================================="
echo " NB05 production run"
echo "=============================================================="

run_nb "05_downstream_economics.ipynb" "05_downstream_economics_executed" \
       "$THESIS_ROOT/logs/nb05.log"
rc=$?

echo
echo "=============================================================="
[ $rc -eq 0 ] && echo " NB05 COMPLETE" || echo " NB05 FAILED — see logs/nb05.log"
echo " figures : $(ls -1 "$THESIS_FIG_DIR"/nb05*.pdf 2>/dev/null | wc -l) pdf"
ls -1 "$THESIS_OUT_DIR"/nb05_*.parquet 2>/dev/null | sed 's/^/   /'
echo "=============================================================="
exit $rc
