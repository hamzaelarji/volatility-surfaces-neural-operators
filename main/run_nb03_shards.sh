#!/usr/bin/env bash
# NB03 — derivative-constrained deep smoother, FULL production run.
#
# NB03 is sequential per process, so a single process would use one core out of
# the 128 on lame25. The notebook already supports NB03_SHARD, so we run one
# process per year in parallel. NB05 reads the per-day .npz packs with a glob,
# so the sharding is completely transparent downstream.
#
# Run this INSIDE tmux on a CPU node (lame25 / lame26 / lamedell18):
#   ssh lame25
#   cd ~/thesis && tmux new -s nb03
#   ./run_nb03_shards.sh
#   ctrl+B then D     (detach; you can now close your laptop)
#
# Reattach later with:  ssh lame25 && tmux a -t nb03

set -u
cd "$(dirname "$0")" || exit 1
source ./env_thesis.sh

YEARS="${YEARS:-2018 2019 2020 2021 2022 2023 2024 2025}"

# Each shard is single-threaded at the Python level, but numpy/BLAS will happily
# grab every core and thrash when 8 shards run at once. Pin them.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# no GPU needed here: hide it so we never take a slot another student needs
export CUDA_VISIBLE_DEVICES=""

# ---- production settings -----------------------------------------------------
export NB03_LIMIT_DATES=None    # every trading day in the window
export NB03_EXPORT_ALL=1        # one .npz pack per day  (NB05's input)
export NB03_LAM0_STRIDE=5       # lambda=0 dirty counterfactual every 5th day
export NB03_CKPT_EVERY=25       # incremental checkpoint: a crash costs <= 25 days
export NB03_RUN_LAM0_CHECK=1
export NB03_SKIP_SYNTHETIC=1    # sections 3/3b/3c/4 are shard-independent (see below)

echo "=============================================================="
echo " NB03 production run"
echo " root    : $THESIS_ROOT"
echo " years   : $YEARS"
echo " figures : $THESIS_FIG_DIR"
echo "=============================================================="

pids=()
for y in $YEARS; do
  (
    export NB03_SHARD="$y"
    export NB03_DATE_START="${y}-01-01"
    export NB03_DATE_END="${y}-12-31"
    export THESIS_FIG_PREFIX="nb03"
    run_nb "03_deep_smoother.ipynb" "03_deep_smoother_${y}_executed" \
           "$THESIS_ROOT/logs/nb03_${y}.log"
  ) &
  pids+=($!)
  echo "  launched shard $y (pid ${pids[-1]})"
  sleep 3          # stagger: avoids 8 processes reading the parquet at once
done

# The synthetic / stress sections (3, 3b, 3c, 4) are identical in every shard, so
# they are skipped above. Run them ONCE, on a short window, to produce the
# stress-test figures the thesis cites.
(
  export NB03_SHARD="synth"
  export NB03_SKIP_SYNTHETIC=0
  export NB03_LIMIT_DATES=2
  export NB03_EXPORT_ALL=0
  export NB03_LAM0_STRIDE=0
  export NB03_RUN_LAM0_CHECK=1
  export THESIS_FIG_PREFIX="nb03synth"
  run_nb "03_deep_smoother.ipynb" "03_deep_smoother_synth_executed" \
         "$THESIS_ROOT/logs/nb03_synth.log"
) &
pids+=($!)
echo "  launched synthetic/stress pass (pid ${pids[-1]})"

echo
echo "waiting for ${#pids[@]} processes ..."
fail=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "  !! pid $p exited non-zero"; fail=1; }
done

echo
echo "=============================================================="
if [ "$fail" -eq 0 ]; then
  echo " NB03 COMPLETE — all shards succeeded"
else
  echo " NB03 FINISHED WITH ERRORS — check $THESIS_ROOT/logs/nb03_*.log"
fi
echo " packs   : $(ls -1 "$THESIS_OUT_DIR/nb03_surfaces"/*.npz 2>/dev/null | wc -l) .npz files"
echo " figures : $(ls -1 "$THESIS_FIG_DIR"/nb03*.pdf 2>/dev/null | wc -l) pdf"
echo "=============================================================="
exit $fail
