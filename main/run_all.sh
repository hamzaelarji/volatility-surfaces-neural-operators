#!/usr/bin/env bash
# ============================================================================
# run_all.sh — ONE launch, walk away, come back in 2-3 days to full results.
#
# Runs the whole pipeline SEQUENTIALLY on a single GPU node (gpu1/2/3):
#     smoke test  ->  NB03  ->  NB04 (GPU)  ->  NB05
# No second machine, no manual step between notebooks. If the smoke test fails,
# it stops immediately instead of wasting a day.
#
# USE A GPU NODE. NB04 needs CUDA; on a CPU node it would crawl for days.
#
#   ssh gpu1
#   cd ~/thesis && conda activate thesis
#   nvidia-smi                     # pick a free GPU index (usually 0)
#   tmux new -s allrun
#   CUDA_DEVICE=0 ./run_all.sh
#   ctrl+B then D                  # detach, close the laptop, leave for 3 days
#
# Reattach anytime:  ssh gpu1 && tmux a -t allrun
# Live progress:     tail -f ~/thesis/logs/run_all.log
# ============================================================================

set -u
cd "$(dirname "$0")" || exit 1
export THESIS_ROOT="${THESIS_ROOT:-/home/infres/$USER/thesis}"
source ./env_thesis.sh

MASTER_LOG="$THESIS_ROOT/logs/run_all.log"
GPU="${CUDA_DEVICE:-0}"

banner () { echo -e "\n\n########################################################################\n# $*\n########################################################################" | tee -a "$MASTER_LOG"; }
stamp  () { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

# gpu-node CPUs are shared (24 cores); the rule is a third. Cap BLAS so the
# CPU notebooks don't hog the box.
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

banner "PIPELINE START on $(hostname) — GPU $GPU — $(date)"

# ---------------------------------------------------------------------------
# 0. SMOKE TEST — tiny end-to-end run in a sandbox. Fail fast.
# ---------------------------------------------------------------------------
banner "STEP 0/4 — smoke test (a few minutes)"
if CUDA_DEVICE="$GPU" ./run_smoke_test.sh >>"$MASTER_LOG" 2>&1; then
    stamp "smoke test PASSED"
else
    stamp "smoke test FAILED — see $MASTER_LOG (search 'FAILED'). Nothing else run."
    exit 1
fi
rm -rf "$THESIS_ROOT/data/smoke"

# From here on, real production settings and real output directory.
export THESIS_OUT_DIR="$THESIS_ROOT/data/clean"
export THESIS_OPT_PARQUET="$THESIS_OUT_DIR/option_prices_clean.parquet"
export THESIS_FIG_DIR="$THESIS_OUT_DIR/figures"
mkdir -p "$THESIS_FIG_DIR"

# ---------------------------------------------------------------------------
# 1. NB03 — deep smoother, full history, sharded by year IN PARALLEL.
#    Even on 24 cores this beats one sequential process. BLAS pinned low so the
#    8 shards share the node politely.
# ---------------------------------------------------------------------------
banner "STEP 1/4 — NB03 deep smoother (all years)"
export CUDA_VISIBLE_DEVICES=""          # NB03 is CPU; never hold the GPU here
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
export NB03_LIMIT_DATES=None NB03_EXPORT_ALL=1 NB03_LAM0_STRIDE=5 \
       NB03_CKPT_EVERY=25 NB03_RUN_LAM0_CHECK=1 NB03_SKIP_SYNTHETIC=1

nb03_pids=()
for y in 2018 2019 2020 2021 2022 2023 2024 2025; do
  (
    export NB03_SHARD="$y" NB03_DATE_START="${y}-01-01" NB03_DATE_END="${y}-12-31" \
           THESIS_FIG_PREFIX="nb03"
    run_nb "03_deep_smoother.ipynb" "03_deep_smoother_${y}_executed" \
           "$THESIS_ROOT/logs/nb03_${y}.log"
  ) &
  nb03_pids+=($!); stamp "  NB03 shard $y launched (pid ${nb03_pids[-1]})"; sleep 3
done
# synthetic / stress figures once, short window
(
  export NB03_SHARD="synth" NB03_SKIP_SYNTHETIC=0 NB03_LIMIT_DATES=2 \
         NB03_EXPORT_ALL=0 NB03_LAM0_STRIDE=0 THESIS_FIG_PREFIX="nb03synth"
  run_nb "03_deep_smoother.ipynb" "03_deep_smoother_synth_executed" \
         "$THESIS_ROOT/logs/nb03_synth.log"
) &
nb03_pids+=($!); stamp "  NB03 synthetic pass launched (pid ${nb03_pids[-1]})"

nb03_fail=0
for p in "${nb03_pids[@]}"; do wait "$p" || nb03_fail=1; done
n3=$(ls -1 "$THESIS_OUT_DIR/nb03_surfaces"/*.npz 2>/dev/null | wc -l)
stamp "NB03 done — $n3 .npz packs — fail=$nb03_fail"
[ "$n3" -eq 0 ] && { stamp "FATAL: NB03 produced no packs. Stopping."; exit 1; }

# ---------------------------------------------------------------------------
# 2. NB04 — neural operators (incl. prior-embedded), on the GPU.
# ---------------------------------------------------------------------------
banner "STEP 2/4 — NB04 neural operators (GPU $GPU)"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python3 - <<'PY' >>"$MASTER_LOG" 2>&1
import sys
try:
    import torch
except ImportError:
    sys.exit("FATAL: torch not installed — install it on this GPU node first "
             "(pip install torch --index-url https://download.pytorch.org/whl/cu121)")
if not torch.cuda.is_available():
    sys.exit("FATAL: CUDA not available — NB04 would train on CPU for days.")
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
PY
if [ $? -ne 0 ]; then
    stamp "NB04 pre-check FAILED. NB03 results are safe; NB04/NB05 skipped."
    stamp "Install torch on the GPU node, then run ./run_nb04_gpu.sh and ./run_nb05.sh."
    exit 1
fi

export NB04_DEVICE=cuda NB04_LIMIT_DAYS=None NB04_RUN_PRIOR=1 NB04_RESUME=0 NB04_SEED=0 \
       NB04_EPOCHS_PRE="${NB04_EPOCHS_PRE:-120}" \
       NB04_EPOCHS_R1="${NB04_EPOCHS_R1:-120}" \
       NB04_EPOCHS_FT="${NB04_EPOCHS_FT:-60}" \
       NB04_N_SYNTH="${NB04_N_SYNTH:-4000}" \
       NB04_MAX_QUOTES_PER_DAY=2500 NB04_EXPORT_DAYS=100000 THESIS_FIG_PREFIX="nb04"

if run_nb "04_neural_operator.ipynb" "04_neural_operator_executed" "$THESIS_ROOT/logs/nb04.log"; then
    n4=$(ls -1 "$THESIS_OUT_DIR/nb04_surfaces"/*.npz 2>/dev/null | wc -l)
    stamp "NB04 done — $n4 .npz packs"
else
    stamp "NB04 FAILED — see logs/nb04.log. NB03 results are safe; NB05 skipped."
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. NB05 — downstream economics. Consumes NB03 + NB04 packs.
# ---------------------------------------------------------------------------
banner "STEP 3/4 — NB05 downstream economics"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export NB05_F1_MAX_DAYS=100000 NB05_F2_HORIZONS="1,3,5,10" NB05_G_NREP=500 \
       NB05_MIN_SIGNAL_NAMES=30 THESIS_FIG_PREFIX="nb05"

if run_nb "05_downstream_economics.ipynb" "05_downstream_economics_executed" "$THESIS_ROOT/logs/nb05.log"; then
    stamp "NB05 done"
else
    stamp "NB05 FAILED — see logs/nb05.log"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. DONE
# ---------------------------------------------------------------------------
banner "STEP 4/4 — PIPELINE COMPLETE — $(date)"
stamp "executed notebooks : $(ls -1 "$THESIS_ROOT"/*_executed.ipynb 2>/dev/null | wc -l)"
stamp "NB03 packs         : $(ls -1 "$THESIS_OUT_DIR/nb03_surfaces"/*.npz 2>/dev/null | wc -l)"
stamp "NB04 packs         : $(ls -1 "$THESIS_OUT_DIR/nb04_surfaces"/*.npz 2>/dev/null | wc -l)"
stamp "figures (pdf)      : $(ls -1 "$THESIS_FIG_DIR"/*.pdf 2>/dev/null | wc -l)"
stamp "nb05 tables        : $(ls -1 "$THESIS_OUT_DIR"/nb05_*.parquet 2>/dev/null | wc -l)"
echo | tee -a "$MASTER_LOG"
stamp "Collect from your laptop with:"
stamp "  rsync -avP -e 'ssh -J $USER@ssh.enst.fr' $USER@$(hostname).enst.fr:$THESIS_OUT_DIR/ ./data/clean/"
banner "ALL DONE"
