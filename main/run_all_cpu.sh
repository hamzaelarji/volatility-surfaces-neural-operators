#!/usr/bin/env bash
# ============================================================================
# run_all_cpu.sh — ONE launch, CPU-only, on a big CPU node (lame25 / lame26).
#
# The Telecom GPU nodes are Tesla K80 (2014, compute 3.7), which modern PyTorch
# no longer supports. So the whole pipeline runs on CPU here. NB04's models are
# tiny (~200k params), so CPU is fine — and lame25 has 128 cores.
#
#     smoke test  ->  NB03 (sharded)  ->  NB04 (CPU)  ->  NB05
#
#   ssh lame25
#   cd ~/thesis && conda activate thesis
#   tmux new -s allrun
#   ./run_all_cpu.sh
#   ctrl+B then D          # detach, leave for 2-3 days
#
# Reattach:  ssh lame25 && tmux a -t allrun
# Progress:  tail -f ~/thesis/logs/run_all.log
# ============================================================================

set -u
cd "$(dirname "$0")" || exit 1
export THESIS_ROOT="${THESIS_ROOT:-/home/infres/$USER/thesis}"
source ./env_thesis.sh

MASTER_LOG="$THESIS_ROOT/logs/run_all.log"

banner () { echo -e "\n\n########################################################################\n# $*\n########################################################################" | tee -a "$MASTER_LOG"; }
stamp  () { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

# No GPU anywhere in this run.
export CUDA_VISIBLE_DEVICES=""
export NB04_DEVICE=cpu

# lame25 rule: a third of 128 cores. Cap total concurrency accordingly.
CORE_CAP=40

banner "CPU PIPELINE START on $(hostname) — $(date)"

# ---------------------------------------------------------------------------
# pre-check: torch present, CPU is fine
# ---------------------------------------------------------------------------
python3 - <<'PY' >>"$MASTER_LOG" 2>&1
import sys
try:
    import torch
except ImportError:
    sys.exit("FATAL: torch not installed. Run: "
             "pip install torch --index-url https://download.pytorch.org/whl/cpu")
print("torch", torch.__version__, "| device: cpu | threads:", torch.get_num_threads())
PY
[ $? -ne 0 ] && { stamp "torch missing — see log. Install torch CPU, then rerun."; exit 1; }

# ---------------------------------------------------------------------------
# 0. SMOKE TEST
# ---------------------------------------------------------------------------
banner "STEP 0/4 — smoke test"
if ./run_smoke_test.sh >>"$MASTER_LOG" 2>&1; then
    stamp "smoke test PASSED"
else
    stamp "smoke test FAILED — see $MASTER_LOG. Nothing else run."
    exit 1
fi
rm -rf "$THESIS_ROOT/data/smoke"

export THESIS_OUT_DIR="$THESIS_ROOT/data/clean"
export THESIS_OPT_PARQUET="$THESIS_OUT_DIR/option_prices_clean.parquet"
export THESIS_FIG_DIR="$THESIS_OUT_DIR/figures"
mkdir -p "$THESIS_FIG_DIR"

# ---------------------------------------------------------------------------
# 1. NB03 — sharded by year, 8 parallel processes, BLAS pinned low.
# ---------------------------------------------------------------------------
banner "STEP 1/4 — NB03 deep smoother (all years, 8 shards)"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
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
(
  export NB03_SHARD="synth" NB03_SKIP_SYNTHETIC=0 NB03_LIMIT_DATES=2 \
         NB03_EXPORT_ALL=0 NB03_LAM0_STRIDE=0 THESIS_FIG_PREFIX="nb03synth"
  run_nb "03_deep_smoother.ipynb" "03_deep_smoother_synth_executed" \
         "$THESIS_ROOT/logs/nb03_synth.log"
) &
nb03_pids+=($!); stamp "  NB03 synthetic pass launched (pid ${nb03_pids[-1]})"

for p in "${nb03_pids[@]}"; do wait "$p" || stamp "  a NB03 shard (pid $p) exited non-zero"; done
n3=$(ls -1 "$THESIS_OUT_DIR/nb03_surfaces"/*.npz 2>/dev/null | wc -l)
stamp "NB03 done — $n3 .npz packs"
[ "$n3" -eq 0 ] && { stamp "FATAL: NB03 produced no packs. Stopping."; exit 1; }

# ---------------------------------------------------------------------------
# 2. NB04 — neural operators on CPU.
#    torch multithreads internally, so this is ONE process using many cores.
#    Epoch budgets are trimmed vs the GPU plan because CPU epochs cost more.
# ---------------------------------------------------------------------------
banner "STEP 2/4 — NB04 neural operators (CPU)"
export OMP_NUM_THREADS=$CORE_CAP OPENBLAS_NUM_THREADS=$CORE_CAP MKL_NUM_THREADS=$CORE_CAP
export NB04_DEVICE=cpu NB04_LIMIT_DAYS=None NB04_RUN_PRIOR=1 NB04_RESUME=0 NB04_SEED=0 \
       NB04_EPOCHS_PRE="${NB04_EPOCHS_PRE:-40}" \
       NB04_EPOCHS_R1="${NB04_EPOCHS_R1:-40}" \
       NB04_EPOCHS_FT="${NB04_EPOCHS_FT:-20}" \
       NB04_N_SYNTH="${NB04_N_SYNTH:-1500}" \
       NB04_MAX_QUOTES_PER_DAY=2500 NB04_EXPORT_DAYS=100000 THESIS_FIG_PREFIX="nb04"

if run_nb "04_neural_operator.ipynb" "04_neural_operator_executed" "$THESIS_ROOT/logs/nb04.log"; then
    n4=$(ls -1 "$THESIS_OUT_DIR/nb04_surfaces"/*.npz 2>/dev/null | wc -l)
    stamp "NB04 done — $n4 .npz packs"
else
    stamp "NB04 FAILED — see logs/nb04.log. NB03 results are safe; NB05 skipped."
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. NB05 — downstream economics.
# ---------------------------------------------------------------------------
banner "STEP 3/4 — NB05 downstream economics"
export OMP_NUM_THREADS=$CORE_CAP OPENBLAS_NUM_THREADS=$CORE_CAP
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
stamp "Collect from your laptop:"
stamp "  rsync -avP -e 'ssh -J $USER@ssh.enst.fr' $USER@$(hostname).enst.fr:$THESIS_OUT_DIR/ ./data/clean/"
banner "ALL DONE"
