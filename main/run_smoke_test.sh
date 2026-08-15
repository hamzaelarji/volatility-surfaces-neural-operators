#!/usr/bin/env bash
# SMOKE TEST — run this FIRST, before committing hours of compute.
#
# Executes the whole NB03 -> NB04 -> NB05 chain with tiny settings (a few
# minutes). It proves the environment, the paths, the GPU and the figure export
# all work end to end. Almost every failed cluster run fails in the first two
# minutes for a reason this test would have caught.
#
#   ssh gpu1                 # a GPU node, so NB04's cuda path is exercised too
#   cd ~/thesis && ./run_smoke_test.sh
#
# It writes to a SEPARATE output directory, so it cannot pollute production
# results.

set -u
cd "$(dirname "$0")" || exit 1

export THESIS_ROOT="${THESIS_ROOT:-/home/infres/$USER/thesis}"
source ./env_thesis.sh

# redirect all outputs to a sandbox
export THESIS_OUT_DIR="$THESIS_ROOT/data/smoke"
export THESIS_FIG_DIR="$THESIS_OUT_DIR/figures"
mkdir -p "$THESIS_OUT_DIR" "$THESIS_FIG_DIR"

# the quotes and the NB02 benchmarks still come from the real clean dir
export THESIS_OPT_PARQUET="$THESIS_ROOT/data/clean/option_prices_clean.parquet"
for f in benchmark_svi_slices_full.parquet benchmark_ssvi_days_full.parquet; do
  [ -f "$THESIS_ROOT/data/clean/$f" ] && cp -n "$THESIS_ROOT/data/clean/$f" "$THESIS_OUT_DIR/" 2>/dev/null
done
[ -f "$THESIS_OPT_PARQUET" ] || { echo "FATAL: quotes not found at $THESIS_OPT_PARQUET"; exit 1; }

echo "############ 0. environment ############"
python3 - <<'PY' || exit 1
import importlib, sys
need = ["numpy", "scipy", "polars", "plotly", "kaleido", "autograd", "nbformat", "nbconvert"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"FATAL: missing packages: {missing}")
import plotly.graph_objects as go
go.Figure(go.Scatter(x=[0, 1], y=[0, 1])).write_image("/tmp/_smoke.pdf")
print("static figure export: OK")
try:
    import torch
    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
except ImportError:
    print("torch: NOT INSTALLED (fine on a CPU node, fatal on the GPU node)")
PY

echo "############ 1. NB03 (2 days) ############"
export NB03_SHARD="smoke" NB03_LIMIT_DATES=2 NB03_EXPORT_ALL=1 \
       NB03_SKIP_SYNTHETIC=1 NB03_LAM0_STRIDE=1 NB03_CKPT_EVERY=1
export THESIS_FIG_PREFIX="nb03"
export CUDA_VISIBLE_DEVICES=""
run_nb "03_deep_smoother.ipynb" "smoke_03_executed" "$THESIS_ROOT/logs/smoke_nb03.log" \
  || { echo "NB03 FAILED — see logs/smoke_nb03.log"; tail -30 "$THESIS_ROOT/logs/smoke_nb03.log"; exit 1; }

echo "############ 2. NB04 (5 days, tiny epochs) ############"
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export NB04_LIMIT_DAYS=5 NB04_EPOCHS_PRE=2 NB04_EPOCHS_R1=2 NB04_EPOCHS_FT=1 \
       NB04_N_SYNTH=40 NB04_RUN_PRIOR=1 NB04_EXPORT_DAYS=5 NB04_RESUME=0
export THESIS_FIG_PREFIX="nb04"
run_nb "04_neural_operator.ipynb" "smoke_04_executed" "$THESIS_ROOT/logs/smoke_nb04.log" \
  || { echo "NB04 FAILED — see logs/smoke_nb04.log"; tail -30 "$THESIS_ROOT/logs/smoke_nb04.log"; exit 1; }

echo "############ 3. NB05 ############"
export CUDA_VISIBLE_DEVICES=""
export NB05_F1_MAX_DAYS=2 NB05_G_NREP=5 NB05_F2_HORIZONS="1,3"
export THESIS_FIG_PREFIX="nb05"
run_nb "05_downstream_economics.ipynb" "smoke_05_executed" "$THESIS_ROOT/logs/smoke_nb05.log" \
  || { echo "NB05 FAILED — see logs/smoke_nb05.log"; tail -30 "$THESIS_ROOT/logs/smoke_nb05.log"; exit 1; }

echo
echo "=============================================================="
echo " SMOKE TEST PASSED"
echo " nb03 packs : $(ls -1 "$THESIS_OUT_DIR/nb03_surfaces"/*.npz 2>/dev/null | wc -l)"
echo " nb04 packs : $(ls -1 "$THESIS_OUT_DIR/nb04_surfaces"/*.npz 2>/dev/null | wc -l)"
echo " figures    : $(ls -1 "$THESIS_FIG_DIR"/*.pdf 2>/dev/null | wc -l) pdf"
echo
echo " Check that the figure count is non-zero, then launch production:"
echo "   lame25 : tmux new -s nb03 ; ./run_nb03_shards.sh"
echo "   gpu1   : tmux new -s nb04 ; CUDA_DEVICE=0 ./run_nb04_gpu.sh"
echo " Delete the sandbox when satisfied:  rm -rf $THESIS_OUT_DIR"
echo "=============================================================="
