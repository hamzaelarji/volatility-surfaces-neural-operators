#!/usr/bin/env bash
# ============================================================================
# run.sh — single entry point for every cluster run.
#
#   ./run.sh smoke                sandboxed end-to-end NB03 -> NB04 -> NB05 (minutes)
#   ./run.sh nb03 [years...]      NB03 sharded by year in parallel + one synthetic pass
#   ./run.sh nb04 [gpu|cpu]       NB04 operators (default gpu; CUDA_DEVICE picks the card)
#   ./run.sh nb05                 NB05 downstream economics (needs NB03+NB04 packs)
#   ./run.sh nb06                 NB06 path-dependence / barrier experiment
#   ./run.sh all [gpu|cpu]        smoke -> nb03 -> nb04 -> nb05, sequentially, one node
#
# Always inside tmux for anything but smoke:
#   ssh <node> ; cd ~/thesis ; tmux new -s run ; ./run.sh all
#   ctrl+B then D to detach; tmux a -t run to reattach.
# Progress: tail -f ~/thesis/logs/*.log
# ============================================================================

set -u
cd "$(dirname "$0")" || exit 1
export THESIS_ROOT="${THESIS_ROOT:-/home/infres/$USER/thesis}"
source ./env_thesis.sh

MASTER_LOG="$THESIS_ROOT/logs/run_all.log"
banner () { echo -e "\n########################################################################\n# $*\n########################################################################" | tee -a "$MASTER_LOG"; }
stamp  () { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }
blas   () { export OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 MKL_NUM_THREADS=$1 NUMEXPR_NUM_THREADS=$1; }

check_torch_cuda () {
  # fail NOW, not after six hours of silent CPU training
  python3 - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("FATAL: torch is not installed "
             "(pip install torch --index-url https://download.pytorch.org/whl/cu121)")
print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() is False — you would silently "
             "train on CPU for days. Check the driver and the torch CUDA build.")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {name} | compute capability {cap[0]}.{cap[1]}")
if cap[0] < 5:
    sys.exit(f"FATAL: {name} (sm_{cap[0]}{cap[1]}) is too old for modern torch. "
             "Use another node or install a CUDA 11 build of torch.")
x = torch.randn(2000, 2000, device="cuda")
print("matmul smoke test:", float((x @ x).sum()) != 0.0)
PY
}

# ---------------------------------------------------------------------------
# NB03 — deep smoother, sharded by year in parallel + one synthetic/stress pass.
# Each shard is single-threaded at the Python level; BLAS is pinned so eight
# shards share the node politely. No GPU: hide it so we never hold a slot.
# ---------------------------------------------------------------------------
do_nb03 () {
  local years="${*:-2018 2019 2020 2021 2022 2023 2024 2025}"
  banner "NB03 deep smoother — years: $years"
  blas 4
  export CUDA_VISIBLE_DEVICES=""
  export NB03_LIMIT_DATES=None NB03_EXPORT_ALL=1 NB03_LAM0_STRIDE=5 \
         NB03_CKPT_EVERY=25 NB03_RUN_LAM0_CHECK=1 NB03_SKIP_SYNTHETIC=1

  local pids=() fail=0 y p
  for y in $years; do
    (
      export NB03_SHARD="$y" NB03_DATE_START="${y}-01-01" NB03_DATE_END="${y}-12-31" \
             THESIS_FIG_PREFIX="nb03"
      run_nb "03_deep_smoother.ipynb" "03_deep_smoother_${y}_executed" \
             "$THESIS_ROOT/logs/nb03_${y}.log"
    ) &
    pids+=($!); stamp "  NB03 shard $y launched (pid ${pids[-1]})"
    sleep 3          # stagger: avoids 8 processes reading the parquet at once
  done
  # The synthetic / stress sections (3, 3b, 3c, 4) are identical in every shard,
  # so they are skipped above and run ONCE here, on a short window.
  (
    export NB03_SHARD="synth" NB03_SKIP_SYNTHETIC=0 NB03_LIMIT_DATES=2 \
           NB03_EXPORT_ALL=0 NB03_LAM0_STRIDE=0 THESIS_FIG_PREFIX="nb03synth"
    run_nb "03_deep_smoother.ipynb" "03_deep_smoother_synth_executed" \
           "$THESIS_ROOT/logs/nb03_synth.log"
  ) &
  pids+=($!); stamp "  NB03 synthetic pass launched (pid ${pids[-1]})"

  for p in "${pids[@]}"; do wait "$p" || { stamp "  !! pid $p exited non-zero"; fail=1; }; done
  local n3; n3=$(ls -1 "$THESIS_OUT_DIR/nb03_surfaces"/*.npz 2>/dev/null | wc -l)
  stamp "NB03 done — $n3 .npz packs — fail=$fail"
  [ "$n3" -eq 0 ] && { stamp "FATAL: NB03 produced no packs."; return 1; }
  return $fail
}

# ---------------------------------------------------------------------------
# NB04 — neural operators (DeepONet + GNO, plain AND prior-embedded).
# gpu (default): full epoch budget, CUDA sanity-checked first.
# cpu: for nodes whose GPUs are too old for modern torch — the models are tiny
#      (~200k params) so CPU works; epoch budgets trimmed since epochs cost more.
# ---------------------------------------------------------------------------
do_nb04 () {
  local mode="${1:-gpu}"
  banner "NB04 neural operators ($mode)"
  export NB04_LIMIT_DAYS=None NB04_RUN_PRIOR=1 NB04_RESUME="${NB04_RESUME:-0}" NB04_SEED=0 \
         NB04_MAX_QUOTES_PER_DAY=2500 NB04_EXPORT_DAYS=100000 THESIS_FIG_PREFIX="nb04"
  if [ "$mode" = "gpu" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
    blas 8
    check_torch_cuda || return 1
    export NB04_DEVICE=cuda \
           NB04_EPOCHS_PRE="${NB04_EPOCHS_PRE:-120}" NB04_EPOCHS_R1="${NB04_EPOCHS_R1:-120}" \
           NB04_EPOCHS_FT="${NB04_EPOCHS_FT:-60}" NB04_N_SYNTH="${NB04_N_SYNTH:-4000}"
  else
    local cap="${CORE_CAP:-$(( $(nproc) / 3 ))}"; [ "$cap" -lt 8 ] && cap=8
    blas "$cap"
    export NB04_DEVICE=cpu \
           NB04_EPOCHS_PRE="${NB04_EPOCHS_PRE:-40}" NB04_EPOCHS_R1="${NB04_EPOCHS_R1:-40}" \
           NB04_EPOCHS_FT="${NB04_EPOCHS_FT:-20}" NB04_N_SYNTH="${NB04_N_SYNTH:-1500}"
  fi
  stamp "epochs pre/R1/ft: $NB04_EPOCHS_PRE/$NB04_EPOCHS_R1/$NB04_EPOCHS_FT | synth: $NB04_N_SYNTH"
  run_nb "04_neural_operator.ipynb" "04_neural_operator_executed" \
         "$THESIS_ROOT/logs/nb04.log" || { stamp "NB04 FAILED — see logs/nb04.log"; return 1; }
  stamp "NB04 done — $(ls -1 "$THESIS_OUT_DIR/nb04_surfaces"/*.npz 2>/dev/null | wc -l) .npz packs"
}

# ---------------------------------------------------------------------------
# NB05 — downstream economics. Consumes the NB03 + NB04 packs: refuse to run on
# empty pack directories rather than "succeed" with every section skipped.
# ---------------------------------------------------------------------------
do_nb05 () {
  banner "NB05 downstream economics"
  export CUDA_VISIBLE_DEVICES=""; blas 16
  local n3 n4
  n3=$(ls -1 "$THESIS_OUT_DIR/nb03_surfaces"/*.npz 2>/dev/null | wc -l)
  n4=$(ls -1 "$THESIS_OUT_DIR/nb04_surfaces"/*.npz 2>/dev/null | wc -l)
  stamp "NB03 packs: $n3 | NB04 packs: $n4"
  if [ "$n3" -eq 0 ] || [ "$n4" -eq 0 ]; then
    stamp "REFUSING TO RUN: a pack directory is empty. Finish NB03/NB04 first."
    return 1
  fi
  [ -f "$THESIS_VIX_RAW" ] || stamp "WARNING: $THESIS_VIX_RAW missing — VIX section will skip."
  export NB05_F1_MAX_DAYS=100000 NB05_F2_HORIZONS="1,3,5,10" NB05_G_NREP=500 \
         NB05_MIN_SIGNAL_NAMES=30 THESIS_FIG_PREFIX="nb05"
  run_nb "05_downstream_economics.ipynb" "05_downstream_economics_executed" \
         "$THESIS_ROOT/logs/nb05.log" || { stamp "NB05 FAILED — see logs/nb05.log"; return 1; }
  stamp "NB05 done"
  ls -1 "$THESIS_OUT_DIR"/nb05_*.parquet 2>/dev/null | sed 's/^/   /'
}

# ---------------------------------------------------------------------------
# NB06 — path-dependence / barrier experiment. Synthetic sections (B1-B7) are
# self-contained; the real-pack section (B8) skips cleanly if packs are absent.
# ---------------------------------------------------------------------------
do_nb06 () {
  banner "NB06 path-dependence / barrier"
  export CUDA_VISIBLE_DEVICES=""; blas 16
  export THESIS_FIG_PREFIX="nb06"
  run_nb "06_path_dependence_barrier.ipynb" "06_path_dependence_barrier_executed" \
         "$THESIS_ROOT/logs/nb06.log" || { stamp "NB06 FAILED — see logs/nb06.log"; return 1; }
  stamp "NB06 done"
  ls -1 "$THESIS_OUT_DIR"/nb06_*.parquet 2>/dev/null | sed 's/^/   /'
}

# ---------------------------------------------------------------------------
# SMOKE — tiny end-to-end NB03 -> NB04 -> NB05 in a sandboxed output directory.
# Proves environment, paths, GPU and figure export before committing days.
# ---------------------------------------------------------------------------
do_smoke () {
  banner "SMOKE TEST (sandboxed)"
  export THESIS_OUT_DIR="$THESIS_ROOT/data/smoke"
  export THESIS_FIG_DIR="$THESIS_OUT_DIR/figures"
  mkdir -p "$THESIS_OUT_DIR" "$THESIS_FIG_DIR"
  # quotes and NB02 benchmarks still come from the real clean dir
  export THESIS_OPT_PARQUET="$THESIS_ROOT/data/clean/option_prices_clean.parquet"
  local f
  for f in benchmark_svi_slices_full.parquet benchmark_ssvi_days_full.parquet; do
    [ -f "$THESIS_ROOT/data/clean/$f" ] && cp -n "$THESIS_ROOT/data/clean/$f" "$THESIS_OUT_DIR/" 2>/dev/null
  done
  [ -f "$THESIS_OPT_PARQUET" ] || { stamp "FATAL: quotes not found at $THESIS_OPT_PARQUET"; return 1; }

  python3 - <<'PY' || return 1
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

  stamp "smoke 1/3 — NB03 (2 days)"
  ( export NB03_SHARD="smoke" NB03_LIMIT_DATES=2 NB03_EXPORT_ALL=1 NB03_SKIP_SYNTHETIC=1 \
           NB03_LAM0_STRIDE=1 NB03_CKPT_EVERY=1 THESIS_FIG_PREFIX="nb03" CUDA_VISIBLE_DEVICES=""
    run_nb "03_deep_smoother.ipynb" "smoke_03_executed" "$THESIS_ROOT/logs/smoke_nb03.log"
  ) || { stamp "NB03 smoke FAILED"; tail -30 "$THESIS_ROOT/logs/smoke_nb03.log"; return 1; }

  stamp "smoke 2/3 — NB04 (5 days, tiny epochs)"
  ( export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}" NB04_LIMIT_DAYS=5 NB04_EPOCHS_PRE=2 \
           NB04_EPOCHS_R1=2 NB04_EPOCHS_FT=1 NB04_N_SYNTH=40 NB04_RUN_PRIOR=1 \
           NB04_EXPORT_DAYS=5 NB04_RESUME=0 THESIS_FIG_PREFIX="nb04"
    run_nb "04_neural_operator.ipynb" "smoke_04_executed" "$THESIS_ROOT/logs/smoke_nb04.log"
  ) || { stamp "NB04 smoke FAILED"; tail -30 "$THESIS_ROOT/logs/smoke_nb04.log"; return 1; }

  stamp "smoke 3/3 — NB05"
  ( export CUDA_VISIBLE_DEVICES="" NB05_F1_MAX_DAYS=2 NB05_G_NREP=5 NB05_F2_HORIZONS="1,3" \
           THESIS_FIG_PREFIX="nb05"
    run_nb "05_downstream_economics.ipynb" "smoke_05_executed" "$THESIS_ROOT/logs/smoke_nb05.log"
  ) || { stamp "NB05 smoke FAILED"; tail -30 "$THESIS_ROOT/logs/smoke_nb05.log"; return 1; }

  stamp "SMOKE TEST PASSED"
  stamp " nb03 packs: $(ls -1 "$THESIS_OUT_DIR/nb03_surfaces"/*.npz 2>/dev/null | wc -l)"
  stamp " nb04 packs: $(ls -1 "$THESIS_OUT_DIR/nb04_surfaces"/*.npz 2>/dev/null | wc -l)"
  stamp " figures   : $(ls -1 "$THESIS_FIG_DIR"/*.pdf 2>/dev/null | wc -l) pdf (must be non-zero)"
  stamp "Delete the sandbox when satisfied:  rm -rf $THESIS_OUT_DIR"
}

# ---------------------------------------------------------------------------
# ALL — one launch, walk away: smoke -> NB03 -> NB04 -> NB05 on a single node.
# ---------------------------------------------------------------------------
do_all () {
  local mode="${1:-gpu}"
  banner "PIPELINE START on $(hostname) ($mode) — $(date)"
  ( do_smoke >>"$MASTER_LOG" 2>&1 ) || { stamp "smoke test FAILED — nothing else run."; return 1; }
  stamp "smoke test PASSED"
  rm -rf "$THESIS_ROOT/data/smoke"
  # real production output directory from here on
  export THESIS_OUT_DIR="$THESIS_ROOT/data/clean"
  export THESIS_OPT_PARQUET="$THESIS_OUT_DIR/option_prices_clean.parquet"
  export THESIS_FIG_DIR="$THESIS_OUT_DIR/figures"
  mkdir -p "$THESIS_FIG_DIR"
  do_nb03 || return 1
  do_nb04 "$mode" || { stamp "NB03 results are safe; NB05 skipped."; return 1; }
  do_nb05 || return 1
  banner "PIPELINE COMPLETE — $(date)"
  stamp "executed notebooks : $(ls -1 "$THESIS_ROOT"/*_executed.ipynb 2>/dev/null | wc -l)"
  stamp "figures (pdf)      : $(ls -1 "$THESIS_FIG_DIR"/*.pdf 2>/dev/null | wc -l)"
  stamp "Collect from your laptop with:"
  stamp "  rsync -avP $USER@ssh.enst.fr:$THESIS_OUT_DIR/ ./data/clean/"
}

case "${1:-}" in
  smoke) do_smoke ;;
  nb03)  shift; do_nb03 "$@" ;;
  nb04)  shift; do_nb04 "${1:-gpu}" ;;
  nb05)  do_nb05 ;;
  nb06)  do_nb06 ;;
  all)   shift || true; do_all "${1:-gpu}" ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
