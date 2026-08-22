#!/usr/bin/env bash
# NB04 — neural operators (DeepONet + GNO, plain AND prior-embedded), FULL run.
#
# This is the only notebook that uses the GPU. It trains 4 families x 3 regimes:
#   deeponet, gno                    -- the plain operators
#   deeponet_prior, gno_prior        -- iv = iv_SSVI * exp(s*tanh(operator)),
#                                       zero-initialised so training STARTS at the
#                                       certified prior (the answer to the
#                                       prior-embedding critique)
#
# Run this INSIDE tmux on a GPU node (gpu1 / gpu2 / gpu3):
#   ssh gpu1
#   nvidia-smi                       # pick a free GPU, note its index
#   cd ~/thesis && tmux new -s nb04
#   CUDA_DEVICE=0 ./run_nb04_gpu.sh
#   ctrl+B then D
#
# Reattach with:  ssh gpu1 && tmux a -t nb04

set -u
cd "$(dirname "$0")" || exit 1
source ./env_thesis.sh

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export THESIS_FIG_PREFIX="nb04"

# ---- sanity check: fail NOW, not after six hours of CPU training -------------
python3 - <<'PY' || exit 1
import sys
try:
    import torch
except ImportError:
    sys.exit("FATAL: torch is not installed in this environment")
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

export NB04_DEVICE=cuda

# ---- production settings -----------------------------------------------------
export NB04_LIMIT_DAYS=None       # every available day
export NB04_RUN_PRIOR=1           # prior-embedded families ON
export NB04_RESUME=0              # 0 = train from scratch
export NB04_SEED=0

# raised epoch budgets (the whole point of moving to GPU)
export NB04_EPOCHS_PRE="${NB04_EPOCHS_PRE:-120}"
export NB04_EPOCHS_R1="${NB04_EPOCHS_R1:-120}"
export NB04_EPOCHS_FT="${NB04_EPOCHS_FT:-60}"

export NB04_N_SYNTH="${NB04_N_SYNTH:-4000}"
export NB04_MAX_QUOTES_PER_DAY=2500
export NB04_EXPORT_DAYS=100000    # export every test day; NB05 wants them all

echo "=============================================================="
echo " NB04 production run (GPU ${CUDA_VISIBLE_DEVICES})"
echo " epochs pre/R1/ft : $NB04_EPOCHS_PRE / $NB04_EPOCHS_R1 / $NB04_EPOCHS_FT"
echo " synthetics       : $NB04_N_SYNTH"
echo " prior families   : $NB04_RUN_PRIOR"
echo "=============================================================="

run_nb "04_neural_operator.ipynb" "04_neural_operator_executed" \
       "$THESIS_ROOT/logs/nb04.log"
rc=$?

echo
echo "=============================================================="
[ $rc -eq 0 ] && echo " NB04 COMPLETE" || echo " NB04 FAILED — see logs/nb04.log"
echo " packs   : $(ls -1 "$THESIS_OUT_DIR/nb04_surfaces"/*.npz 2>/dev/null | wc -l) .npz files"
echo " figures : $(ls -1 "$THESIS_FIG_DIR"/nb04*.pdf 2>/dev/null | wc -l) pdf"
echo "=============================================================="
exit $rc
