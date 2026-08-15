#!/usr/bin/env bash
# Shared environment for every thesis run on the Telecom cluster.
# Sourced by run_nb03_shards.sh / run_nb04_gpu.sh / run_nb05.sh
#
#   source env_thesis.sh

# ---- EDIT THIS ONE LINE ------------------------------------------------------
export THESIS_ROOT="${THESIS_ROOT:-/home/infres/$USER/thesis}"
# ------------------------------------------------------------------------------

export THESIS_OUT_DIR="$THESIS_ROOT/data/clean"
export THESIS_OPT_PARQUET="$THESIS_OUT_DIR/option_prices_clean.parquet"
export THESIS_VIX_RAW="$THESIS_ROOT/data/raw/VIX_History.csv"
export THESIS_VIX_CSV="$THESIS_OUT_DIR/vix_daily.csv"

# headless execution: no browser, figures written to disk as vector PDF
export THESIS_HEADLESS=1
export THESIS_FIG_DIR="$THESIS_OUT_DIR/figures"
export THESIS_FIG_FORMAT=pdf
export THESIS_FIG_HTML=1

mkdir -p "$THESIS_OUT_DIR" "$THESIS_FIG_DIR" "$THESIS_ROOT/logs"

# activate the virtualenv if present
if [ -f "$THESIS_ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$THESIS_ROOT/venv/bin/activate"
fi

# Execute one notebook headless with no cell timeout.
# usage: run_nb <notebook.ipynb> <output_suffix> <logfile>
run_nb () {
  local nb="$1" out="$2" log="$3"
  echo "[$(date '+%F %T')] START $nb -> $out" | tee -a "$log"
  jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "$out" \
    --output-dir "$THESIS_ROOT" \
    "$THESIS_ROOT/$nb" >>"$log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END   $nb rc=$rc" | tee -a "$log"
  return $rc
}
