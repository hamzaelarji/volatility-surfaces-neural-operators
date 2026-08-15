#!/usr/bin/env python3
"""
run_thesis.py — execute the thesis notebooks sequentially, one kernel per notebook.

Usage:
    python run_thesis.py               # run all notebooks in order
    python run_thesis.py 04 05        # run only NB04 and NB05
    python run_thesis.py --list       # show the plan and env, run nothing

Each notebook runs in its OWN subprocess (memory is fully released between
notebooks — NB03/NB04 hold a lot). Executed copies are saved next to the
originals as <name>_executed_<timestamp>.ipynb. The pipeline stops at the
first failure (everything downstream depends on the upstream parquets).

If papermill is installed (pip install papermill) it is used, because on
failure it still writes the partially-executed notebook WITH the traceback
embedded — your usual debugging workflow. Otherwise falls back to
jupyter nbconvert.
"""

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ======================================================================
# CONFIG — edit this block, nothing else
# ======================================================================
NB_DIR = Path(__file__).resolve().parent      # folder containing the .ipynb files

NOTEBOOKS = {
    "01": "01_data_cleaning.ipynb",
    "02": "02_svi_ssvi_benchmark.ipynb",
    "03": "03_deep_smoother_v8-2.ipynb",
    "04": "04_neural_operator.ipynb",
    "05": "05_downstream_economics.ipynb",
}

# Environment shared by every notebook.
ENV_GLOBAL = {
    # "THESIS_OUT_DIR": "data/clean",
    "PLOTLY_RENDERER": "notebook_connected",
}

# Per-notebook environment. Uncomment / edit for the production run.
ENV_PER_NB = {
    "03": {
        "NB03_LIMIT_DATES": "None",
        "NB03_EXPORT_ALL": "1",
        "NB03_LAM0_STRIDE": "1",      # 3 si tu veux gagner ~8h wall
        "NB03_SKIP_SYNTHETIC": "1",
        "NB03_CKPT_EVERY": "25",
    },
    "04": {"NB04_RESUME": "1", "NB04_EXPORT_DAYS": "386"},
    "05": {"NB05_F1_MAX_DAYS": "1926"},
}

TIMEOUT_S = None          # per-notebook wall clock limit; None = unlimited
STOP_ON_FAILURE = True
# ======================================================================


def find_notebook(name: str) -> Path:
    """Exact name, else the most recent file whose name starts with the same stem
    (handles your *_executed_<ts>.ipynb copies and versioned names like _v2)."""
    p = NB_DIR / name
    if p.exists():
        return p
    stem = name.split(".ipynb")[0][:2]        # match on the NN prefix
    cands = sorted(NB_DIR.glob(f"{stem}_*.ipynb"), key=lambda x: x.stat().st_mtime)
    cands = [c for c in cands if "_executed_" not in c.name] or cands
    if not cands:
        sys.exit(f"[runner] no notebook found for '{name}' in {NB_DIR}")
    return cands[-1]


def run_one(key: str, ts: str, use_papermill: bool) -> tuple[bool, float, Path]:
    src = find_notebook(NOTEBOOKS[key])
    out = src.with_name(f"{src.stem}_executed_{ts}.ipynb")

    # The SHELL wins over ENV_PER_NB: a variable exported on the command line
    # (NB03_SHARD, NB03_DATE_START, a test override...) must never be clobbered by the
    # config block, or a "10-minute test" silently becomes the full production run.
    env = os.environ.copy()
    env.update(ENV_GLOBAL)
    defaults = {k: v for k, v in ENV_PER_NB.get(key, {}).items() if k not in os.environ}
    env.update(defaults)
    shown = {**ENV_GLOBAL, **defaults,
             **{k: os.environ[k] for k in ENV_PER_NB.get(key, {}) if k in os.environ}}

    if use_papermill:
        cmd = [sys.executable, "-m", "papermill", str(src), str(out),
               "--cwd", str(NB_DIR), "--log-output", "--request-save-on-cell-execute"]
    else:
        cmd = [sys.executable, "-m", "jupyter", "nbconvert",
               "--to", "notebook", "--execute",
               "--ExecutePreprocessor.timeout=-1",
               "--output", out.name, "--output-dir", str(NB_DIR), str(src)]

    print(f"\n{'=' * 78}\n[runner] NB{key}: {src.name}\n"
          f"[runner] env: {shown or '(defaults)'}\n"
          f"[runner] -> {out.name}\n{'=' * 78}", flush=True)

    t0 = time.time()
    try:
        res = subprocess.run(cmd, env=env, cwd=NB_DIR, timeout=TIMEOUT_S)
        ok = res.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[runner] NB{key} TIMEOUT after {TIMEOUT_S}s", flush=True)
        ok = False
    dt = time.time() - t0
    return ok, dt, out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    keys = args if args else list(NOTEBOOKS)
    bad = [k for k in keys if k not in NOTEBOOKS]
    if bad:
        sys.exit(f"[runner] unknown notebook key(s) {bad}; valid: {list(NOTEBOOKS)}")

    use_papermill = _module_exists("papermill")   # must be importable by THIS interpreter
    engine = "papermill" if use_papermill else "jupyter nbconvert"

    if "--list" in sys.argv:
        print(f"engine: {engine}\nplan  : {[NOTEBOOKS[k] for k in keys]}\n"
              f"global env: {ENV_GLOBAL}\nper-nb env: { {k: ENV_PER_NB.get(k, {}) for k in keys} }")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = NB_DIR / f"pipeline_{ts}.log"
    print(f"[runner] engine={engine} | run id {ts} | log -> {log.name}")

    results = []
    t_all = time.time()
    for k in keys:
        ok, dt, out = run_one(k, ts, use_papermill)
        results.append((k, ok, dt, out.name))
        with open(log, "a") as fh:
            fh.write(f"NB{k}\t{'OK' if ok else 'FAIL'}\t{dt:.0f}s\t{out.name}\n")
        print(f"[runner] NB{k} {'OK' if ok else 'FAIL'} in {dt / 60:.1f} min", flush=True)
        if not ok and STOP_ON_FAILURE:
            print(f"[runner] stopping: downstream notebooks depend on NB{k}'s outputs."
                  f"{' Traceback is embedded in ' + out.name if use_papermill else ''}")
            break

    print(f"\n{'=' * 78}\n[runner] SUMMARY ({(time.time() - t_all) / 60:.1f} min total)")
    for k, ok, dt, name in results:
        print(f"  NB{k}  {'OK  ' if ok else 'FAIL'}  {dt / 60:>7.1f} min  {name}")
    sys.exit(0 if all(ok for _, ok, _, _ in results) else 1)


def _module_exists(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


if __name__ == "__main__":
    main()