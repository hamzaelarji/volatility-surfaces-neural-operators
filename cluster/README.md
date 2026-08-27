# Running NB03–NB06 on a compute cluster

Five files, one entry point. Everything below assumes `~/thesis` on a shared
filesystem visible from every node, so a CPU node and a GPU node can write into
the same `data/clean/` and you transfer once, collect once.

```
run.sh               single entry point (smoke / nb03 / nb04 / nb05 / nb06 / all)
env_thesis.sh        shared environment + the nbconvert helper (sourced by run.sh)
patch_notebooks.py   makes NB03–NB06 headless-safe and exports every figure to disk
requirements.txt     pinned Python environment
```

## 0. Build the patched notebooks (on your laptop)

The notebooks in `notebooks/` are the canonical source. The cluster runs a
*patched* copy: `patch_notebooks.py` inserts one bootstrap cell per notebook
that makes `fig.show()` headless-safe (no browser front-end can kill the run)
and writes every plotly figure to disk as vector PDF, prefixed per notebook and
per shard so parallel shards never overwrite each other. It is idempotent —
`build/` is disposable and gitignored, regenerate it, never edit it.

```bash
mkdir -p build && cp notebooks/0{3,4,5,6}_*.ipynb build/
python cluster/patch_notebooks.py build
```

## 1. Transfer (once)

The scripts expect a **flat** `$THESIS_ROOT`: notebooks and scripts side by side.

```bash
rsync -avP build/*.ipynb cluster/ $USER@<cluster>:~/thesis/
rsync -avP data/clean/option_prices_clean.parquet \
           data/clean/benchmark_svi_slices_full.parquet \
           data/clean/benchmark_ssvi_days_full.parquet \
           $USER@<cluster>:~/thesis/data/clean/
rsync -avP data/raw/VIX_History.csv $USER@<cluster>:~/thesis/data/raw/
```

## 2. Environment (once)

```bash
ssh <node>
cd ~/thesis
python3 -m pip install virtualenv --user
virtualenv venv --python=python3 && source venv/bin/activate
pip install -r requirements.txt
# torch only matters for NB04; match the CUDA version nvidia-smi reports
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`kaleido==0.2.1` is pinned deliberately: newer kaleido needs a system Chrome
you cannot install without root.

## 3. Smoke test (minutes — do not skip)

```bash
./run.sh smoke
```

Runs the full NB03 → NB04 → NB05 chain on a handful of days in a sandboxed
output directory (`data/smoke/`, deleted afterwards). It proves the paths, the
packages, CUDA and the figure export end to end — almost every failed cluster
run would have failed here first. If the figure count at the end is zero, fix
that before launching production.

## 4. Production

**Everything on one node, one launch** (a GPU node; `cpu` mode exists for
nodes whose GPUs are too old for modern torch):

```bash
tmux new -s run
CUDA_DEVICE=0 ./run.sh all          # or: ./run.sh all cpu
# ctrl+B then D -> detach, close the laptop
```

**Or split across nodes, in parallel** — NB03 is CPU-bound and sharded by year
(8 processes, ~30 h → ~4 h on a big node), NB04 is the only GPU consumer:

```bash
# CPU node                              # GPU node, at the same time
tmux new -s nb03                        tmux new -s nb04
./run.sh nb03                           CUDA_DEVICE=0 ./run.sh nb04
```

Then, once both are done (NB05 refuses to start on empty pack directories
rather than "succeeding" with every section skipped):

```bash
./run.sh nb05
./run.sh nb06
```

Monitoring, from anywhere: `tail -f ~/thesis/logs/*.log`.

## 5. Collect

```bash
rsync -avP $USER@<cluster>:~/thesis/data/clean/ ./data/clean/
rsync -avP $USER@<cluster>:'~/thesis/*_executed.ipynb' ./runs/
rsync -avP $USER@<cluster>:~/thesis/logs/ ./logs/
```

You get the executed notebooks (all outputs inline), every `nb0*.parquet`
table, the `.npz` surface packs, and `figures/*.pdf` ready for LaTeX.

## Restarting after a crash

NB03 checkpoints every 25 days per shard and each shard is independent —
relaunch only the failed year with `./run.sh nb03 2021`. NB04 supports
`NB04_RESUME=1`, which reloads the saved state dicts and re-runs only
evaluation, audit and export.

## Cluster etiquette

BLAS threads are capped per shard so parallel shards don't thrash the node, and
`CUDA_VISIBLE_DEVICES=""` is set on every CPU notebook so they never hold a GPU
slot someone else needs. Kill your tmux sessions once the runs are collected.
