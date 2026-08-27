# Production run on the Telecom cluster

Everything below assumes `~/thesis` on the shared filesystem `/home/infres/$USER`,
which is visible from **every** node (`lame*`, `gpu*`). That is what makes the split
work: NB03 on a CPU node and NB04 on a GPU node write into the *same* `data/clean/`,
so you transfer once and collect once.

## What changed in the notebooks

The cluster does not run `notebooks/` directly, it runs a patched copy built from
the repo root:

```bash
mkdir -p build && cp notebooks/0{3,4,5,6}_*.ipynb build/
python cluster/patch_notebooks.py build
```

`patch_notebooks.py` is idempotent and inserts one bootstrap cell per notebook,
so `build/` is disposable and gitignored — regenerate it, never edit it. The
patch:

- monkey-patches `plotly.graph_objects.Figure.show`, so **all 18 existing `fig.show()`
  calls now write the figure to disk** as vector PDF (plus interactive HTML) in
  `data/clean/figures/`. Without this the 30-hour run produces no file you can
  `\includegraphics` into the thesis;
- makes `fig.show()` a no-op display-wise under `THESIS_HEADLESS=1`, so a missing
  browser front-end cannot kill the run mid-way;
- prefixes figure names per notebook and per shard, so parallel shards never
  overwrite each other;
- writes a `*_figure_manifest.parquet` mapping each file to its figure title.

Your notebook logic is untouched. Nothing was hardcoded: everything stays driven
by the env vars you already built.

## 0. Transfer (once)

The scripts expect a **flat** `$THESIS_ROOT` on the server: notebooks and `.sh`
files side by side, not in `build/` and `cluster/`.

```bash
rsync -avP build/*.ipynb cluster/*.sh cluster/requirements_cluster.txt \
      $USER@ssh.enst.fr:/home/infres/$USER/thesis/
rsync -avP data/ $USER@ssh.enst.fr:/home/infres/$USER/thesis/data/
```

Needed on the server: the three patched notebooks, the `.sh` scripts,
`data/clean/option_prices_clean.parquet`, the two NB02 outputs
(`benchmark_svi_slices_full.parquet`, `benchmark_ssvi_days_full.parquet`),
and `data/raw/VIX_History.csv`.

## 1. Environment (once)

```bash
ssh lame25
cd /home/infres/$USER/thesis
python3 -m pip install virtualenv --user
virtualenv venv --python=python3
source venv/bin/activate
pip install -r requirements_cluster.txt
# torch only matters on the GPU node; match the CUDA version nvidia-smi reports
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`kaleido==0.2.1` is pinned deliberately: newer kaleido needs a system Chrome you
cannot install without root.

## 2. Smoke test (5 minutes — do not skip)

```bash
ssh gpu1
cd /home/infres/$USER/thesis && source venv/bin/activate
./run_smoke_test.sh
```

Runs the full NB03 → NB04 → NB05 chain on a handful of days, in a sandbox
directory. Confirms paths, packages, CUDA and figure export. If the figure count
at the end is zero, fix that before launching production.

## 3. Production

Two nodes, in parallel, each in its own tmux session.

**NB03 on a CPU node.** NB03 is sequential per process, so one process would use
one core out of 128. It is sharded by year instead: 8 parallel processes, roughly
30 h → 4 h.

```bash
ssh lame25
cd /home/infres/$USER/thesis && source venv/bin/activate
tmux new -s nb03
./run_nb03_shards.sh
# ctrl+B then D  -> detach, close the laptop
```

**NB04 on a GPU node**, at the same time:

```bash
ssh gpu1
nvidia-smi                       # pick a free GPU index
cd /home/infres/$USER/thesis && source venv/bin/activate
tmux new -s nb04
CUDA_DEVICE=0 ./run_nb04_gpu.sh
# ctrl+B then D
```

NB04 trains 4 families × 3 regimes, including `deeponet_prior` and `gno_prior`
(the SSVI-prior-embedded operators, `NB04_RUN_PRIOR=1`).

**Monitoring**, from anywhere:

```bash
ssh lame25 'tail -f /home/infres/'$USER'/thesis/logs/nb03_2018.log'
ssh gpu1   'nvidia-smi'
```

## 4. NB05, once both are done

```bash
ssh lame25
cd /home/infres/$USER/thesis && source venv/bin/activate
tmux new -s nb05
./run_nb05.sh
```

It refuses to start if either pack directory is empty, rather than "succeeding"
with every section silently skipped.

## 5. Collect everything

From your laptop:

```bash
rsync -avP $USER@ssh.enst.fr:/home/infres/$USER/thesis/data/clean/ ./data/clean/
rsync -avP $USER@ssh.enst.fr:'/home/infres/'$USER'/thesis/*_executed.ipynb' ./
rsync -avP $USER@ssh.enst.fr:/home/infres/$USER/thesis/logs/ ./logs/
```

You get the executed notebooks (all outputs inline), every `nb0*.parquet`, the
`.npz` packs, and `data/clean/figures/*.pdf` ready for LaTeX.

## Restarting after a crash

NB03 checkpoints every 25 days per shard, and each shard is independent: relaunch
only the failed year with `YEARS="2021" ./run_nb03_shards.sh`. NB04 supports
`NB04_RESUME=1`, which reloads the saved state dicts and re-runs only
evaluation, audit and export.

## Cluster etiquette

BLAS threads are capped at 4 per shard so 8 shards do not thrash 128 cores, and
`CUDA_VISIBLE_DEVICES=""` is set on the CPU notebooks so they never hold a GPU
slot someone else needs. Kill your tmux sessions when the runs are collected.
