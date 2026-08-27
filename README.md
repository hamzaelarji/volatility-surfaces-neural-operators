# volatility-surfaces-neural-operators

MSc thesis: arbitrage-aware smoothing and neural-operator models of the SPX
implied volatility surface.

## Layout

```
notebooks/   the pipeline — six notebooks, run in order
cluster/     tooling to execute NB03–NB06 headless on the Telecom cluster
thesis/      LaTeX sources — main.tex, chapters/, figures/
```

Local-only, gitignored: `data/` (raw quotes, parquet packs, `.npz` model packs),
`logs/` (cluster run logs) and `runs/` (the `*_executed.ipynb` collected back
from the cluster).

## The pipeline

| Notebook | Role |
|---|---|
| `01_data_cleaning.ipynb` | raw SPX quotes → cleaned panel |
| `02_svi_ssvi_benchmark.ipynb` | SVI / SSVI parametric baselines |
| `03_deep_smoother.ipynb` | arbitrage-penalised deep smoother |
| `04_neural_operator.ipynb` | DeepONet / GNO operators, incl. SSVI-prior variants |
| `05_downstream_economics.ipynb` | Dupire local vol, densities, trading the violations, VIX replication |
| `06_path_dependence_barrier.ipynb` | what an arbitrage-dirty surface costs: Heston truth engine, barrier dose–response, delta hedging, digitals |

NB01–NB02 run on a laptop. NB03–NB06 are the expensive ones (~30 h) and are
meant for the cluster — see [cluster/README.md](cluster/README.md).

NB01–NB02 carry their outputs inline. NB03–NB06 are stored **clean** (small
cells, markdown narrative, no outputs): their results live in the executed
copies collected back from the cluster under `runs/` (local-only).

## Running NB03–NB06 remotely

The notebooks in `notebooks/` are the canonical source. The cluster runs a
*patched* copy: `cluster/patch_notebooks.py` inserts one bootstrap cell that
makes `fig.show()` headless-safe and writes every plotly figure to disk.

```bash
mkdir -p build && cp notebooks/0{3,4,5,6}_*.ipynb build/
python cluster/patch_notebooks.py build
```

The patch is idempotent, so `build/` is disposable and gitignored. Then rsync
`build/` plus `cluster/*.sh` to `~/thesis` on the cluster and follow
[cluster/README.md](cluster/README.md) — note the scripts expect a *flat*
`$THESIS_ROOT`, notebooks and `.sh` side by side.

`cluster/patch_gno_prior.py` is a separate one-off patch that adds
`NB04_SKIP_GNO_PRIOR`, so `gno_prior` can be skipped while `deeponet_prior`
still runs.

## Thesis

`thesis/main.tex` pulls figures from `thesis/figures/` via `\maybegraphics`,
which degrades to a placeholder box rather than failing when a file is missing.
All 36 figures it references are present.
