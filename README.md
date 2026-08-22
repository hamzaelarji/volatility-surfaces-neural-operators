# volatility-surfaces-neural-operators

MSc thesis: arbitrage-aware smoothing and neural-operator models of the SPX
implied volatility surface.

## Layout

```
notebooks/   the pipeline — five notebooks, run in order, outputs inline
cluster/     tooling to execute NB03–NB05 headless on the Telecom cluster
figures/     figures exported from the notebooks
thesis/      LaTeX sources (main.tex, chapters/, bibliography.bib)
```

`data/` and `logs/` are local-only (gitignored): raw quotes, parquet packs,
`.npz` model packs and run logs are never committed.

## The pipeline

| Notebook | Role |
|---|---|
| `01_data_cleaning.ipynb` | raw SPX quotes → cleaned panel |
| `02_svi_ssvi_benchmark.ipynb` | SVI / SSVI parametric baselines |
| `03_deep_smoother.ipynb` | arbitrage-penalised deep smoother |
| `04_neural_operator.ipynb` | DeepONet / GNO operators, incl. SSVI-prior variants |
| `05_downstream_economics.ipynb` | hedging and P&L evaluation |

NB01–NB02 run on a laptop. NB03–NB05 are the expensive ones (~30 h) and are
meant for the cluster — see [cluster/README.md](cluster/README.md).

## Running NB03–NB05 remotely

The notebooks in `notebooks/` are the canonical source. The cluster runs a
*patched* copy: `cluster/patch_notebooks.py` inserts one bootstrap cell that
makes `fig.show()` headless-safe and writes every plotly figure to disk.

```bash
mkdir -p build && cp notebooks/0{3,4,5}_*.ipynb build/
python cluster/patch_notebooks.py build
```

Then rsync `build/` plus `cluster/*.sh` to `~/thesis` on the cluster and follow
[cluster/README.md](cluster/README.md). The patch is idempotent, so the patched
copies are disposable and are not tracked here.

## Thesis figures

`thesis/main.tex` pulls figures from `thesis/figures/` via `\maybegraphics`,
which falls back to a placeholder box when a file is missing. The names it
expects (`nb02_buckets.png`, `nb03_smiles.png`, …) are *not* the names the
notebooks currently export into `figures/` (`nb02_c41_0.png`, …) — that mapping
still has to be done before the final build.
