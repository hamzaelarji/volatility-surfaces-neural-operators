# Arbitrage-Free Implied Volatility Surface Construction: From Parametric Models to Neural Operators

MSc thesis, Mathematics and Finance, Imperial College London (2025–2026).
Full text: [thesis/These_Hamza_Imperial.pdf](thesis/These_Hamza_Imperial.pdf) · LaTeX sources in [thesis/](thesis/).

## The problem

The implied volatility surface is the central object of equity derivatives markets, yet it is
never observed directly: the market publishes a sparse, noisy, irregular cloud of option
quotes whose strike/maturity grid changes every day. Everything downstream — marking books,
Greeks, calibrating local and stochastic volatility models, stress testing — consumes not the
quotes but a *smoothed and completed surface* built from them.

The construction is constrained: the surface must be free of **static arbitrage** — butterfly
arbitrage across strikes and calendar arbitrage across maturities. Violations are not
cosmetic: through Breeden–Litzenberger, butterfly arbitrage means a negative risk-neutral
density; through Dupire's formula $\sigma_{\mathrm{loc}}^2 = \partial_\tau w / g$, violations
become negative local variance, and no Monte Carlo path can be drawn until the surface is
"repaired". There is therefore a genuine trade-off between fit and structure.

The thesis compares the three families of the literature on a single unified dataset —
S&P 500 options (OptionMetrics), **1,926 trading days, 2018–2025** — under one evaluation
protocol: hold-out reconstruction error, a static-arbitrage audit built on the Durrleman
condition, robustness to quote sparsity, and behaviour across market regimes.

1. **Parametric models** (SVI per slice, SSVI surface-wide): closed forms, explicit
   no-arbitrage conditions, limited flexibility.
2. **Per-surface neural smoothers** (Ackerer et al.; Hoshisashi et al.): a parametric prior
   times a neural corrector, arbitrage controlled through soft penalties on the network's
   exact autodiff derivatives.
3. **Neural operators** (Operator Deep Smoothing; DeepONet, graph neural operators): learn
   the smoothing *map* itself and apply to unseen days without retraining.

## The central finding: the collocation blind spot

While training the deep smoother, the calendar penalty was **exactly zero** on its training
grid — while an independent, finer audit grid showed calendar violations on **28% of the
domain**. Both numbers were true: soft constraints are enforced only *where they are
sampled*, and the exponential maturity spacing the literature recommends (dense where the
butterfly constraint bites) left a ~half-year gap exactly where the calendar constraint
needed nodes. The surface violated *between* the nodes.

The thesis turns this observation into its main line of argument:

- **Diagnosis and repair.** The two constraints have opposite grid needs, so each gets its
  own collocation grid; a node-gap theorem bounds the violation that can hide between
  neighbouring nodes. On a controlled stress test, material calendar violations fall from
  **34.4% to 0.0%** — at a measured, honest price in fit on the stressed slice.
- **The blind spot is transversal.** For a neural operator the pathology becomes
  two-dimensional: blind between grid nodes *and* across days. The most accurate model of
  the study (the graph neural operator) is materially butterfly-violating on **355 of 386
  out-of-sample days** while its own training-grid audit reads near zero.
- **Certificates beat penalties.** The Gatheral–Jacquier conditions are used
  *constructively*: the SSVI prior is **proved** statically arbitrage-free on its working
  domain (endpoint certification), not luckily so. Embedding that certified prior inside the
  operator — the network learns only a bounded multiplicative correction — removes **every
  material violation on all 386 test days** while *improving* the fit from 2.69 to 1.32
  vol pts.
- **The audit is priced.** Violations become pricing objects: up to ~50% of the dirty
  operator surface requires repair before simulation, implied densities carry up to 230%
  negative mass, and residual information degrades measurably. Meanwhile a bid/ask-level
  scanner finds essentially **no executable static arbitrage in eight years of SPX quotes**:
  the violations are model risk, not a trading opportunity.

## Contributions

As stated in the thesis (Chapter 1):

1. **A controlled comparison of the three model families.** Parametric surfaces,
   per-surface neural smoothers and neural operators are implemented on the same
   S&P 500 dataset of 1,926 trading days (2018–2025) and evaluated under the same
   protocol and the same arbitrage audit. To our knowledge, no previous study
   compares all three families in the same setting.
2. **Replication and validation of the Gatheral–Jacquier framework.** The
   no-arbitrage conditions and calibration procedures are reproduced, matching the
   results of the original paper to 1e-7 — and the replication uncovers and
   **corrects a missing term in the published inverse of the natural-parameter
   transformation** (thesis, Remark on the GJ typo).
3. **Identification and repair of the collocation blind spot.** No-arbitrage
   penalties can miss violations between the points where they are evaluated. The
   source of the problem is explained and repaired with a separate collocation
   grid per constraint; the node-vs-audit gap and a materiality threshold are
   introduced to measure it.
4. **An arbitrage-free prior and a bound on the collocation gap.** The SSVI prior
   is *proved* statically arbitrage-free on its working domain (endpoint
   certification), so its protective role is guaranteed rather than observed; a
   node-gap theorem explains how violations develop between collocation points and
   why one grid cannot serve both constraints.
5. **Extension to neural operators and a prior-based solution.** The blind spot
   reappears in DeepONet and graph neural operators, both between grid points and
   on unseen trading days. Embedding the certified prior in the operator — the
   network learns only a bounded correction — removes material violations on all
   386 out-of-sample days while also improving the fit.
6. **Economic consequences of arbitrage violations.** Violations propagate into
   Dupire local volatility, Breeden–Litzenberger densities, and the pricing and
   hedging of a path-dependent barrier option under an independent Heston
   benchmark — the most accurate unconstrained operator misprices the barrier by
   60%. A static-arbitrage scanner on raw bid/ask quotes and a detection-power
   analysis show these violations are model risk, not trading opportunities.

## The notebooks

The pipeline is six notebooks, run in order. NB01–NB02 carry their outputs inline; NB03–NB06
are stored clean (small cells, markdown narrative, no outputs) — their results live in the
executed copies collected from the cluster under `runs/` (local-only).

| # | Notebook | What it does | What it shows |
|---|---|---|---|
| 01 | `01_data_cleaning.ipynb` | OptionMetrics SPX quotes → cleaned panel: filters, forward/discount recovery, OTM selection, log-moneyness/total-variance coordinates, independent IV re-computation | A reproducible, audited dataset of 1,926 days (2018–2025); every later number traces back to these cleaning rules |
| 02 | `02_svi_ssvi_benchmark.ipynb` | Full Gatheral–Jacquier replication: SVI in raw / natural / jump-wings parameterisations with exact conversions, Vogt smile, SSVI calibration with the GJ no-arbitrage certificates | Replication matches the published values to 1e-7 and fixes a missing term in the published natural-parameter inverse. The benchmark range: per-slice SVI fits held-out quotes to **0.64 vp** but leaks butterfly arbitrage on **12% of slices**; certified SSVI is arbitrage-free and pays ~3× the error |
| 03 | `03_deep_smoother.ipynb` | Derivative-constrained deep smoother: certified SSVI prior × positive tanh-MLP corrector, exact autodiff derivatives, soft butterfly/calendar penalties on **constraint-specific collocation grids**, independent audit grid, materiality thresholds, equal-epoch ablations, a gated stress test, a leak-free paired sparsity study, and the day-by-day real-SPX production driver | The **collocation blind spot**: penalty exactly zero at its nodes, 28% violations on the audit grid; repaired by constraint-specific grids (34.4% → 0.0% material). On clean data the certified prior does the protective work (penalty gradients exactly zero); sparsity robustness is inherited from prior-calibration robustness |
| 04 | `04_neural_operator.ipynb` | Neural operators: DeepONet and graph neural operator (GNO), three training regimes (real-only / synthetic-only / pretrain+fine-tune), the same node-vs-audit methodology plus per-day hull audits, fit gates, discretization-invariance and latency studies — and the **prior-embedded variants** (per-day certified SSVI prior, zero-initialised bounded correction) | The blind spot is **transversal**: the GNO is the accuracy leader and materially butterfly-violating on 355/386 unseen days while its training-grid audit reads clean. Embedding the certified prior removes all material violations on 386/386 days **and** improves fit (2.69 → 1.32 vp) |
| 05 | `05_downstream_economics.ipynb` | Prices the audit: Dupire local volatility (repair rate, MC repricing), Breeden–Litzenberger densities, an executable static-arbitrage scanner at bid/ask, a cost-aware residual relative-value backtest on a common universe, a detection-power frontier calibrated on real spreads and measured noise, VIX variance-swap replication | Violations as pricing objects: ~50% of the dirty operator surface needs repair; up to 230% negative density mass; the clean surface beats its λ=0 dirty counterfactual in every matched backtest cell. **No executable arbitrage** survives the touch in 8 years of quotes, and the frontier shows residuals must exceed ~3 vp with multi-day persistence to clear round-trip costs — clean surfaces matter for marking and market-making, not taker-side alpha |
| 06 | `06_path_dependence_barrier.ipynb` | Path-dependence under a controlled experiment: an independent **Heston truth engine** (characteristic-function vanillas + validated MC), clean vs dirty surfaces whose defect is *exactly zero on every quoted strike* (vanilla RMSE identical by construction), Dupire extraction with a vega-conditioning mask, a barrier-aligned Crank–Nicolson PDE with a three-way error budget, a barrier dose–response sweep, paired delta-hedging, digitals, and a gated section on the real NB03/NB04 packs | A defect **invisible to the vanilla objective** moves a down-and-out call's price without bound in the defect amplitude and generates a pure "wrong Greeks" hedging P&L — the cost of a blind spot is model risk on path-dependent products. *Per the thesis, these results are preliminary: the numerical components had not yet been verified to the standard of the rest at submission time* |

## Repository layout

```
notebooks/   the pipeline — six notebooks, run in order
cluster/     tooling to execute NB03–NB06 headless on a compute cluster (tmux + nbconvert)
thesis/      LaTeX sources (main.tex, chapters/, figures/) and the compiled PDF
```

Local-only, gitignored: `data/` (raw quotes, parquet tables, `.npz` surface packs), `logs/`
(cluster run logs) and `runs/` (executed notebook copies collected from the cluster).

## Reproducing

NB01–NB02 run on a laptop. NB03–NB06 are the expensive ones (the production run is ~30 h)
and are meant for a cluster — see [cluster/README.md](cluster/README.md).

The notebooks in `notebooks/` are the canonical source. The cluster runs a *patched* copy:
`cluster/patch_notebooks.py` inserts one bootstrap cell per notebook that makes `fig.show()`
headless-safe and exports every figure to disk:

```bash
mkdir -p build && cp notebooks/0{3,4,5,6}_*.ipynb build/
python cluster/patch_notebooks.py build
```

The patch is idempotent and `build/` is disposable (gitignored). Then rsync `build/` plus
the `cluster/` files to a flat `$THESIS_ROOT` on the cluster and drive everything through
the single entry point:

```bash
./run.sh smoke        # sandboxed end-to-end check (minutes) — do not skip
./run.sh all          # smoke -> NB03 -> NB04 -> NB05 on one node, one launch
./run.sh nb03 2021    # or piecewise: relaunch a single NB03 year shard, etc.
```

Every knob is env-driven (`NB03_*`, `NB04_*`, `NB05_*`, `NB06_*`), so staging and
production runs differ only in environment variables.

## Key references

The papers the project is built on:

**Parametric surfaces and static arbitrage**
- J. Gatheral. *A Parsimonious Arbitrage-Free Implied Volatility Parameterization…* (SVI). Global Derivatives, 2004.
- J. Gatheral, A. Jacquier. *Arbitrage-Free SVI Volatility Surfaces*. Quantitative Finance 14(1), 2014 — the SSVI framework, replicated and certified here.
- M. R. Fengler. *Arbitrage-Free Smoothing of the Implied Volatility Surface*. Quantitative Finance 9(4), 2009.
- N. Kahalé. *An Arbitrage-Free Interpolation of Volatilities*. Risk 17(5), 2004.
- R. W. Lee. *The Moment Formula for Implied Volatility at Extreme Strikes*. Mathematical Finance 14(3), 2004.

**Neural smoothing and soft constraints**
- D. Ackerer, N. Tagasovska, T. Vatter. *Deep Smoothing of the Implied Volatility Surface*. NeurIPS, 2020 — the prior × corrector architecture of NB03.
- K. Hoshisashi, C. E. Phelan, P. Barucca. *No-Arbitrage Deep Calibration for Volatility Smile and Skewness*. arXiv:2310.16703, 2023 — exact autodiff derivatives in the penalties.
- M. Chataigner, S. Crépey, M. Dixon. *Deep Local Volatility*. Risks 8(3), 2020 — the "constraints act only on the grid" caveat this thesis quantifies.
- A. G. Baydin, B. A. Pearlmutter, A. A. Radul, J. M. Siskind. *Automatic Differentiation in Machine Learning: a Survey*. JMLR 18(153), 2018.

**Neural operators**
- R. Wiedemann, A. Jacquier, L. Gonon. *Operator Deep Smoothing for Implied Volatility*. ICLR, 2025 — the operator setting of NB04.
- L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis. *Learning Nonlinear Operators via DeepONet…*. Nature Machine Intelligence, 2021.
- Z. Li, N. Kovachki, et al. *Neural Operator: Graph Kernel Network for PDEs*. arXiv:2003.03485, 2020.
- N. Kovachki, Z. Li, et al. *Neural Operator: Learning Maps Between Function Spaces…*. JMLR 24(89), 2023.

**Pricing the audit**
- D. T. Breeden, R. H. Litzenberger. *Prices of State-Contingent Claims Implicit in Option Prices*. Journal of Business 51(4), 1978.
- B. Dupire. *Pricing with a Smile*. Risk 7(1), 1994.

The full bibliography is in [thesis/chapters/bibliography.tex](thesis/chapters/bibliography.tex).
