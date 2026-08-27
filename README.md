# Arbitrage-Free Implied Volatility Surface Construction: From parametric models to neural operators

MSc thesis in Mathematics and Finance, Imperial College London, 2025 to 2026.

The full thesis is available in [`thesis/These_Hamza_Imperial.pdf`](thesis/These_Hamza_Imperial.pdf), together with the LaTeX sources in [`thesis/`](thesis/).

## Overview

An implied volatility surface is one of the main objects used in equity derivatives modelling. The difficulty is that the market never gives us the surface itself. What we observe is a sparse and irregular collection of option quotes, with strikes and maturities changing from one trading day to the next.

Before those quotes can be used for pricing, Greeks, local volatility calibration, stochastic volatility models or stress testing, they need to be turned into a smooth and complete surface.

That smoothing step is not only about obtaining a good numerical fit. The resulting surface also needs to respect static no-arbitrage conditions. In particular, it should avoid butterfly arbitrage across strikes and calendar arbitrage across maturities.

These constraints have direct financial consequences. Through the Breeden-Litzenberger relationship, butterfly arbitrage corresponds to a negative risk-neutral density. In Dupire's local volatility framework,

$$
\sigma_{\mathrm{loc}}^2 = \frac{\partial_\tau w}{g},
$$

violations can produce negative local variance and make the surface unusable for simulation without some form of repair.

The central question of this project is therefore simple:

**How do we build an implied volatility surface that fits the market well without losing the structural properties required for pricing and risk management?**

To study that question, the thesis compares three main families of models on the same S&P 500 options dataset from OptionMetrics, covering **1,926 trading days from 2018 to 2025**.

Every model is evaluated under the same protocol, using:

* held-out reconstruction error;
* a static-arbitrage audit based on the Durrleman condition;
* robustness to sparse quotes;
* behaviour across different market regimes.

The three model families are:

1. **Parametric models**, with SVI fitted maturity by maturity and SSVI fitted at surface level.
2. **Per-surface neural smoothers**, where a parametric prior is corrected by a neural network and arbitrage constraints are handled through penalties on exact automatic derivatives.
3. **Neural operators**, including DeepONet and graph neural operators, which learn the smoothing map itself and can be applied directly to unseen trading days.

## Main result: the collocation blind spot

The most important result of the thesis came from a discrepancy that initially looked contradictory.

During training, the deep smoother reported a calendar-arbitrage penalty of exactly zero on its collocation grid. However, when the same surface was checked independently on a finer audit grid, calendar violations appeared on **28% of the domain**.

Both measurements were correct.

The problem was not the derivative calculation or the implementation of the constraint. It was the grid on which the constraint was being checked.

Soft no-arbitrage penalties only control the model at the points where they are evaluated. The exponential maturity grid commonly used in the literature places many points where the butterfly condition is difficult to satisfy, but in this experiment it also created a gap of roughly half a year in a region where the calendar constraint needed additional coverage.

The model was therefore clean at the sampled nodes while violating the constraint between them.

This observation became the main thread of the thesis.

### Constraint-specific collocation grids

Butterfly and calendar constraints do not need the same sampling geometry.

Instead of forcing both constraints onto a single grid, the deep smoother uses a separate collocation grid for each one. A node-gap result is then used to understand how much violation can develop between neighbouring points.

On a controlled stress test, this change reduces material calendar violations from **34.4% to 0.0%**.

The improvement is not presented as free. Removing the violations comes with a measurable loss of fit on the stressed slice, which makes the trade-off between accuracy and structure explicit.

### The same problem appears in neural operators

For neural operators, the blind spot becomes more difficult because it exists in two directions.

A model can miss arbitrage:

* between the grid points where its constraints are evaluated;
* across unseen trading days that were never part of training.

The graph neural operator achieves the best raw reconstruction accuracy among the unconstrained operator models, but it is materially butterfly-violating on **355 of 386 out-of-sample days**, even though its training-grid audit is close to clean.

This shows that a low training penalty is not, by itself, a certificate of arbitrage freedom.

### A certified prior works better than penalties alone

The thesis then takes a different approach.

Instead of asking the neural network to learn both the surface and the no-arbitrage structure from soft penalties, it starts from an SSVI prior whose static arbitrage properties are certified on the working domain using the Gatheral-Jacquier conditions.

The neural operator only learns a bounded multiplicative correction around that prior.

With this construction, material violations disappear on **all 386 test days**. At the same time, reconstruction error improves from **2.69 to 1.32 volatility points**.

The result is important because the structural guarantee does not come at the expense of predictive accuracy in this experiment. The certified prior makes the learning problem easier while also protecting the model from the most damaging failures.

### Arbitrage violations are treated as pricing errors

The audit is also connected to downstream economic quantities rather than being treated as a purely mathematical diagnostic.

For the unconstrained operator surfaces:

* roughly **50% of the surface** can require repair before local-volatility simulation;
* implied densities can contain up to **230% negative mass**;
* residual signals deteriorate once realistic trading costs are introduced.

At the same time, a bid/ask-level scanner finds essentially **no executable static arbitrage across eight years of SPX quotes**.

The violations produced by the models should therefore be interpreted primarily as **model risk**, not as obvious trading opportunities in the underlying option market.

## Contributions

The thesis makes six main contributions.

### 1. A controlled comparison of three model families

Parametric surfaces, per-surface neural smoothers and neural operators are implemented on the same dataset of 1,926 S&P 500 trading days and evaluated with the same reconstruction protocol and arbitrage audit.

This makes it possible to compare the different approaches without changing the data, the evaluation procedure or the definition of a violation between experiments.

### 2. Replication and validation of the Gatheral-Jacquier framework

The SVI and SSVI calibration machinery is reproduced, including the raw, natural and jump-wings parameterisations.

The implementation matches the published reference values to approximately **1e-7**.

During that replication, the project also identifies and corrects a missing term in the published inverse transformation from natural parameters.

### 3. Identification of the collocation blind spot

The experiments show directly that a neural model can satisfy its no-arbitrage penalties at every collocation point while remaining materially arbitrage-violating between those points.

The thesis separates the training-node audit from an independent finer audit and introduces a materiality threshold so that small numerical effects can be distinguished from economically meaningful violations.

### 4. An arbitrage-free prior and a bound on the node gap

The SSVI prior is certified as statically arbitrage-free on the working domain rather than being accepted simply because no violation happened to appear numerically.

A node-gap result is also developed to explain how violations can emerge between collocation points and why a single grid is poorly suited to both butterfly and calendar constraints.

### 5. Extension to neural operators

The same audit methodology is applied to DeepONet and graph neural operators.

The experiments show that the collocation problem survives the move from one-surface-at-a-time neural smoothers to amortised operators. In that setting, generalisation across trading days becomes an additional source of structural error.

Embedding the certified SSVI prior into the operator removes material violations on all 386 out-of-sample days while improving reconstruction accuracy.

### 6. Economic consequences of structural violations

The final part of the study follows the effect of arbitrage violations into quantities that matter for pricing and hedging.

The analysis covers:

* Dupire local volatility;
* Breeden-Litzenberger risk-neutral densities;
* static-arbitrage tests at executable bid and ask prices;
* residual relative-value signals;
* VIX variance-swap replication;
* path-dependent barrier pricing and hedging.

In the controlled barrier experiment, the most accurate unconstrained operator can misprice the barrier option by **60%** relative to an independent Heston benchmark.

This illustrates a broader point: two surfaces can look almost identical under a vanilla fitting objective while behaving very differently once derivatives, densities or path-dependent prices are computed from them.

## Notebooks

The project is organised as a six-notebook pipeline.

| #  | Notebook                                                                         | Purpose                                                                                                                                                                                                                                                                                                                              | Main result                                                                                                                                                                                                                                                                                                                          |
| -- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 01 | [`01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb)                     | Cleans the OptionMetrics SPX quotes, applies the market filters, recovers forwards and discount factors, selects OTM options, builds log-moneyness and total-variance coordinates, and independently recomputes implied volatility.                                                                                                  | Produces the audited panel of 1,926 trading days used throughout the project. Every later experiment follows from the same cleaning rules.                                                                                                                                                                                           |
| 02 | [`02_svi_ssvi_benchmark.ipynb`](notebooks/02_svi_ssvi_benchmark.ipynb)           | Reproduces the Gatheral-Jacquier framework, including raw, natural and jump-wings SVI, exact parameter conversions, the Vogt smile and SSVI calibration with static no-arbitrage conditions.                                                                                                                                         | Matches published reference values to 1e-7 and identifies the missing term in the natural-parameter inverse. Per-slice SVI reaches **0.64 volatility points** of held-out error but produces butterfly violations on about **12% of slices**. Certified SSVI remains arbitrage-free at roughly three times the reconstruction error. |
| 03 | [`03_deep_smoother.ipynb`](notebooks/03_deep_smoother.ipynb)                     | Implements the derivative-constrained deep smoother using a certified SSVI prior multiplied by a positive tanh-MLP correction. It includes exact automatic derivatives, separate collocation grids, an independent audit grid, materiality thresholds, controlled ablations, sparsity experiments and the real-SPX production study. | Exposes the collocation blind spot. A zero training-node penalty can coexist with 28% violations on the audit grid. Constraint-specific grids reduce material violations from **34.4% to 0.0%** in the stress experiment.                                                                                                            |
| 04 | [`04_neural_operator.ipynb`](notebooks/04_neural_operator.ipynb)                 | Studies DeepONet and graph neural operators under real-only, synthetic-only and pretraining plus fine-tuning regimes. It also includes hull-based daily audits, discretisation tests, latency measurements and prior-embedded operator variants.                                                                                     | The graph neural operator gives the strongest unconstrained fit but is materially butterfly-violating on **355 of 386 test days**. The certified-prior variant removes all material violations and improves error from **2.69 to 1.32 volatility points**.                                                                           |
| 05 | [`05_downstream_economics.ipynb`](notebooks/05_downstream_economics.ipynb)       | Connects the structural audit to local volatility, densities, executable static-arbitrage tests, residual relative-value signals, detection power and VIX replication.                                                                                                                                                               | Shows that structural violations can require large repairs and generate severely distorted densities, while essentially no executable static arbitrage survives bid/ask costs in the raw SPX data.                                                                                                                                   |
| 06 | [`06_path_dependence_barrier.ipynb`](notebooks/06_path_dependence_barrier.ipynb) | Builds a controlled path-dependence experiment using an independent Heston benchmark, Dupire extraction, a barrier-aligned Crank-Nicolson PDE, a barrier dose-response study, paired delta hedging and digital-option diagnostics.                                                                                                   | Shows how a defect that is invisible at every quoted strike can still materially alter barrier prices and hedging P&L. These numerical results are marked as preliminary in the thesis because the corresponding numerical components had not yet been validated to the same standard as the rest of the project at submission time. |

## Repository structure

```text
notebooks/
    Six notebooks covering the complete empirical pipeline.

cluster/
    Scripts used to prepare and execute the computationally expensive notebooks on a compute cluster.

thesis/
    LaTeX sources, chapters, figures and the compiled MSc thesis.
```

The notebooks are the main experimental source of the project. The thesis provides the full mathematical motivation, methodology, proofs, results and interpretation.

## Running the computational pipeline

The first two notebooks contain the data preparation and parametric benchmark stages.

The later notebooks contain the more computationally expensive neural smoothing, neural operator and downstream pricing experiments.

The repository includes tooling in [`cluster/`](cluster/) for preparing those notebooks for headless execution. The patching script adds the bootstrap required for figure export without changing the canonical notebooks.

For example:

```bash
mkdir -p build
cp notebooks/0{3,4,5,6}_*.ipynb build/
python cluster/patch_notebooks.py build
```

The execution interface is provided through `run.sh`.

```bash
./run.sh smoke
./run.sh all
./run.sh nb03 2021
```

The smoke configuration is intended as an end-to-end validation of the pipeline before running the larger experiments.

The execution settings for NB03 to NB06 are controlled through environment variables using the corresponding `NB03_*`, `NB04_*`, `NB05_*` and `NB06_*` prefixes.

For additional execution details, see [`cluster/README.md`](cluster/README.md).

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

## Takeaway

The main lesson of the project is that fitting an implied volatility surface and controlling its arbitrage properties are not the same problem.

A neural model can look perfectly constrained at the points used during training and still fail between those points or on unseen market days. Increasing model capacity does not remove that issue.

The most reliable results in this study come from combining learning with structure. A certified SSVI prior provides the no-arbitrage foundation, while the neural model is left to learn the part where additional flexibility is actually useful.

In this setting, that combination gives a better fit, removes material static-arbitrage violations on the complete test set, and produces surfaces that remain usable for the downstream quantities that ultimately matter: densities, local volatility, pricing and hedging.


