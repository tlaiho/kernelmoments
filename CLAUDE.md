# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

kernelmoments provides data visualization and estimation tools based on fast kernel regression (Nadaraya-Watson), using KernelTree from `kernelboost` as the estimation engine. The focus is data exploration rather than prediction. Each kernel fit gets its own bandwidth, selected via pilot-AICc by default (favors oversmoothing, which suits visualization).

## Commands

Always use the right virtual environment: source ~/KB/bin/activate

```bash
# Run all tests (tests/ is gitignored; create locally if needed)
pytest tests/

# Run a single test file
pytest tests/estimator_unit_tests.py

# Run a specific test class or method
pytest tests/estimator_unit_tests.py::TestMeanEstimator::test_fit_predict

# Install in development mode
pip install -e .
```

No linting or formatting tools are configured.

## Architecture

### Estimators (`kernelmoments/estimators.py`)

All estimators follow sklearn-style `fit()`/`predict()` and share KernelTree parameters via `BaseEstimator`. `seed: int | None = None` on `BaseEstimator` is forwarded to KernelTree; the default is an unseeded (irreproducible) bandwidth search — pass an int for deterministic fits. Tests pass `seed=42`.

- **MeanEstimator** — E[Y|X]. Thin wrapper over KernelTree.
- **VarianceEstimator** — Var[Y|X]. Fan & Yao double kernel: pass 1 fits the mean and gets residuals, pass 2 fits E[ε²|X] with a separately optimized (typically larger) bandwidth. Exposes `mean_estimator_` for the pass-1 model.
- **CovarianceEstimator** — Cov[Y,Z|X]. Three kernel regressions: E[Y|X], E[Z|X], then E[εʸ·εᶻ|X]. Also provides `fit_correlation()`/`predict_correlation()` for conditional correlation.
- **Leverage correction** — `correction: bool = True` on `BaseEstimator` (stored outside `tree_params`). Variance/Covariance estimators smooth leave-one-out residuals `(y − ŷ)/(1 − S_ii)` with `S_ii` from `KernelTree.self_weights()` (kernelboost ≥ 0.4.0), removing own-observation suppression. `correction=False` restores raw in-sample residuals. See `residual_correction_plan.md`.
- **Cross-fitting** — `cross_fit: int = 0` on `BaseEstimator` (outside `tree_params`; 0 = off, k ≥ 2 = k-fold, 1/negative raise). When set, in-sample residuals/fitted values are out-of-fold via `cross_fit_residualize` (fold seed = estimator's `seed`), superseding `correction`; `predict()` always comes from the full-sample fits. `MeanEstimator` additionally stores OOF `training_predictions_`/`residuals_`.
- **`cross_fit_residualize()`** — Chernozhukov et al. (2018) cross-fitted residualization. Used by the estimators' `cross_fit` machinery and by `Plotter`'s `partial_out` path (partialling out lives in the plotter, not the estimators: it residualizes the conditioning variable too, which has no well-defined `predict()` semantics at the estimator level).

### Plotter (`kernelmoments/plotter.py`)

`Plotter` class wraps the estimators for visualization with pandas/polars DataFrame support. `plot_relationship()` is a standalone convenience function for quick NumPy-based plots. Supports mean (with ±1.96·√Var bands), variance, covariance, and correlation moments. Caches fitted estimators.

Estimator-level args (`correction`, `cross_fit`) are named parameters on both `Plotter` and `plot_relationship`, kept in `Plotter.estimator_params` separately from `tree_params` (which holds only KernelTree kwargs — the split matters because `cross_fit_residualize` splats `tree_params` into its fold fits). `Plotter.seed` (default 42) seeds subsampling, partial-out folds, and kernel fits, so plots are reproducible by default. `partial_out=W` residualizes y, x (and z) on W — FWL-style; the fold count is `cross_fit` when set, else 3. `bands=True` reuses the variance estimator's internal mean fit instead of fitting a separate `MeanEstimator`.

### Utils (`kernelmoments/utils.py`)

`validate(X, *ys)` — input validation: ensures X is 2D (reshapes 1D), checks for NaN, verifies sample count alignment, casts to float32.

### Key dependency

`kernelboost` provides `KernelTree` (tree-partitioned kernel regression) and `KernelEstimator` (bandwidth optimization). All estimation in this library delegates to these.

## Test Conventions

Tests use pytest with synthetic data fixtures (seed=42, n=2000). Fast tree params for tests: `min_sample=100, max_sample=500, max_depth=2, search_rounds=5`. Note: `tests/` is in `.gitignore` and not committed to the repo.
