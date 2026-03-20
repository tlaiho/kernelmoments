# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

kernelmoments provides data visualization and estimation tools based on fast kernel regression (Nadaraya-Watson), using KernelTree from `kernelboost` as the estimation engine. The focus is data exploration rather than prediction. Each kernel fit gets its own bandwidth via LOO-CV.

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

All estimators follow sklearn-style `fit()`/`predict()` and share KernelTree parameters via `BaseEstimator`.

- **MeanEstimator** — E[Y|X]. Thin wrapper over KernelTree.
- **VarianceEstimator** — Var[Y|X]. Fan & Yao double kernel: pass 1 fits the mean and gets residuals, pass 2 fits E[ε²|X] with a separately optimized (typically larger) bandwidth. Exposes `mean_estimator_` for the pass-1 model.
- **CovarianceEstimator** — Cov[Y,Z|X]. Three kernel regressions: E[Y|X], E[Z|X], then E[εʸ·εᶻ|X]. Also provides `fit_correlation()`/`predict_correlation()` for conditional correlation.
- **`cross_fit_residualize()`** — Chernozhukov et al. (2018) cross-fitted residualization for partialling out confounders (used by `CovarianceEstimator` when `partial_out` is set).

### Plotter (`kernelmoments/plotter.py`)

`Plotter` class wraps the estimators for visualization with pandas/polars DataFrame support. `plot_relationship()` is a standalone convenience function for quick NumPy-based plots. Supports mean (with ±1.96·√Var bands), variance, covariance, and correlation moments. Caches fitted estimators.

### Utils (`kernelmoments/utils.py`)

`validate(X, *ys)` — input validation: ensures X is 2D (reshapes 1D), checks for NaN, verifies sample count alignment, casts to float32.

### Key dependency

`kernelboost` provides `KernelTree` (tree-partitioned kernel regression) and `KernelEstimator` (bandwidth optimization via LOO-CV). All estimation in this library delegates to these.

## Test Conventions

Tests use pytest with synthetic data fixtures (seed=42, n=2000). Fast tree params for tests: `min_sample=100, max_sample=500, max_depth=2, search_rounds=5`. Note: `tests/` is in `.gitignore` and not committed to the repo.
