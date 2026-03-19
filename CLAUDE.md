# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

kernelmoments provides data visualization and estimation tools based on fast kernel regression (Nadaraya-Watson), using KernelTree from `kernelboost` as the estimation engine. The focus is data exploration rather than prediction. Each kernel fit gets its own bandwidth via LOO-CV.

## Commands

Always use the right virtual environment: source /home/tuomas/ml/bin/activate

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/estimator_unit_tests.py

# Run a specific test class or method
pytest tests/estimator_unit_tests.py::TestMeanEstimator
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
- **CovarianceEstimator** — Cov[Y,Z|X]. Three kernel regressions: E[Y|X], E[Z|X], then E[εʸ·εᶻ|X]. Supports partial covariance via `partial_out` (cross-fitted residualization).

### Plotter (`kernelmoments/plotter.py`)

`Plotter` class wraps the estimators for visualization. `plot_relationship()` handles mean (with ±1.96·√Var bands), variance, and covariance moments. Supports pandas/polars DataFrames with string column names. Caches fitted estimators.

### Utils (`kernelmoments/utils.py`)

- `validate(X, *ys)` — input validation, NaN checking, shape alignment
- `DataFrameAdapter` — lazy-import adapter for pandas/polars with uniform `get_column()`/`get_columns()` interface
- `resolve_columns(data, **kwargs)` — extract named columns from DataFrames

### Key dependency

`kernelboost` provides `KernelTree` (tree-partitioned kernel regression) and `KernelEstimator` (bandwidth optimization via LOO-CV). All estimation in this library delegates to these.

## Test Conventions

Tests use pytest with synthetic data fixtures (seed=42, n=2000). Fast tree params for tests: `min_sample=100, max_sample=500, max_depth=2, search_rounds=5`.
