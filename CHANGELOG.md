# Changelog

## 0.1.0 (2025-03-10)

Initial release.

- Core estimators: `MeanEstimator`, `VarianceEstimator`, `CovarianceEstimator`
- Conditional correlation via `CovarianceEstimator.fit_correlation()` / `predict_correlation()`
- Cross-fitted residualization for confounder control (Chernozhukov et al. 2018)
- `Plotter` class for DataFrame-based visualization with estimator caching
- `plot_relationship()` for one-call plotting from NumPy arrays
- Pandas and Polars DataFrame support (auto-detected, no hard dependency)
- Optional GPU acceleration via CuPy
