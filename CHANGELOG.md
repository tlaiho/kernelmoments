# Changelog

## 0.2.0 (2026-07-30)

- Default bandwidth optimization is now AICc (`precision_method="pilot-aicc"`)
- Leverage/LOO residual correction for `VarianceEstimator` and `CovarianceEstimator` (`correction=True` by default; requires kernelboost ≥ 0.4.0) — removes own-observation suppression of in-sample residuals
- `cross_fit=k` on all estimators: out-of-fold residuals, superseding `correction`; `predict()` stays full-sample
- `seed` on `BaseEstimator` (default `None`) for reproducible fits
- `Plotter`: `correction`/`cross_fit`/`seed` as named args; `partial_out` fold count follows `cross_fit`; `bands=True` reuses the variance estimator's mean fit
- Fixed `NameError` in `plot_relationship` (`self.style` in a standalone function)
- Fixed `partial_out` display: plots now shown in residualized space (scaler inverse-transform no longer applies)§
- Fixed `Plotter.x_scaler`: cloned per x column instead of shared
- Renamed `unconditional*()` → `mean_conditional_{variance,covariance,correlation}()`

## 0.1.0 (2025-03-10)

Initial release.

- Core estimators: `MeanEstimator`, `VarianceEstimator`, `CovarianceEstimator`
- Conditional correlation via `CovarianceEstimator.fit_correlation()` / `predict_correlation()`
- Cross-fitted residualization for confounder control (Chernozhukov et al. 2018)
- `Plotter` class for DataFrame-based visualization with estimator caching
- `plot_relationship()` for one-call plotting from NumPy arrays
- Pandas and Polars DataFrame support (auto-detected, no hard dependency)
- Optional GPU acceleration via CuPy
