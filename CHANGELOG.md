# Changelog

## 0.2.1 (2026-10-27)

- `RankTransformer` from `kernelboost` re-exported as `kernelmoments.RankTransformer`
- Removed the `gpu` extra (it pinned `cupy-cuda12x`). Install the CuPy wheel matching your CUDA version instead, e.g. `cupy-cuda13x`
- Readme and docstring wording updated. 


## 0.2.0 (2026-07-30)

- Default bandwidth optimization is now AICc (`precision_method="pilot-aicc"`)
- Leverage/LOO residual correction for `VarianceEstimator` and `CovarianceEstimator`. The correction removes own-observation suppression of in-sample residuals
- `cross_fit=k` on all estimators: out-of-fold residuals, superseding `correction`
- `Plotter`: `correction`/`cross_fit`/`seed` added as named args.
- Fixed `NameError` bug in `plot_relationship` (`self.style` in a standalone function)
- Fixed `partial_out` bug display: plots now shown in residualized space (scaler inverse-transform no longer applies)
- Fixed `Plotter.x_scaler` bug: cloned per x column instead of shared

## 0.1.0 (2025-03-10)

Initial release.

- Core estimators: `MeanEstimator`, `VarianceEstimator`, `CovarianceEstimator`
- Conditional correlation via `CovarianceEstimator.fit_correlation()` / `predict_correlation()`
- Cross-fitted residualization for confounder control (Chernozhukov et al. 2018)
- `Plotter` class for DataFrame-based visualization with estimator caching
- `plot_relationship()` for one-call plotting from NumPy arrays
- Pandas and Polars DataFrame support (auto-detected, no hard dependency)
- Optional GPU acceleration via CuPy
