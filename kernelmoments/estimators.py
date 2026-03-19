import numpy as np
from kernelboost.tree import KernelTree
from .utils import validate


class BaseEstimator:
    """Shared constructor logic for all estimators."""

    def __init__(
        self,
        max_sample: int = 2500,
        min_sample: int = 750,
        max_depth: int = 3,
        overlap_epsilon: float = 0.1,
        use_gpu: bool = False,
        kernel_type: str = "laplace",
        search_rounds: int = 15,
        bounds: tuple[float, float] = (0.10, 8.5),
        initial_precision: float = 1.0,
        sample_share: float = 1.0,
        precision_method: str = "pilot-cv",
        pilot_factor: float = 2.0,
    ) -> None:
        self.tree_params: dict[str, object] = dict(
            min_sample=min_sample,
            max_sample=max_sample,
            max_depth=max_depth,
            overlap_epsilon=overlap_epsilon,
            use_gpu=use_gpu,
            kernel_type=kernel_type,
            search_rounds=search_rounds,
            bounds=bounds,
            initial_precision=initial_precision,
            sample_share=sample_share,
            precision_method=precision_method,
            pilot_factor=pilot_factor,
        )

    def __repr__(self) -> str:
        name = type(self).__name__
        fitted = (
            (hasattr(self, "tree_") and self.tree_ is not None)
            or (hasattr(self, "variance_tree_") and self.variance_tree_ is not None)
            or (hasattr(self, "covariance_tree_") and self.covariance_tree_ is not None)
        )
        status = "fitted" if fitted else "not fitted"
        return f"{name}({status})"

    def _make_tree(self) -> KernelTree:
        return KernelTree(**self.tree_params)


class MeanEstimator(BaseEstimator):
    """Estimates E[Y|X] using kernel regression (Nadaraya-Watson via KernelTree)."""

    def __init__(self, use_gpu: bool = False, **kwargs) -> None:
        super().__init__(use_gpu=use_gpu, **kwargs)
        self.tree_: KernelTree | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanEstimator":
        X, y = validate(X, y)
        self.tree_ = self._make_tree().fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        (X,) = validate(X)
        return self.tree_.predict(X).ravel()


class VarianceEstimator(BaseEstimator):
    """Estimates Var[Y|X] using the Fan & Yao double kernel approach.

    Pass 1: Fit KernelTree on (X, y) with bandwidth h1 to get residuals.
    Pass 2: Fit KernelTree on (X, residuals^2) with independently optimized
    bandwidth h2 to estimate the conditional variance.
    """

    def __init__(self, use_gpu: bool = False, **kwargs) -> None:
        super().__init__(use_gpu=use_gpu, **kwargs)
        self.mean_estimator_: MeanEstimator | None = None
        self.variance_tree_: KernelTree | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VarianceEstimator":
        X, y = validate(X, y)
        self.X_ = X

        # pass 1: estimate conditional mean
        self.mean_estimator_ = MeanEstimator(**self.tree_params).fit(X, y)
        self.training_mean_ = self.mean_estimator_.predict(X)
        residuals = y - self.training_mean_

        # pass 2: estimate conditional variance from squared residuals
        squared_residuals = residuals**2
        self.variance_tree_ = self._make_tree().fit(X, squared_residuals)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.variance_tree_ is None:
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        (X,) = validate(X)
        variance = self.variance_tree_.predict(X).ravel()
        return np.maximum(variance, 0.0)

    def unconditional(self) -> float:
        """Return unconditional variance (average of conditional variance over training X)."""
        if self.variance_tree_ is None:
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        return float(np.mean(self.predict(self.X_)))


class CovarianceEstimator(BaseEstimator):
    """Estimates Cov[Y, Z|X] using the double kernel approach.

    Fits E[Y|X] and E[Z|X] separately, computes residuals, then fits
    E[residual_y * residual_z | X] with its own bandwidth.
    """

    def __init__(self, use_gpu: bool = False, **kwargs) -> None:
        super().__init__(use_gpu=use_gpu, **kwargs)
        self.mean_y_estimator_: MeanEstimator | None = None
        self.mean_z_estimator_: MeanEstimator | None = None
        self.covariance_tree_: KernelTree | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, z: np.ndarray) -> "CovarianceEstimator":
        X, y, z = validate(X, y, z)

        # fit conditional means
        self.mean_y_estimator_ = MeanEstimator(**self.tree_params).fit(X, y)
        self.mean_z_estimator_ = MeanEstimator(**self.tree_params).fit(X, z)

        # compute residual products
        self.training_mean_y_ = self.mean_y_estimator_.predict(X)
        self.training_mean_z_ = self.mean_z_estimator_.predict(X)
        residuals_y = y - self.training_mean_y_
        residuals_z = z - self.training_mean_z_
        residual_product = residuals_y * residuals_z

        # fit conditional covariance
        self.covariance_tree_ = self._make_tree().fit(X, residual_product)

        # store for fit_correlation()
        self.residuals_y_ = residuals_y
        self.residuals_z_ = residuals_z
        self.X_ = X

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.covariance_tree_ is None:
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        (X,) = validate(X)
        return self.covariance_tree_.predict(X).ravel()

    def unconditional(self) -> float:
        """Return unconditional covariance (average of conditional covariance over training X)."""
        if self.covariance_tree_ is None:
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        return float(np.mean(self.predict(self.X_)))

    def fit_correlation(self) -> "CovarianceEstimator":
        """Fit variance trees for correlation, reusing residuals from fit().

        Must be called after fit(). Fits E[residual_y²|X] and E[residual_z²|X]
        with independently optimized bandwidths.

        Returns self.
        """
        if self.covariance_tree_ is None:
            raise RuntimeError("Call fit() before fit_correlation().")
        self.variance_y_tree_ = self._make_tree().fit(self.X_, self.residuals_y_**2)
        self.variance_z_tree_ = self._make_tree().fit(self.X_, self.residuals_z_**2)
        return self

    def predict_correlation(self, X: np.ndarray) -> np.ndarray:
        """Predict Corr[Y, Z|X] = Cov / sqrt(Var_y * Var_z)."""
        if not hasattr(self, "variance_y_tree_") or self.variance_y_tree_ is None:
            raise RuntimeError("Call fit_correlation() before predict_correlation().")
        (X,) = validate(X)
        cov = self.covariance_tree_.predict(X).ravel()
        var_y = np.maximum(self.variance_y_tree_.predict(X).ravel(), 0.0)
        var_z = np.maximum(self.variance_z_tree_.predict(X).ravel(), 0.0)
        denom = np.maximum(np.sqrt(var_y * var_z), 1e-8)
        return np.clip(cov / denom, -1.0, 1.0)

    def unconditional_correlation(self) -> float:
        """Return unconditional correlation (average of conditional correlation over training X)."""
        if not hasattr(self, "variance_y_tree_") or self.variance_y_tree_ is None:
            raise RuntimeError(
                "Call fit_correlation() before unconditional_correlation()."
            )
        return float(np.mean(self.predict_correlation(self.X_)))


def cross_fit_residualize(
    W: np.ndarray,
    *targets: np.ndarray,
    tree_params: dict[str, object] | None = None,
    n_folds: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, ...]:
    """Cross-fitted residualization (sample splitting, cf. Chernozhukov et al. 2018).

    Splits data into n_folds. For each fold, fits E[target|W] on the
    other folds and predicts on the held-out fold. Returns out-of-fold
    residuals for all targets. This avoids overfitting bias in the
    residualization step.

    Args:
        W: Confounders.
        *targets: Variables to residualize (1D arrays).
        tree_params: Dict of KernelTree params (default: BaseEstimator defaults).
        n_folds: Number of cross-fitting folds (default 3).
        seed: RNG seed for fold assignment.

    Returns:
        Tuple of residualized arrays, same order as targets.
    """
    W, *targets = validate(W, *targets)
    n = W.shape[0]
    if tree_params is None:
        tree_params = {}

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, n_folds)

    residuals = [np.empty_like(t) for t in targets]

    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        W_train, W_test = W[train_idx], W[test_idx]

        for i, target in enumerate(targets):
            y_train = target[train_idx]
            est = MeanEstimator(**tree_params).fit(W_train, y_train)
            residuals[i][test_idx] = target[test_idx] - est.predict(W_test)

    return tuple(residuals)
