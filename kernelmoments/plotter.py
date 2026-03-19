""" Classes and functions for quick plotting."""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any
from .estimators import (
    MeanEstimator,
    VarianceEstimator,
    CovarianceEstimator,
    BaseEstimator,
    cross_fit_residualize,
)
from .utils import validate


@dataclass
class PlotStyle:
    """Visual styling for kernel moment plots."""

    # scatter
    scatter_color: str = "C0"
    scatter_alpha: float = 1.0
    scatter_size: float = 10
    scatter_max_points: int | None = 500

    # lines
    line_color: str = "C1"
    line_width: float = 2
    band_line_width: float = 1
    band_linestyle: str = "--"
    zero_line_color: str = "grey"
    zero_line_width: float = 0.5
    zero_linestyle: str = "--"

    # axes
    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | None = None
    x_quantile_trim: float = 0.01


@dataclass
class PlotResult:
    """Holds plot result and underlying data."""

    x_values: np.ndarray
    y_values: np.ndarray
    estimator: object
    fig: plt.Figure
    ax: plt.Axes


class Plotter:
    """Estimates and plots conditional relationships between variables.

    Args:
        df: pandas or polars DataFrame. Required for fit() and plot_relationship().
        x_scaler: Sklearn-compatible scaler applied to the conditioning 
        variable before fitting.
        n_sample: If set, randomly subsample data to this many observations
            before fitting. Speeds up estimation with little quality loss.
        seed: RNG seed for subsampling.
        **tree_params: Forwarded to KernelTree (use_gpu, kernel_type, etc.).
    """

    def __init__(
        self,
        df: Any,
        use_gpu: bool = False,
        x_scaler: Any = None,
        n_sample: int | None = None,
        seed: int = 42,
        style: PlotStyle | None = None,
        **tree_params,
    ) -> None:
        self.df = df
        self.tree_params: dict[str, object] = {"use_gpu": use_gpu, **tree_params}
        self.x_scaler = x_scaler
        self.n_sample = n_sample
        self.seed = seed
        self.style = style if style is not None else PlotStyle()

        self.estimators: dict[tuple, BaseEstimator] = {}

    def fit(
        self,
        x: str,
        y: str,
        z: str | None = None,
        partial_out: str | list[str] | None = None,
    ) -> "Plotter":
        """Pre-fit estimators for the relationship between y, x and possibly z.
        Args:
            x: Column name for conditioning variable.
            y: Column name for dependent variable.
            z: Column name for second dependent variable (covariance).
            partial_out: Column names to partial out.

        Returns:
            self
        """
        x_arr, y_arr, z_arr = self._prepare_arrays(x, y, z, partial_out)

        po_key = self._partial_out_key(partial_out)

        self.estimators[("mean", x, y, po_key)] = self._fit_moment("mean", x_arr, y_arr)
        self.estimators[("variance", x, y, po_key)] = self._fit_moment(
            "variance", x_arr, y_arr
        )
        if z is not None:
            self.estimators[("covariance", x, y, z, po_key)] = self._fit_moment(
                "covariance", x_arr, y_arr, z_arr
            )
            self.estimators[("correlation", x, y, z, po_key)] = self._fit_moment(
                "correlation", x_arr, y_arr, z_arr
            )

        return self

    @staticmethod
    def _partial_out_key(partial_out: str | list[str] | None) -> tuple[str, ...] | None:
        """Normalize partial_out into a cache key."""
        if partial_out is None:
            return None
        if isinstance(partial_out, str):
            return (partial_out,)
        return tuple(partial_out)

    def _prepare_arrays(
        self,
        x: str,
        y: str,
        z: str | None = None,
        partial_out: str | list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Resolve columns, residualize confounders, and subsample.

        Returns:
            (x_arr, y_arr, z_arr) — z_arr is None when z is not given.
        """
        cols = self._resolve(x=x, y=y, z=z, w=partial_out)
        x_arr, y_arr = cols["x"].ravel(), cols["y"].ravel()
        z_arr = cols.get("z")
        w_arr = cols.get("w")

        if self.x_scaler is not None:
            x_arr = self.x_scaler.fit_transform(x_arr.reshape(-1, 1)).ravel()

        has_controls = w_arr is not None
        has_instrument = z_arr is not None

        if self.n_sample is not None and self.n_sample < x_arr.shape[0]:
            idx = np.random.default_rng(self.seed).choice(
                x_arr.shape[0], self.n_sample, replace=False
            )
            x_arr, y_arr = x_arr[idx], y_arr[idx]
            if has_instrument:
                z_arr = z_arr[idx]
            if has_controls:
                w_arr = w_arr[idx]

        if has_controls and has_instrument:
            y_arr, x_arr, z_arr = cross_fit_residualize(
                w_arr, y_arr, x_arr, z_arr, tree_params=self.tree_params
            )
        elif has_controls:
            y_arr, x_arr = cross_fit_residualize(
                w_arr, y_arr, x_arr, tree_params=self.tree_params
            )

        return x_arr.reshape(-1, 1), y_arr, z_arr

    def _resolve(self, **names: Any) -> dict[str, np.ndarray]:
        """Extract named columns from self.df as numpy arrays."""
        result: dict[str, np.ndarray] = {}
        for key, value in names.items():
            if value is None:
                continue
            if isinstance(value, str):
                result[key] = self._df_to_numpy(value)
            elif isinstance(value, (list, tuple)):
                result[key] = self._df_to_numpy(list(value))
            else:
                result[key] = value
        return result

    def _df_to_numpy(self, names: str | list[str]) -> np.ndarray:
        """Extract column(s) from self.df as float32 numpy array."""
        dtype = type(self.df).__module__.split(".")[0]
        if dtype == "pandas":
            return self.df[names].to_numpy(dtype=np.float32)
        if dtype == "polars":
            return self.df[names].to_numpy().astype(np.float32)
        raise TypeError(
            f"Unsupported DataFrame type: {type(self.df).__name__}. "
            "Expected pandas or polars DataFrame."
        )

    def _get_estimator(
        self,
        key: tuple,
        moment: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None = None,
    ) -> BaseEstimator:
        """Return cached estimator or fit and cache a new one."""
        if key not in self.estimators:
            self.estimators[key] = self._fit_moment(moment, x, y, z)
        return self.estimators[key]

    def _fit_moment(
        self,
        moment: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None = None,
    ) -> BaseEstimator:
        """Fit and return the estimator for a given moment."""
        if moment == "mean":
            return MeanEstimator(**self.tree_params).fit(x, y)
        elif moment == "variance":
            return VarianceEstimator(**self.tree_params).fit(x, y)
        elif moment == "covariance":
            if z is None:
                raise ValueError("moment='covariance' requires z")
            return CovarianceEstimator(**self.tree_params).fit(x, y, z)
        elif moment == "correlation":
            if z is None:
                raise ValueError("moment='correlation' requires z")
            return (
                CovarianceEstimator(**self.tree_params)
                .fit(x, y, z)
                .fit_correlation()
            )
        else:
            raise ValueError(f"Unknown moment: {moment!r}")

    def plot(
        self,
        x: str,
        y: str,
        z: str | None = None,
        moment: str = "mean",
        partial_out: str | list[str] | None = None,
        bands: bool = False,
        ax: plt.Axes | None = None,
        n_grid: int = 200,
        style: PlotStyle | None = None,
    ) -> PlotResult:
        """Plot the conditional moment of y given x.

        Args:
            x: Column name for conditioning variable.
            y: Column name for dependent variable.
            moment: 'mean', 'variance', or 'covariance'.
            z: Column name for second dependent variable (covariance).
            partial_out: Column names to partial out.
            bands: Show ±1.96·√Var prediction bands for moment='mean'.
            ax: Optional matplotlib Axes.
            n_grid: Number of points for the smooth curve.
            style: Plot styling overrides (defaults to Plotter's style).

        Returns:
            PlotResult with figure, axes, grid, predicted values, and estimator.
        """
        style = style or self.style

        _validate_moment(moment, z)

        po_key = self._partial_out_key(partial_out)
        if z:
            cache_key = (moment, x, y, z, po_key)
        else:
            cache_key = (moment, x, y, po_key)

        # display labels: append "| controls" when partialling out
        if partial_out is not None:
            controls_label = ", ".join(po_key)
            x_label = f"{x} | {controls_label}"
            y_label = f"{y} | {controls_label}"
        else:
            x_label, y_label = x, y
            controls_label = None

        x_arr, y_arr, z_arr = self._prepare_arrays(x, y, z, partial_out)

        estimator = self._get_estimator(cache_key, moment, x_arr, y_arr, z_arr)

        # build grid in scaled space for prediction, original space for display
        x_grid_scaled = np.linspace(
            np.quantile(x_arr, style.x_quantile_trim),
            np.quantile(x_arr, 1 - style.x_quantile_trim),
            n_grid,
            dtype=np.float32
            )
        
        if self.x_scaler is not None:
            x_grid = self.x_scaler.inverse_transform(
                x_grid_scaled.reshape(-1, 1)
            ).ravel()
        else:
            x_grid = x_grid_scaled

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if moment == "correlation":
            y_values = estimator.predict_correlation(x_grid_scaled.reshape(-1, 1))
        else:
            y_values = estimator.predict(x_grid_scaled.reshape(-1, 1))

        # use original-scale x for scatter and axis display
        if self.x_scaler is not None:
            x_arr_display = self.x_scaler.inverse_transform(x_arr).ravel()
        else:
            x_arr_display = x_arr.ravel()

        if moment == "mean":
            y_var = None
            if bands:
                var_est = self._get_estimator(
                    ("variance", x, y, po_key), "variance", x_arr, y_arr
                )
                y_var = var_est.predict(x_grid_scaled.reshape(-1, 1))
            _plot_mean(
                ax, x_arr_display, y_arr, x_grid, y_values, style, y_var, y_name=y_label
            )
        elif moment == "variance":
            _plot_variance(ax, x_grid, y_values, style, x_name=x, y_name=y, controls=controls_label)
        elif moment == "covariance":
            _plot_covariance(ax, x_grid, y_values, style, x_name=x, y_name=y, z_name=z, controls=controls_label)
        elif moment == "correlation":
            _plot_correlation(ax, x_grid, y_values, style, x_name=x, y_name=y, z_name=z, controls=controls_label)

        ax.set_xlabel(x_label)
        if style.x_lim is not None:
            ax.set_xlim(*style.x_lim)
        if style.y_lim is not None:
            ax.set_ylim(*style.y_lim)
        return PlotResult(
            x_values=x_grid,
            y_values=y_values,
            estimator=estimator,
            fig=fig,
            ax=ax,
        )


def _validate_moment(moment: str, z: np.ndarray | None = None) -> None:
    valid = ("mean", "variance", "covariance", "correlation")
    if moment not in valid:
        raise ValueError(f"Unknown moment: {moment!r}")
    if moment in ("covariance", "correlation") and z is None:
        raise ValueError(f"moment={moment!r} requires z")


def _plot_mean(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    style: PlotStyle,
    y_variance: np.ndarray = None,
    y_name: str = "Y",
) -> None:
    x_scatter, y_scatter = _downsample_scatter(x, y, style)
    ax.scatter(
        x_scatter,
        y_scatter,
        alpha=style.scatter_alpha,
        s=style.scatter_size,
        color=style.scatter_color,
        edgecolors="none",
    )
    ax.plot(x_grid, y_mean, color=style.line_color, linewidth=style.line_width)
    if y_variance is not None:
        std = np.sqrt(np.maximum(y_variance, 0))
        ax.plot(
            x_grid,
            y_mean - 1.96 * std,
            color=style.line_color,
            linewidth=style.band_line_width,
            linestyle=style.band_linestyle,
        )
        ax.plot(
            x_grid,
            y_mean + 1.96 * std,
            color=style.line_color,
            linewidth=style.band_line_width,
            linestyle=style.band_linestyle,
        )
    ax.set_ylabel(y_name)


def _plot_variance(
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_variance: np.ndarray,
    style: PlotStyle,
    x_name: str = "X",
    y_name: str = "Y",
    controls: str | None = None,
) -> None:
    ax.plot(x_grid, y_variance, color=style.line_color, linewidth=style.line_width)
    cond = f"{x_name}, {controls}" if controls else x_name
    ax.set_ylabel(f"Var[{y_name}|{cond}]")


def _plot_covariance(
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_covariance: np.ndarray,
    style: PlotStyle,
    x_name: str = "X",
    y_name: str = "Y",
    z_name: str = "Z",
    controls: str | None = None,
) -> None:
    ax.plot(x_grid, y_covariance, color=style.line_color, linewidth=style.line_width)
    ax.axhline(
        0,
        color=style.zero_line_color,
        linewidth=style.zero_line_width,
        linestyle=style.zero_linestyle,
    )
    cond = f"{x_name}, {controls}" if controls else x_name
    ax.set_ylabel(f"Cov[{y_name}, {z_name}|{cond}]")


def _plot_correlation(
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_correlation: np.ndarray,
    style: PlotStyle,
    x_name: str = "X",
    y_name: str = "Y",
    z_name: str = "Z",
    controls: str | None = None,
) -> None:
    ax.plot(x_grid, y_correlation, color=style.line_color, linewidth=style.line_width)
    ax.axhline(
        0,
        color=style.zero_line_color,
        linewidth=style.zero_line_width,
        linestyle=style.zero_linestyle,
    )
    ax.set_ylim(-1.1, 1.1)
    cond = f"{x_name}, {controls}" if controls else x_name
    ax.set_ylabel(f"Corr[{y_name}, {z_name}|{cond}]")


def _downsample_scatter(
    x: np.ndarray,
    y: np.ndarray,
    style: PlotStyle,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin scatter points for display only (estimation is unaffected)."""
    if style.scatter_max_points is not None and len(x) > style.scatter_max_points:
        idx = np.random.default_rng(0).choice(
            len(x), style.scatter_max_points, replace=False
        )
        return x[idx], y[idx]
    return x, y


def plot_relationship(
    x: np.ndarray,
    y: np.ndarray,
    moment: str = "mean",
    z: np.ndarray | None = None,
    x_plot: np.ndarray | None = None,
    partial_out: np.ndarray | None = None,
    bands: bool = True,
    ax: plt.Axes | None = None,
    n_grid: int = 200,
    style: PlotStyle | None = None,
    x_name: str = "X",
    y_name: str = "Y",
    z_name: str = "Z",
    **tree_params,
) -> PlotResult:
    """Plot the conditional moment of y given x.

    Args:
        x: Conditioning variable (1D array).
        y: Dependent variable (1D array).
        moment: 'mean', 'variance', or 'covariance'.
        z: Second dependent variable for covariance (1D array).
        x_plot: Original-scale x values for display (1D array) so 
        the plot axes show the original scale.
        partial_out: Variables to partial out.
        bands: Show ±1.96·√Var prediction bands for moment='mean'.
        ax: Optional matplotlib Axes.
        n_grid: Number of points for the smooth curve.
        style: Plot styling (defaults to PlotStyle()).
        x_name: Label for the x-axis.
        y_name: Label for the y variable.
        z_name: Label for the z variable (covariance/correlation).
        **tree_params: Forwarded to KernelTree (use_gpu, kernel_type, etc.).

    Returns:
        PlotResult with figure, axes, grid, predicted values, and estimator.
    """
    if style is None:
        style = PlotStyle()

    x = np.asarray(x, dtype=np.float32).ravel()
    if x_plot is None:
        x_plot = x
    else:
        x_plot = np.asarray(x_plot, dtype=np.float32).ravel()
    if z is not None:
        _, y, z = validate(x.reshape(-1, 1), y, z)
    else:
        _, y = validate(x.reshape(-1, 1), y)

    if partial_out is not None:
        (W,) = validate(partial_out)
        if z is not None:
            y, x, z = cross_fit_residualize(W, y, x, z, tree_params=tree_params)
        else:
            y, x = cross_fit_residualize(W, y, x, tree_params=tree_params)

    x_2d = x.reshape(-1, 1)

    _validate_moment(moment, z)

    if moment == "mean":
        estimator = MeanEstimator(**tree_params).fit(x_2d, y)
    elif moment == "variance":
        estimator = VarianceEstimator(**tree_params).fit(x_2d, y)
    elif moment == "covariance":
        estimator = CovarianceEstimator(**tree_params).fit(x_2d, y, z)
    elif moment == "correlation":
        estimator = CovarianceEstimator(**tree_params).fit(x_2d, y, z).fit_correlation()

    x_pred = np.linspace(
        np.quantile(x, style.x_quantile_trim),
        np.quantile(x, 1 - style.x_quantile_trim),
        n_grid,
        dtype=np.float32,
    )
    x_grid = np.linspace(
        np.quantile(x_plot, style.x_quantile_trim),
        np.quantile(x_plot, 1 - style.x_quantile_trim),
        n_grid,
        dtype=np.float32,
    )
    if moment == "correlation":
        y_values = estimator.predict_correlation(x_pred.reshape(-1, 1))
    else:
        y_values = estimator.predict(x_pred.reshape(-1, 1))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if moment == "mean":
        y_var = None
        if bands:
            var_est = VarianceEstimator(**tree_params).fit(x_2d, y)
            y_var = var_est.predict(x_pred.reshape(-1, 1))
        _plot_mean(ax, x_plot, y, x_grid, y_values, style, y_var, y_name=y_name)
    elif moment == "variance":
        _plot_variance(ax, x_grid, y_values, style, x_name=x_name, y_name=y_name)
    elif moment == "covariance":
        _plot_covariance(ax, x_grid, y_values, style, x_name=x_name, y_name=y_name, z_name=z_name)
    elif moment == "correlation":
        _plot_correlation(ax, x_grid, y_values, style, x_name=x_name, y_name=y_name, z_name=z_name)

    ax.set_xlabel(x_name)
    if style.x_lim is not None:
        ax.set_xlim(*style.x_lim)
    if style.y_lim is not None:
        ax.set_ylim(*style.y_lim)

    return PlotResult(
        x_values=x_grid,
        y_values=y_values,
        estimator=estimator,
        fig=fig,
        ax=ax,
    )
