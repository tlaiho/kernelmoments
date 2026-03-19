"""Benchmark wall-clock fitting times for all estimators across sample sizes."""

import statistics
import time
import numpy as np
from kernelmoments import (
    CovarianceEstimator,
    MeanEstimator,
    VarianceEstimator,
)

CPU_SAMPLE_SIZES = [1_000, 5_000, 10_000]
GPU_SAMPLE_SIZES = [1_000, 5_000, 10_000, 25_000, 50_000]
N_RUNS = 3

ESTIMATORS = [
    # (name, class, needs_z, correlation)
    ("MeanEstimator", MeanEstimator, False, False),
    ("VarianceEstimator", VarianceEstimator, False, False),
    ("CovarianceEstimator", CovarianceEstimator, True, False),
    ("Covariance+Correlation", CovarianceEstimator, True, True),
]


def check_gpu():
    """Return True if CuPy is available and at least one GPU is detected."""
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def generate_data(n, seed=42):
    """Generate synthetic benchmark data: X ~ U(0,10), y = sin(X) + noise, z = cos(X) + noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=n)
    y = np.sin(X) + 0.3 * rng.standard_normal(n)
    z = np.cos(X) + 0.3 * rng.standard_normal(n)
    return X, y, z


def time_fit(estimator_cls, use_gpu, X, y, z=None, correlation=False):
    """Create a fresh estimator and return wall-clock time (seconds) for fit()."""
    est = estimator_cls(use_gpu=use_gpu)
    if z is not None:
        start = time.perf_counter()
        est.fit(X, y, z)
        if correlation:
            est.fit_correlation()
        return time.perf_counter() - start
    else:
        start = time.perf_counter()
        est.fit(X, y)
        return time.perf_counter() - start


def print_results(results, has_gpu):
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if has_gpu:
        header = f"{'Estimator':<24} {'N':>8}  {'CPU (s)':>9}  {'GPU (s)':>9}  {'Speedup':>8}"
    else:
        header = f"{'Estimator':<24} {'N':>8}  {'CPU (s)':>9}  {'GPU (s)':>9}"
    print(header)
    print("-" * len(header))

    for name, n, cpu_time, gpu_time in results:
        cpu_str = f"{cpu_time:.4f}" if cpu_time is not None else "---"
        if gpu_time is not None:
            gpu_str = f"{gpu_time:.4f}"
            if cpu_time is not None:
                speedup = f"{cpu_time / gpu_time:.1f}x" if gpu_time > 0 else "---"
            else:
                speedup = "---"
        else:
            gpu_str = "---"
            speedup = "---"
        if has_gpu:
            print(f"{name:<24} {n:>8,}  {cpu_str:>9}  {gpu_str:>9}  {speedup:>8}")
        else:
            print(f"{name:<24} {n:>8,}  {cpu_str:>9}  {gpu_str:>9}")

    print("=" * 72)


if __name__ == "__main__":
    has_gpu = check_gpu()
    all_sizes = sorted(set(CPU_SAMPLE_SIZES) | (set(GPU_SAMPLE_SIZES) if has_gpu else set()))
    print(f"GPU available: {has_gpu}")
    print(f"CPU sample sizes: {CPU_SAMPLE_SIZES}")
    if has_gpu:
        print(f"GPU sample sizes: {GPU_SAMPLE_SIZES}")
    print(f"Runs per config: {N_RUNS} (median reported)")
    print()

    results = []

    for n in all_sizes:
        run_cpu = n in CPU_SAMPLE_SIZES
        run_gpu = has_gpu and n in GPU_SAMPLE_SIZES
        X, y, z = generate_data(n)
        print(f"--- N = {n:,} ---")

        for name, cls, needs_z, correlation in ESTIMATORS:
            z_arg = z if needs_z else None

            # CPU runs
            if run_cpu:
                cpu_times = [time_fit(cls, False, X, y, z_arg, correlation) for _ in range(N_RUNS)]
                cpu_median = statistics.median(cpu_times)
            else:
                cpu_median = None

            # GPU runs
            if run_gpu:
                gpu_times = [time_fit(cls, True, X, y, z_arg, correlation) for _ in range(N_RUNS)]
                gpu_median = statistics.median(gpu_times)
            else:
                gpu_median = None

            # progress output
            parts = []
            if cpu_median is not None:
                parts.append(f"CPU {cpu_median:.4f}s")
            if gpu_median is not None:
                parts.append(f"GPU {gpu_median:.4f}s")
            if cpu_median is not None and gpu_median is not None and gpu_median > 0:
                parts.append(f"speedup {cpu_median / gpu_median:.1f}x")
            print(f"  {name:<24} {'  '.join(parts)}")

            results.append((name, n, cpu_median, gpu_median))

    print_results(results, has_gpu)
