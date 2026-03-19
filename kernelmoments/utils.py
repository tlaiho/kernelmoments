import numpy as np


def validate(X: np.ndarray, *ys: np.ndarray) -> tuple[np.ndarray, ...]:
    """Validate X (2D) and one or more y arrays (1D), check for NaN and matching sample counts."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1D or 2D, got {X.ndim}D")
    if np.isnan(X).any():
        raise ValueError("X contains NaN values")
    validated: list[np.ndarray] = []
    for y in ys:
        y = np.asarray(y, dtype=np.float32).ravel()
        if np.isnan(y).any():
            raise ValueError("y contains NaN values")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}"
            )
        validated.append(y)
    return (X, *validated)
