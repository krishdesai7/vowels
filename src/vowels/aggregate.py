import numpy as np


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def steady_state_index(
    f1: np.ndarray,
    f2: np.ndarray,
    rel_time: np.ndarray,
    lo: float,
    hi: float,
) -> int:
    f1z = _zscore(f1)
    f2z = _zscore(f2)
    velocity = np.full(f1.shape, np.inf)
    velocity[1:] = np.hypot(np.diff(f1z), np.diff(f2z))

    in_window = (rel_time >= lo) & (rel_time <= hi)
    if not in_window.any():
        in_window = np.ones(f1.shape, dtype=bool)

    masked = np.where(in_window, velocity, np.inf)
    return int(np.argmin(masked))
