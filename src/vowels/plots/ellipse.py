import numpy as np
from numpy.typing import NDArray


def precompute_ellipse(
    f2: NDArray,
    f1: NDArray,
    n_std: float = 2.0,
    n_points: int = 100,
) -> list[dict[str, float]] | None:
    if len(f2) < 3:
        return None

    cov = np.cov(f2, f1)
    mean_f2, mean_f1 = float(np.mean(f2)), float(np.mean(f1))

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = float(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    half_w = n_std * float(np.sqrt(max(eigenvalues[0], 0.0)))
    half_h = n_std * float(np.sqrt(max(eigenvalues[1], 0.0)))

    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    pts_f2 = mean_f2 + half_w * np.cos(t) * cos_a - half_h * np.sin(t) * sin_a
    pts_f1 = mean_f1 + half_w * np.cos(t) * sin_a + half_h * np.sin(t) * cos_a

    return [{"F2": float(x), "F1": float(y)} for x, y in zip(pts_f2, pts_f1)]
