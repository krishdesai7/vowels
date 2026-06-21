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

    cov: NDArray[np.double] = np.cov(f2, f1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order: NDArray[np.intp] = eigenvalues.argsort()[::-1]
    eigenvalues: NDArray[np.double] = eigenvalues[order]
    eigenvectors: NDArray[np.double] = eigenvectors[:, order]

    angle: np.double = np.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    half_w: np.double = n_std * np.sqrt(max(eigenvalues[0], 0.0))
    half_h: np.double = n_std * np.sqrt(max(eigenvalues[1], 0.0))

    t: NDArray[np.double] = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cos_a: np.double = np.cos(angle)
    sin_a: np.double = np.sin(angle)
    pts_f2: NDArray[np.double] = (
        f2.mean() + half_w * np.cos(t) * cos_a - half_h * np.sin(t) * sin_a
    )
    pts_f1: NDArray[np.double] = (
        f1.mean() + half_w * np.cos(t) * sin_a + half_h * np.sin(t) * cos_a
    )

    return [{"F2": x, "F1": y} for x, y in zip(pts_f2, pts_f1, strict=True)]
