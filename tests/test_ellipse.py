import numpy as np
import pytest
from numpy.typing import NDArray

from vowels import precompute_ellipse


def test_returns_none_for_fewer_than_three_points() -> None:
    f2: NDArray[np.double] = np.array([1000.0, 1100.0])
    f1: NDArray[np.double] = np.array([400.0, 420.0])
    assert precompute_ellipse(f2, f1) is None


def test_returns_none_for_single_point() -> None:
    assert precompute_ellipse(np.array([1000.0]), np.array([400.0])) is None


def test_returns_polygon_with_correct_length() -> None:
    rng: np.random.Generator = np.random.default_rng(42)
    f2: NDArray[np.double] = rng.normal(1500.0, 100.0, 20)
    f1: NDArray[np.double] = rng.normal(500.0, 50.0, 20)
    pts: list[dict[str, float]] | None = precompute_ellipse(f2, f1, n_points=60)
    assert pts is not None
    assert len(pts) == 60


def test_polygon_centroid_near_mean() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    f2: NDArray[np.double] = rng.normal(1500.0, 80.0, 30)
    f1: NDArray[np.double] = rng.normal(450.0, 40.0, 30)
    pts: list[dict[str, float]] | None = precompute_ellipse(f2, f1)
    assert pts is not None
    centroid_f2: np.floating = np.mean([p["F2"] for p in pts])
    centroid_f1: np.floating = np.mean([p["F1"] for p in pts])
    assert centroid_f2 == pytest.approx(np.mean(f2), abs=5.0)
    assert centroid_f1 == pytest.approx(np.mean(f1), abs=5.0)


def test_collinear_points_handled() -> None:
    # Degenerate case: all points on a line → one eigenvalue ≈ 0
    f2: NDArray[np.double] = np.linspace(1000.0, 2000.0, 10)
    f1: NDArray[np.double] = f2 * 0.3  # perfectly correlated
    pts: list[dict[str, float]] | None = precompute_ellipse(f2, f1)
    assert pts is not None  # should not raise
