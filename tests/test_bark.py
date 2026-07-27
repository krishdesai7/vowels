import polars as pl
import pytest

from vowels.bark import add_bark_dims


def _z(f: float) -> float:
    return (26.81 * f) / (1960 + f) - 0.53


def test_add_bark_dims_default_columns() -> None:
    lf = pl.LazyFrame({"F0": [120.0], "F1": [500.0], "F2": [1500.0], "F3": [2500.0]})

    z0, z1, z2, z3 = map(_z, (120.0, 500.0, 1500.0, 2500.0))

    actual: tuple[float, float, float] = (
        add_bark_dims(lf).select("Openness", "Frontness", "Roundness").collect().row(0)
    )

    assert actual == pytest.approx((z1 - z0, z2 - z1, z3 - z2))


def test_add_bark_dims_accepts_smoothed_columns() -> None:
    lf = pl.LazyFrame(
        {"F0": [120.0], "F1_s": [500.0], "F2_s": [1500.0], "F3_s": [2500.0]}
    )
    actual: float = (
        add_bark_dims(lf, formant_cols=("F1_s", "F2_s", "F3_s"))
        .select("Frontness")
        .collect()
        .row(0)[0]
    )
    assert actual == pytest.approx(_z(1500.0) - _z(500.0))
