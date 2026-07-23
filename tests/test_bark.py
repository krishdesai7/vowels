import polars as pl
import pytest

from vowels.bark import add_bark_dims


def _z(f: float) -> float:
    return (26.81 * f) / (1960 + f) - 0.53


def test_add_bark_dims_default_columns() -> None:
    df = pl.DataFrame({"F0": [120.0], "F1": [500.0], "F2": [1500.0], "F3": [2500.0]})
    out = add_bark_dims(df)
    z0, z1, z2, z3 = _z(120.0), _z(500.0), _z(1500.0), _z(2500.0)
    assert out["Openness"][0] == pytest.approx(z1 - z0)
    assert out["Frontness"][0] == pytest.approx(z2 - z1)
    assert out["Roundness"][0] == pytest.approx(z3 - z2)


def test_add_bark_dims_accepts_smoothed_columns() -> None:
    df = pl.DataFrame(
        {"F0": [120.0], "F1_s": [500.0], "F2_s": [1500.0], "F3_s": [2500.0]}
    )
    out = add_bark_dims(df, formant_cols=("F1_s", "F2_s", "F3_s"))
    assert out["Frontness"][0] == pytest.approx(_z(1500.0) - _z(500.0))
