from typing import Final

import polars as pl
import pytest

from vowels import parse_labels

REQUIRED_COLUMNS: Final[set[str]] = {"time", "label", "F1", "F2", "F3", "set", "word"}


def _make_df(labels: list[str]) -> pl.LazyFrame:
    n: int = len(labels)
    return parse_labels(
        pl.LazyFrame(
            {
                "time": [float(i) for i in range(n)],
                "label": labels,
                "F1": [500.0] * n,
                "F2": [1500.0] * n,
                "F3": [2500.0] * n,
            }
        )
    )


def test_output_has_required_columns() -> None:
    df: pl.LazyFrame = _make_df(["FLEECE_beat", "TRAP_bad"])
    assert REQUIRED_COLUMNS.issubset(set(df.columns))


def test_no_null_set_or_word_for_standard_labels() -> None:
    labels: list[str] = [
        "FLEECE_beat",
        "TRAP_bad",
        "GOOSE_food",
        "2coMMA_sofa",
        "2haPPY_city",
        "PRICE_try:1",
    ]
    df: pl.LazyFrame = _make_df(labels)
    assert (
        df.select(pl.col("set"))
        .unique()
        .collect()
        .get_column("set")
        .to_list()
        .count(None)
        == 0
    )
    assert (
        df.select(pl.col("word"))
        .unique()
        .collect()
        .get_column("word")
        .to_list()
        .count(None)
        == 0
    )


def test_time_and_formant_columns_preserved() -> None:
    df: pl.LazyFrame = _make_df(["KIT_sit"])
    assert df.select(pl.col("time")).unique().collect().get_column("time").to_list()[
        0
    ] == pytest.approx(0.0)
    assert df.select(pl.col("F1")).unique().collect().get_column("F1").to_list()[
        0
    ] == pytest.approx(500.0)
    assert df.select(pl.col("F2")).unique().collect().get_column("F2").to_list()[
        0
    ] == pytest.approx(1500.0)
    assert df.select(pl.col("F3")).unique().collect().get_column("F3").to_list()[
        0
    ] == pytest.approx(2500.0)
