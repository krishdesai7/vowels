from typing import Final

import numpy as np
import polars as pl
import pytest

from vowels import parse_labels
from vowels.pipeline.formants import winner_to_rows

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
    assert REQUIRED_COLUMNS.issubset(set(df.collect_schema().names()))


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


def test_winner_to_rows_schema_and_rel_time() -> None:
    winner = pl.DataFrame({
        "time": [1.0, 1.1, 1.2],
        "F1": [500.0, 510.0, 505.0], "F2": [1500.0, 1490.0, 1495.0],
        "F3": [2500.0, 2510.0, 2505.0],
        "F1_s": [502.0, 508.0, 506.0], "F2_s": [1498.0, 1492.0, 1496.0],
        "F3_s": [2502.0, 2508.0, 2506.0],
        "B1": [50.0, 51.0, 52.0], "B2": [80.0, 81.0, 82.0], "B3": [120.0, 121.0, 122.0],
        "max_formant": [5000.0] * 3, "error": [0.1] * 3,
    })
    f0 = np.array([120.0, 121.0, 119.0])
    rows = winner_to_rows(winner, f0, token_id=7, label="2haPPY_coffee", t1=1.0, t2=1.2)
    assert len(rows) == 3
    EXPECTED_KEYS = {
        "token_id", "label", "set", "word", "is_diphthong", "is_disyllabic",
        "time", "rel_time", "F0", "F1", "F2", "F3", "F1_s", "F2_s", "F3_s",
        "B1", "B2", "B3", "max_formant", "error",
    }
    assert set(rows[0].keys()) == EXPECTED_KEYS
    assert rows[0]["token_id"] == 7
    assert rows[0]["set"] == "haPPY"
    assert rows[0]["word"] == "coffee"
    assert rows[0]["is_disyllabic"] is True
    assert rows[0]["is_diphthong"] is False
    assert rows[0]["rel_time"] == pytest.approx(0.0)
    assert rows[2]["rel_time"] == pytest.approx(1.0)
    assert rows[0]["F0"] == pytest.approx(120.0)
    assert rows[1]["F0"] == pytest.approx(121.0)
    assert rows[2]["F0"] == pytest.approx(119.0)


def test_winner_to_rows_zero_span_guard() -> None:
    winner = pl.DataFrame({
        "time": [1.0], "F1": [500.0], "F2": [1500.0], "F3": [2500.0],
        "F1_s": [500.0], "F2_s": [1500.0], "F3_s": [2500.0],
        "B1": [50.0], "B2": [80.0], "B3": [120.0],
        "max_formant": [5000.0], "error": [0.1],
    })
    rows = winner_to_rows(
        winner, np.array([120.0]), token_id=0, label="TRAP_cat", t1=1.0, t2=1.0
    )
    assert rows[0]["rel_time"] == pytest.approx(0.0)
