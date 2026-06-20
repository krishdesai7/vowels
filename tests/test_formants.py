import polars as pl
import pytest

from vowels.pipeline.formants import parse_labels

REQUIRED_COLUMNS = {"time", "label", "F1", "F2", "F3", "set", "word"}


def _make_df(labels: list[str]) -> pl.DataFrame:
    n = len(labels)
    return parse_labels(
        pl.DataFrame(
            {
                "time": [float(i) for i in range(n)],
                "label": labels,
                "F1": [500.0] * n,
                "F2": [1500.0] * n,
                "F3": [2500.0] * n,
            }
        )
    )


def test_output_has_required_columns():
    df = _make_df(["FLEECE_beat", "TRAP_bad"])
    assert REQUIRED_COLUMNS.issubset(set(df.columns))


def test_no_null_set_or_word_for_standard_labels():
    labels = [
        "FLEECE_beat",
        "TRAP_bad",
        "GOOSE_food",
        "2COMMA_sofa",
        "2HAPPY_city",
        "PRICE_try:1",
    ]
    df = _make_df(labels)
    assert df["set"].null_count() == 0
    assert df["word"].null_count() == 0


def test_time_and_formant_columns_preserved():
    df = _make_df(["KIT_sit"])
    assert df["time"][0] == pytest.approx(0.0)
    assert df["F1"][0] == pytest.approx(500.0)
    assert df["F2"][0] == pytest.approx(1500.0)
    assert df["F3"][0] == pytest.approx(2500.0)
