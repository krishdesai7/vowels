from pathlib import Path
from types import SimpleNamespace
from typing import Final

import numpy as np
import parselmouth
import polars as pl
import pytest

from vowels import parse_labels
from vowels.pipeline.formants import _gender_params, extract_formants, winner_to_rows
from vowels.schema import Gender

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


# ---------------------------------------------------------------------------
# _gender_params helper
# ---------------------------------------------------------------------------


def test_gender_params_male() -> None:
    assert _gender_params(Gender.M) == {
        "min_max_formant": 4500,
        "max_max_formant": 5500,
        "window_length": 0.025,
        "pitch_floor": 75,
    }


def test_gender_params_non_male() -> None:
    expected = {
        "min_max_formant": 5000,
        "max_max_formant": 6500,
        "window_length": 0.030,
        "pitch_floor": 100,
    }
    assert _gender_params(Gender.F) == expected
    assert _gender_params(Gender.C) == expected


# ---------------------------------------------------------------------------
# extract_formants orchestration
# ---------------------------------------------------------------------------


def test_extract_formants_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify extract_formants loops correctly: skips silent/empty intervals
    and writes the expected trajectory parquet schema."""
    import vowels.pipeline.formants as formants_mod

    session = "s_test"
    session_d = tmp_path / session
    session_d.mkdir()

    # Redirect all session path resolution into tmp_path
    monkeypatch.setattr(formants_mod, "session_dir", lambda _: session_d)

    # Placeholder WAV — contents irrelevant because process_audio_file is mocked
    (session_d / f"{session}.wav").write_bytes(b"")

    # Build a real TextGrid via parselmouth and write to file
    # Intervals: [0.00, 0.05] "silent"  (skipped — label check)
    #            [0.05, 0.50] "FLEECE_beat"  (processed, token_id 0)
    #            [0.50, 1.00] "PRICE_buy"   (processed, token_id 1)
    tg = parselmouth.praat.call("Create TextGrid", 0.0, 1.0, "silences", "")
    parselmouth.praat.call(tg, "Insert boundary", 1, 0.05)
    parselmouth.praat.call(tg, "Insert boundary", 1, 0.5)
    parselmouth.praat.call(tg, "Set interval text", 1, 1, "silent")
    parselmouth.praat.call(tg, "Set interval text", 1, 2, "FLEECE_beat")
    parselmouth.praat.call(tg, "Set interval text", 1, 3, "PRICE_buy")
    tg_path = session_d / f"{session}_labeled.TextGrid"
    parselmouth.praat.call(tg, "Write to text file", str(tg_path))

    # Stub: process_audio_file -> fake candidates with 3 frames each call
    n_frames = 3
    _winner_df = pl.DataFrame({
        "time": [0.1, 0.2, 0.3],
        "F1": [400.0] * n_frames,
        "F2": [2000.0] * n_frames,
        "F3": [2800.0] * n_frames,
        "F1_s": [400.0] * n_frames,
        "F2_s": [2000.0] * n_frames,
        "F3_s": [2800.0] * n_frames,
        "B1": [50.0] * n_frames,
        "B2": [80.0] * n_frames,
        "B3": [120.0] * n_frames,
        "max_formant": [5000.0] * n_frames,
        "error": [0.01] * n_frames,
    })
    _f0 = np.array([120.0, 121.0, 119.0])
    _fake_winner = SimpleNamespace(to_df=lambda output: _winner_df)
    _fake_candidates = SimpleNamespace(winner=_fake_winner, f0=_f0)
    monkeypatch.setattr(
        formants_mod, "process_audio_file", lambda *a, **kw: _fake_candidates
    )

    extract_formants(session, Gender.M)

    out = pl.read_parquet(session_d / f"{session}_formants.parquet")

    EXPECTED_COLS = {
        "token_id", "label", "set", "word", "is_diphthong", "is_disyllabic",
        "time", "rel_time", "F0", "F1", "F2", "F3",
        "F1_s", "F2_s", "F3_s", "B1", "B2", "B3", "max_formant", "error",
    }
    assert EXPECTED_COLS.issubset(set(out.columns))

    # Silent interval was skipped -> exactly 2 unique token_ids
    assert out["token_id"].n_unique() == 2

    fleece = out.filter(pl.col("set") == "FLEECE")
    price = out.filter(pl.col("set") == "PRICE")
    assert not fleece.is_empty()
    assert not price.is_empty()
    # FLEECE is a monophthong; PRICE is a diphthong
    assert fleece["is_diphthong"][0] == False  # noqa: E712
    assert price["is_diphthong"][0] == True  # noqa: E712
