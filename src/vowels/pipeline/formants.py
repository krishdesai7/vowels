from pathlib import Path

import numpy as np
import parselmouth
import polars as pl
from fasttrackpy import CandidateTracks, OneTrack, process_audio_file
from numpy.typing import NDArray

from ..labels import (
    get_set_name,
    is_diphthong_set,
    normalize_label,
)
from ..paths import session_dir
from ..schema import Gender

_MIN_DURATION: float = 0.05


def parse_labels(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.with_columns(pl.col("label").str.split_exact("_", 1).alias("parts"))
        .with_columns(
            pl.col("parts").struct.field("field_0").alias("raw_set"),
            pl.col("parts").struct.field("field_1").alias("raw_word"),
        )
        .drop("parts")
        .with_columns(
            pl.col("raw_set")
            .str.replace_all(r":\d+", "")
            .str.replace(r"^2([A-Za-z])", "$1")
            .alias("set"),
            pl.col("raw_word").str.replace_all(r":\d+", "").alias("word"),
        )
        .drop(["raw_set", "raw_word"])
    )


def winner_to_rows(
    winner_df: pl.DataFrame,
    f0: np.ndarray,
    token_id: int,
    label: str,
    t1: float,
    t2: float,
) -> list[dict]:
    if len(f0) != winner_df.height:
        raise ValueError(f"f0 length {len(f0)} != winner_df rows {winner_df.height}")
    normalized, disyll = normalize_label(label)
    set_name: str = get_set_name(normalized)
    word: str = normalized.split("_", 1)[1] if "_" in normalized else ""
    diph: bool = is_diphthong_set(set_name)
    span: float = (t2 - t1) or 1.0
    rows: list[dict] = []
    for i, frame in enumerate(winner_df.iter_rows(named=True)):
        rows.append(
            {
                "token_id": token_id,
                "label": label,
                "set": set_name,
                "word": word,
                "is_diphthong": diph,
                "is_disyllabic": disyll,
                "time": frame["time"],
                "rel_time": (frame["time"] - t1) / span,
                "F0": float(f0[i]),
                "F1": frame["F1"],
                "F2": frame["F2"],
                "F3": frame["F3"],
                "F1_s": frame["F1_s"],
                "F2_s": frame["F2_s"],
                "F3_s": frame["F3_s"],
                "B1": frame["B1"],
                "B2": frame["B2"],
                "B3": frame["B3"],
                "max_formant": frame["max_formant"],
                "error": frame["error"],
            }
        )
    return rows


def _gender_params(gender: Gender) -> dict[str, float | int]:
    if gender == Gender.M:
        return {
            "min_max_formant": 4500,
            "max_max_formant": 5500,
            "window_length": 0.025,
            "pitch_floor": 75,
        }
    return {
        "min_max_formant": 5000,
        "max_max_formant": 6500,
        "window_length": 0.030,
        "pitch_floor": 100,
    }


def extract_formants(session: str, gender: Gender = Gender.M) -> None:
    d: Path = session_dir(session)
    wav_path: Path = d / f"{session}.wav"
    tg_path: Path = d / f"{session}_labeled.TextGrid"
    params: dict[str, float | int] = _gender_params(gender)

    tg: parselmouth.TextGrid = parselmouth.read(tg_path.as_posix())
    labeled_tier: int = 1
    n_intervals: int = parselmouth.praat.call(
        tg, "Get number of intervals", labeled_tier
    )

    rows: list[dict] = []
    token_id: int = 0
    for i in range(1, n_intervals + 1):
        label: str | None = parselmouth.praat.call(
            tg, "Get label of interval", labeled_tier, i
        )
        if not label or label.lower() == "silent":
            continue
        t1: float = parselmouth.praat.call(
            tg, "Get start time of interval", labeled_tier, i
        )
        t2: float = parselmouth.praat.call(
            tg, "Get end time of interval", labeled_tier, i
        )
        if t2 - t1 < _MIN_DURATION:
            continue

        candidates: CandidateTracks = process_audio_file(
            wav_path.as_posix(),
            xmin=t1,
            xmax=t2,
            n_formants=4,
            **params,  # type: ignore
        )
        winner: OneTrack = candidates.winner
        # to_df(output="formants") already yields columns:
        # F1..F4, F1_s..F4_s, B1..B4, error, time, max_formant, n_formant,
        # smooth_method (+ metadata). winner_to_rows selects the keys it needs.
        winner_df: pl.DataFrame = winner.to_df(output="formants")
        # candidates.f0 is sampled at winner.time_domain -> 1:1 with winner_df rows.
        f0: NDArray[np.double] = candidates.f0
        rows.extend(winner_to_rows(winner_df, f0, token_id, label, t1, t2))
        token_id += 1

    if not rows:
        print(
            f"Warning: no qualifying vowel intervals found in {session}; "
            "no parquet written"
        )
        return
    pl.DataFrame(rows).write_parquet(d / f"{session}_formants.parquet")
    print(f"Created {d / f'{session}_formants.parquet'} ({token_id} tokens)")
