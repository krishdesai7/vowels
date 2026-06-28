from pathlib import Path

import numpy as np
import parselmouth
import polars as pl

from ..labels import (
    get_set_name,
    is_diphthong_set,
    normalize_label,
)
from ..paths import session_dir
from ..schema import Gender


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
    normalized, disyll = normalize_label(label)
    set_name = get_set_name(normalized)
    word = normalized.split("_", 1)[1] if "_" in normalized else ""
    diph = is_diphthong_set(set_name)
    span = (t2 - t1) or 1.0
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
                "F1": frame["F1"], "F2": frame["F2"], "F3": frame["F3"],
                "F1_s": frame["F1_s"], "F2_s": frame["F2_s"], "F3_s": frame["F3_s"],
                "B1": frame["B1"], "B2": frame["B2"], "B3": frame["B3"],
                "max_formant": frame["max_formant"], "error": frame["error"],
            }
        )
    return rows


def extract_formants(session: str, gender: Gender = Gender.M) -> None:
    d: Path = session_dir(session)
    wav_path: Path = d / f"{session}.wav"
    tg_path: Path = d / f"{session}_nucleus.TextGrid"

    formant_ceiling: int = 5000 if gender == Gender.M else 5500
    window_length: float = 0.025 if gender == Gender.M else 0.030
    pitch_floor: int = 75 if gender == Gender.M else 100
    pitch_ceiling: int = 300 if gender == Gender.M else 500

    sound: parselmouth.Sound = parselmouth.Sound(wav_path.as_posix())
    tg: parselmouth.TextGrid = parselmouth.read(tg_path.as_posix())

    n_tiers: int = parselmouth.praat.call(tg, "Get number of tiers")
    tier_index: int | None = next(
        (
            i
            for i in range(1, n_tiers + 1)
            if parselmouth.praat.call(tg, "Get tier name", i) == "nucleus"
        ),
        None,
    )
    if tier_index is None:
        raise ValueError("No 'nucleus' tier found in TextGrid")

    n_points: int = parselmouth.praat.call(tg, "Get number of points", tier_index)
    print(f"Found {n_points} vowel nuclei")

    formant_obj: parselmouth.Formant = parselmouth.praat.call(
        sound, "To Formant (burg)", 0.0, 5, formant_ceiling, window_length, 50
    )
    pitch_obj: parselmouth.Pitch = parselmouth.praat.call(
        sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling
    )
    records: list[dict[str, float | str]] = []
    for i in range(1, n_points + 1):
        time: float = parselmouth.praat.call(tg, "Get time of point", tier_index, i)
        label: str | None = parselmouth.praat.call(
            tg, "Get label of point", tier_index, i
        )
        if not label:
            continue
        f0: float = parselmouth.praat.call(
            pitch_obj, "Get value at time", time, "Hertz", "Linear"
        )
        f1: float = parselmouth.praat.call(
            formant_obj, "Get value at time", 1, time, "Hertz", "Linear"
        )
        f2: float = parselmouth.praat.call(
            formant_obj, "Get value at time", 2, time, "Hertz", "Linear"
        )
        f3: float = parselmouth.praat.call(
            formant_obj, "Get value at time", 3, time, "Hertz", "Linear"
        )
        records.append(
            {"time": time, "label": label, "F0": f0, "F1": f1, "F2": f2, "F3": f3}
        )

    parse_labels(pl.LazyFrame(records)).sink_parquet(d / f"{session}_formants.parquet")
    print(f"Created {d / f'{session}_formants.parquet'}")
