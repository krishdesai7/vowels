from collections.abc import Callable
from pathlib import Path

import parselmouth
import polars as pl

from .. import Gender, session_dir

parse_labels: Callable[[pl.DataFrame], pl.DataFrame] = lambda df: (  # noqa: E731
    df.with_columns(pl.col("label").str.split_exact("_", 1).alias("parts"))
    .with_columns(
        pl.col("parts").struct.field("field_0").alias("raw_set"),
        pl.col("parts").struct.field("field_1").alias("raw_word"),
    )
    .drop("parts")
    .with_columns(
        pl.col("raw_set")
        .str.replace_all(r":\d+", "")
        .str.to_uppercase()
        .str.replace(r"^2([A-Z])", "$1")
        .alias("set"),
        pl.col("raw_word").str.replace_all(r":\d+", "").alias("word"),
    )
    .drop(["raw_set", "raw_word"])
)


def extract_formants(session: str, gender: Gender = Gender.M) -> None:
    d: Path = session_dir(session)
    wav_path: Path = d / f"{session}.wav"
    tg_path: Path = d / f"{session}_nucleus.TextGrid"

    formant_ceiling: int = 5000 if gender == Gender.M else 5500
    window_length: float = 0.025 if gender == Gender.M else 0.030

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
    records: list[dict[str, float | str]] = []
    for i in range(1, n_points + 1):
        time: float = parselmouth.praat.call(tg, "Get time of point", tier_index, i)
        label: str | None = parselmouth.praat.call(
            tg, "Get label of point", tier_index, i
        )
        if not label:
            continue
        f1: float = parselmouth.praat.call(
            formant_obj, "Get value at time", 1, time, "Hertz", "Linear"
        )
        f2: float = parselmouth.praat.call(
            formant_obj, "Get value at time", 2, time, "Hertz", "Linear"
        )
        f3: float = parselmouth.praat.call(
            formant_obj, "Get value at time", 3, time, "Hertz", "Linear"
        )
        records.append({"time": time, "label": label, "F1": f1, "F2": f2, "F3": f3})

    df: pl.DataFrame = pl.DataFrame(records)
    df = parse_labels(df)
    df.write_csv(d / f"{session}_formants.csv")
    print(f"Created {d / f'{session}_formants.csv'}")
