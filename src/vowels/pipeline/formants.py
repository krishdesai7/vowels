from typing import Literal

import parselmouth
import polars as pl
from parselmouth.praat import call

from vowels.paths import session_dir


def parse_labels(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("label").str.split_exact("_", 1).alias("parts")
        )
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
            pl.col("raw_word")
            .str.replace_all(r":\d+", "")
            .alias("word"),
        )
        .drop(["raw_set", "raw_word"])
    )


def extract_formants(
    session: str, gender: Literal["M", "F", "C"] = "M"
) -> None:
    d = session_dir(session)
    wav_path = d / f"{session}.wav"
    tg_path = d / f"{session}_nucleus.TextGrid"

    formant_ceiling = 5000 if gender == "M" else 5500
    window_length = 0.025 if gender == "M" else 0.030

    sound = parselmouth.Sound(str(wav_path))
    tg = parselmouth.read(str(tg_path))

    n_tiers = call(tg, "Get number of tiers")
    tier_index = next(
        (i for i in range(1, n_tiers + 1) if call(tg, "Get tier name", i) == "nucleus"),
        None,
    )
    if tier_index is None:
        raise ValueError("No 'nucleus' tier found in TextGrid")

    n_points = call(tg, "Get number of points", tier_index)
    print(f"Found {n_points} vowel nuclei")

    formant_obj = call(
        sound, "To Formant (burg)", 0.0, 5, formant_ceiling, window_length, 50
    )
    records: list[dict] = []
    for i in range(1, n_points + 1):
        time = call(tg, "Get time of point", tier_index, i)
        label = call(tg, "Get label of point", tier_index, i)
        if not label:
            continue
        f1 = call(formant_obj, "Get value at time", 1, time, "Hertz", "Linear")
        f2 = call(formant_obj, "Get value at time", 2, time, "Hertz", "Linear")
        f3 = call(formant_obj, "Get value at time", 3, time, "Hertz", "Linear")
        records.append({"time": time, "label": label, "F1": f1, "F2": f2, "F3": f3})

    df = pl.DataFrame(records)
    df = parse_labels(df)
    df.write_csv(d / f"{session}_formants.csv")
    print(f"Created {d / f'{session}_formants.csv'}")
