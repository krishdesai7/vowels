from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal

import parselmouth

from .. import DIPHTHONGS, Mode, Wells, session_dir

DISYLLABLE_PREFIX: Final[Literal["2"]] = "2"
CONSONANT_WEIGHT: Final[float] = 1.0
VOWEL_WEIGHT: Final[float] = 2.0
TOTAL_WEIGHT: Final[float] = 2 * (CONSONANT_WEIGHT + VOWEL_WEIGHT)
SECOND_VOWEL_CENTER_RATIO: Final[float] = (
    CONSONANT_WEIGHT + VOWEL_WEIGHT + CONSONANT_WEIGHT + (0.5 * VOWEL_WEIGHT)
) / TOTAL_WEIGHT


nucleus_time: Callable[[float, float], float] = lambda t1, t2: t1 + 0.5 * (t2 - t1)  # noqa: E731


diphthong_times: Callable[[float, float], tuple[float, float]] = lambda t1, t2: (  # noqa: E731
    t1 + 0.25 * (t2 - t1),
    t1 + 0.75 * (t2 - t1),
)

disyllabic_time: Callable[[float, float], float] = lambda t1, t2: (  # noqa: E731
    t1 + SECOND_VOWEL_CENTER_RATIO * (t2 - t1)
)


def get_set_name(label: str) -> str:
    if "_" in label:
        return label.split("_", 1)[0].upper()
    return label.upper()


def normalize_label(label: str) -> tuple[str, bool]:
    if label.startswith(DISYLLABLE_PREFIX):
        return label[len(DISYLLABLE_PREFIX) :], True
    return label, False


def make_nucleus_points(session: str, mode: Mode = Mode.MONO) -> None:
    d: Path = session_dir(session)
    in_tg: Path = d / f"{session}_labeled.TextGrid"
    out_tg: Path = d / f"{session}_nucleus.TextGrid"

    active_diphthongs: set[Wells] = DIPHTHONGS if mode == Mode.DIPH else set()

    labeled_tier: int = 1
    tg: parselmouth.TextGrid = parselmouth.read(in_tg.as_posix())
    n_tiers: int = parselmouth.praat.call(tg, "Get number of tiers")
    parselmouth.praat.call(tg, "Insert point tier", n_tiers + 1, "nucleus")
    nucleus_tier: int = n_tiers + 1

    n_intervals: int = parselmouth.praat.call(
        tg, "Get number of intervals", labeled_tier
    )
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
        if t2 - t1 <= 0:
            continue

        normalized, is_disyllabic = normalize_label(label)
        set_name: str = get_set_name(normalized)

        if is_disyllabic:
            parselmouth.praat.call(
                tg, "Insert point", nucleus_tier, disyllabic_time(t1, t2), label
            )
            continue

        if set_name in active_diphthongs:
            t_on, t_off = diphthong_times(t1, t2)
            parselmouth.praat.call(tg, "Insert point", nucleus_tier, t_on, f"{label}:1")
            parselmouth.praat.call(
                tg, "Insert point", nucleus_tier, t_off, f"{label}:2"
            )
        else:
            parselmouth.praat.call(
                tg, "Insert point", nucleus_tier, nucleus_time(t1, t2), label
            )

    parselmouth.praat.call(tg, "Write to text file", out_tg.as_posix())
    print(f"Created {out_tg}")
