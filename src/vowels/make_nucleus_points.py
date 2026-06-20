import fire
import parselmouth
from parselmouth.praat import call
from typing import Literal
from pathlib import Path


def make_nucleus_points(session: str) -> None:
    dir: Path = Path(__file__).parent / "sessions" / session
    in_tg: Path = dir / f"{session}_labeled.TextGrid"
    out_tg: Path = dir / f"{session}_nucleus.TextGrid"

    LABELED_TIER_NUMBER: Literal[1] = 1
    DIPHTHONGS: set[str] = {
        # "FACE", "GOAT", "PRICE", "CHOICE", "MOUTH",
        # "NEAR", "SQUARE", "CURE"
    }
    DISYLLABLE_PREFIX: str = "2"

    # Assume CVCV with vowels ≈2× consonant duration;
    # center of final V sits ~5/6 into the interval.
    CONSONANT_WEIGHT: float = 1.0
    VOWEL_WEIGHT: float = 2.0
    TOTAL_WEIGHT: float = 2 * (CONSONANT_WEIGHT + VOWEL_WEIGHT)
    SECOND_VOWEL_CENTER_RATIO: float = (
        CONSONANT_WEIGHT + VOWEL_WEIGHT + CONSONANT_WEIGHT + (0.5 * VOWEL_WEIGHT)
    ) / TOTAL_WEIGHT

    def get_set_name(label: str) -> str:
        if "_" in label:
            return label.split("_", 1)[0].upper()
        return label.upper()

    def normalize_label(label: str) -> tuple[str, bool]:
        if label.startswith(DISYLLABLE_PREFIX):
            return label[len(DISYLLABLE_PREFIX):], True
        return label, False

    tg: parselmouth.Data = parselmouth.read(str(in_tg))
    n_tiers: int = call(tg, "Get number of tiers")
    call(tg, "Insert point tier", n_tiers + 1, "nucleus")
    NUCLEUS_TIER_NUMBER: int = n_tiers + 1

    n_intervals: int = call(tg, "Get number of intervals", LABELED_TIER_NUMBER)

    for i in range(1, n_intervals + 1):
        label: str = call(tg, "Get label of interval", LABELED_TIER_NUMBER, i)
        if not label or label.lower() == "silent":
            continue

        t1: float = call(tg, "Get start time of interval", LABELED_TIER_NUMBER, i)
        t2: float = call(tg, "Get end time of interval",   LABELED_TIER_NUMBER, i)
        dur: float = t2 - t1
        if dur <= 0:
            continue

        normalized_label, is_disyllabic = normalize_label(label)
        set_name: str = get_set_name(normalized_label)

        if is_disyllabic:
            t_v2: float = t1 + SECOND_VOWEL_CENTER_RATIO * dur
            call(tg, "Insert point", NUCLEUS_TIER_NUMBER, t_v2, label)
            continue

        if set_name in DIPHTHONGS:
            # Two points for diphthongs: ~25% and ~75% of the word interval
            t_on: float  = t1 + 0.25 * dur
            t_off: float = t1 + 0.75 * dur
            call(tg, "Insert point", NUCLEUS_TIER_NUMBER, t_on,  f"{label}:1")
            call(tg, "Insert point", NUCLEUS_TIER_NUMBER, t_off, f"{label}:2")
        else:
            # One midpoint for monophthongs/reduced/r-colored
            t_mid: float = t1 + 0.5 * dur
            call(tg, "Insert point", NUCLEUS_TIER_NUMBER, t_mid, label)
    
    call(tg, "Write to text file", str(out_tg))


def main() -> None:
    fire.Fire(make_nucleus_points)


if __name__ == "__main__":
    main()
