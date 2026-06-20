import parselmouth
from parselmouth.praat import call

from vowels.paths import session_dir

DIPHTHONGS: set[str] = {
    "FACE",
    "GOAT",
    "PRICE",
    "CHOICE",
    "MOUTH",
    "NEAR",
    "SQUARE",
    "CURE",
}

DISYLLABLE_PREFIX = "2"
CONSONANT_WEIGHT = 1.0
VOWEL_WEIGHT = 2.0
TOTAL_WEIGHT = 2 * (CONSONANT_WEIGHT + VOWEL_WEIGHT)
SECOND_VOWEL_CENTER_RATIO = (
    CONSONANT_WEIGHT + VOWEL_WEIGHT + CONSONANT_WEIGHT + (0.5 * VOWEL_WEIGHT)
) / TOTAL_WEIGHT


def nucleus_time(t1: float, t2: float) -> float:
    return t1 + 0.5 * (t2 - t1)


def diphthong_times(t1: float, t2: float) -> tuple[float, float]:
    dur = t2 - t1
    return t1 + 0.25 * dur, t1 + 0.75 * dur


def disyllabic_time(t1: float, t2: float) -> float:
    return t1 + SECOND_VOWEL_CENTER_RATIO * (t2 - t1)


def get_set_name(label: str) -> str:
    if "_" in label:
        return label.split("_", 1)[0].upper()
    return label.upper()


def normalize_label(label: str) -> tuple[str, bool]:
    if label.startswith(DISYLLABLE_PREFIX):
        return label[len(DISYLLABLE_PREFIX):], True
    return label, False


def make_nucleus_points(session: str, diphthongs: bool = True) -> None:
    d = session_dir(session)
    in_tg = d / f"{session}_labeled.TextGrid"
    out_tg = d / f"{session}_nucleus.TextGrid"

    active_diphthongs = DIPHTHONGS if diphthongs else set()

    labeled_tier = 1
    tg = parselmouth.read(str(in_tg))
    n_tiers = call(tg, "Get number of tiers")
    call(tg, "Insert point tier", n_tiers + 1, "nucleus")
    nucleus_tier = n_tiers + 1

    n_intervals = call(tg, "Get number of intervals", labeled_tier)
    for i in range(1, n_intervals + 1):
        label = call(tg, "Get label of interval", labeled_tier, i)
        if not label or label.lower() == "silent":
            continue

        t1 = call(tg, "Get start time of interval", labeled_tier, i)
        t2 = call(tg, "Get end time of interval", labeled_tier, i)
        if t2 - t1 <= 0:
            continue

        normalized, is_disyllabic = normalize_label(label)
        set_name = get_set_name(normalized)

        if is_disyllabic:
            call(tg, "Insert point", nucleus_tier, disyllabic_time(t1, t2), label)
            continue

        if set_name in active_diphthongs:
            t_on, t_off = diphthong_times(t1, t2)
            call(tg, "Insert point", nucleus_tier, t_on, f"{label}:1")
            call(tg, "Insert point", nucleus_tier, t_off, f"{label}:2")
        else:
            call(tg, "Insert point", nucleus_tier, nucleus_time(t1, t2), label)

    call(tg, "Write to text file", str(out_tg))
    print(f"Created {out_tg}")
