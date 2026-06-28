from typing import Final

from .schema import DIPHTHONGS

DISYLLABLE_PREFIX: Final[str] = "2"
CONSONANT_WEIGHT: Final[float] = 1.0
VOWEL_WEIGHT: Final[float] = 2.0
_TOTAL_WEIGHT: Final[float] = 2 * (CONSONANT_WEIGHT + VOWEL_WEIGHT)
SECOND_VOWEL_CENTER_RATIO: Final[float] = (
    CONSONANT_WEIGHT + VOWEL_WEIGHT + CONSONANT_WEIGHT + (0.5 * VOWEL_WEIGHT)
) / _TOTAL_WEIGHT

DIPHTHONG_NAMES: Final[frozenset[str]] = frozenset(w.name for w in DIPHTHONGS)


def normalize_label(label: str) -> tuple[str, bool]:
    if label.startswith(DISYLLABLE_PREFIX):
        return label[len(DISYLLABLE_PREFIX) :], True
    return label, False


def get_set_name(label: str) -> str:
    if "_" in label:
        return label.split("_", 1)[0]
    return label


def is_disyllabic(label: str) -> bool:
    return label.startswith(DISYLLABLE_PREFIX)


def is_diphthong_set(set_name: str) -> bool:
    return set_name in DIPHTHONG_NAMES
