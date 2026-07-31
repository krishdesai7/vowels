import csv
from pathlib import Path
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


LABEL_COLUMNS: Final[tuple[str, ...]] = ("syllable", "set", "word")


def row_to_label(syllable: str, set_name: str, word: str) -> str:
    """Build a pipeline label from one `labels.csv` row.

    `syllable` is which syllable of the word carries the target vowel; a target
    in the second syllable is marked with the `2` prefix the parser strips back
    off (see `normalize_label`). Only monosyllabic and disyllabic targets have a
    label spelling, so anything else is rejected rather than silently prefixed.
    """
    n: int = int(syllable)
    if n not in (1, 2):
        raise ValueError(f"syllable must be 1 or 2, got {syllable!r}")
    prefix: str = DISYLLABLE_PREFIX if n == 2 else ""
    # Strip padding a hand-edited CSV may carry; case is left alone on purpose.
    return f"{prefix}{set_name.strip()}_{word.strip()}"


def read_labels(path: Path) -> list[str]:
    """Read a `labels.csv` (`syllable,set,word`) into pipeline label strings.

    Set names are taken verbatim: `haPPY`, `coMMA`, and `leTTER` are
    case-significant and must not be normalised.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader: csv.DictReader[str] = csv.DictReader(f)
        missing: list[str] = [
            c for c in LABEL_COLUMNS if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"{path} is missing column(s): {', '.join(missing)}")
        return [
            row_to_label(row["syllable"], row["set"], row["word"]) for row in reader
        ]
