from pathlib import Path

import pytest

from vowels.labels import (
    DIPHTHONG_NAMES,
    DIPHTHONGS,
    DISYLLABLE_PREFIX,
    SECOND_VOWEL_CENTER_RATIO,
    get_set_name,
    is_diphthong_set,
    is_disyllabic,
    normalize_label,
    read_labels,
    row_to_label,
)
from vowels.paths import project_root
from vowels.schema import Wells


def test_disyllable_prefix_constant() -> None:
    assert DISYLLABLE_PREFIX == "2"


def test_second_vowel_center_ratio() -> None:
    # CVCV weighting C=1, V=2 -> second vowel center at 5/6 of the interval
    assert SECOND_VOWEL_CENTER_RATIO == pytest.approx(5 / 6)


def test_normalize_label_strips_disyllabic_prefix() -> None:
    assert normalize_label("2haPPY_coffee") == ("haPPY_coffee", True)
    assert normalize_label("FLEECE_beat") == ("FLEECE_beat", False)


def test_get_set_name() -> None:
    assert get_set_name("FLEECE_beat") == "FLEECE"
    assert get_set_name("haPPY_coffee") == "haPPY"
    assert get_set_name("STRUT") == "STRUT"


def test_is_disyllabic() -> None:
    assert is_disyllabic("2leTTER_butter") is True
    assert is_disyllabic("TRAP_cat") is False


def test_is_diphthong_set() -> None:
    assert is_diphthong_set("PRICE") is True
    assert is_diphthong_set("FACE") is True  # FACE is a diphthong in this schema
    assert is_diphthong_set("FLEECE") is False


def test_diphthong_names_match_schema() -> None:
    assert DIPHTHONG_NAMES == frozenset(w.name for w in DIPHTHONGS)


def test_row_to_label() -> None:
    assert row_to_label("1", "FLEECE", "beat") == "FLEECE_beat"
    assert row_to_label("2", "haPPY", "coffee") == "2haPPY_coffee"


def test_row_to_label_preserves_mixed_case() -> None:
    # haPPY/coMMA/leTTER are case-significant; normalising would corrupt them
    assert row_to_label("2", "leTTER", "butter") == "2leTTER_butter"
    assert row_to_label("2", "coMMA", "comma") == "2coMMA_comma"


def test_row_to_label_rejects_other_syllables() -> None:
    with pytest.raises(ValueError, match="syllable must be 1 or 2"):
        row_to_label("3", "FLEECE", "beat")


def _write_csv(tmp_path: Path, text: str) -> Path:
    path: Path = tmp_path / "labels.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_read_labels(tmp_path: Path) -> None:
    path: Path = _write_csv(
        tmp_path,
        "syllable,set,word\n1,CURE,your\n1,GOAT,home\n2,coMMA,comma\n",
    )
    assert read_labels(path) == ["CURE_your", "GOAT_home", "2coMMA_comma"]


def test_read_labels_rejects_missing_columns(tmp_path: Path) -> None:
    path: Path = _write_csv(tmp_path, "set,word\nCURE,your\n")
    with pytest.raises(ValueError, match="missing column\\(s\\): syllable"):
        read_labels(path)


def test_read_labels_on_the_shipped_default_list() -> None:
    labels: list[str] = read_labels(project_root() / "data" / "labels.csv")
    assert len(labels) == 99
    assert labels[:2] == ["CURE_your", "GOAT_home"]
    # Every set the default list names must be a known Wells set
    sets: set[str] = {get_set_name(normalize_label(lbl)[0]) for lbl in labels}
    assert sets <= {w.name for w in Wells}
