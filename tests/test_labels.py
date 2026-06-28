import pytest

from vowels.labels import (
    DIPHTHONG_NAMES,
    DISYLLABLE_PREFIX,
    SECOND_VOWEL_CENTER_RATIO,
    get_set_name,
    is_diphthong_set,
    is_disyllabic,
    normalize_label,
)


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
    from vowels.schema import DIPHTHONGS

    assert DIPHTHONG_NAMES == frozenset(w.name for w in DIPHTHONGS)
