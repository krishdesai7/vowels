import pytest

from vowels.pipeline.nucleus import (
    SECOND_VOWEL_CENTER_RATIO,
    diphthong_times,
    disyllabic_time,
    nucleus_time,
)


def test_nucleus_midpoint():
    assert nucleus_time(0.0, 1.0) == pytest.approx(0.5)
    assert nucleus_time(1.0, 3.0) == pytest.approx(2.0)


def test_diphthong_times_placement():
    t_on, t_off = diphthong_times(0.0, 1.0)
    assert t_on == pytest.approx(0.25)
    assert t_off == pytest.approx(0.75)


def test_diphthong_times_non_zero_start():
    t_on, t_off = diphthong_times(2.0, 4.0)
    assert t_on == pytest.approx(2.5)
    assert t_off == pytest.approx(3.5)


def test_disyllabic_ratio():
    # CVCV model: C=1, V=2 → second vowel center at 5/6 into interval
    assert SECOND_VOWEL_CENTER_RATIO == pytest.approx(5 / 6)


def test_disyllabic_time():
    # Interval [0, 6] → second vowel at 5.0
    assert disyllabic_time(0.0, 6.0) == pytest.approx(5.0)
