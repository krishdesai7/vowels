import numpy as np
import polars as pl

from vowels.aggregate import collapse_token, steady_state_index


def test_picks_flat_region_in_center() -> None:
    # Moving edges, flat plateau in the middle -> min velocity in the plateau.
    rel = np.linspace(0.0, 1.0, 11)
    f1 = np.array([300, 400, 500, 500, 500, 500, 500, 500, 600, 700, 800], float)
    f2 = np.array([2000] * 11, float)
    idx = steady_state_index(f1, f2, rel, 0.2, 0.8)
    assert 0.2 <= rel[idx] <= 0.8
    # plateau frames have zero velocity; chosen frame must be on the plateau
    assert f1[idx] == 500


def test_window_restricts_search() -> None:
    # Flat everywhere except a dip near the start that is outside [0.55, 0.9].
    rel = np.linspace(0.0, 1.0, 11)
    f1 = np.array([500, 480, 500, 500, 500, 500, 500, 500, 500, 500, 500], float)
    f2 = np.array([2000] * 11, float)
    idx = steady_state_index(f1, f2, rel, 0.55, 0.9)
    assert 0.55 <= rel[idx] <= 0.9


def test_normalization_balances_f1_f2() -> None:
    # F2 swings by a large Hz amount, F1 by a small one, but equal in z-units.
    # Without normalization the F2 frame would dominate; with it, the flat
    # frames (zero velocity) win. Assert the F2 jump is NOT selected and that a
    # genuinely flat frame is chosen.
    rel = np.linspace(0.0, 1.0, 5)
    f1 = np.array([500, 500, 500, 510, 500], float)
    f2 = np.array([1500, 1500, 1500, 1500, 2500], float)
    idx = steady_state_index(f1, f2, rel, 0.0, 1.0)
    assert idx != 4  # the big raw-Hz F2 jump must not automatically win
    assert f1[idx] == 500.0  # a flat frame, not the F1 nudge at frame 3


def test_empty_window_falls_back_to_all_frames() -> None:
    rel = np.linspace(0.0, 1.0, 5)
    f1 = np.array([500.0, 500.0, 490.0, 500.0, 500.0], float)
    f2 = np.array([1500.0] * 5, float)
    # Window [1.5, 2.0] contains no frames -> fallback searches all frames
    idx = steady_state_index(f1, f2, rel, 1.5, 2.0)
    assert 0 <= idx <= 4  # returned a valid index, did not raise


def _frames(rel, f1, f2, f3=None, f0=None) -> pl.DataFrame:
    n = len(rel)
    return pl.DataFrame(
        {
            "rel_time": list(rel),
            "F0": list(f0) if f0 is not None else [120.0] * n,
            "F1_s": list(f1),
            "F2_s": list(f2),
            "F3_s": list(f3) if f3 is not None else [2500.0] * n,
        }
    )


def test_monophthong_yields_one_point() -> None:
    rel = np.linspace(0.0, 1.0, 11)
    df = _frames(rel, [300, 400, 500, 500, 500, 500, 500, 500, 600, 700, 800],
                 [2000] * 11)
    rows = collapse_token(df, "TRAP_cat")
    assert len(rows) == 1
    assert rows[0]["label"] == "TRAP_cat"
    assert rows[0]["set"] == "TRAP"
    assert rows[0]["word"] == "cat"
    assert rows[0]["F1"] == 500.0


def test_diphthong_yields_two_suffixed_points() -> None:
    rel = np.linspace(0.0, 1.0, 11)
    df = _frames(rel, [400] * 11, [2200] * 11)
    rows = collapse_token(df, "PRICE_buy")
    assert [r["label"] for r in rows] == ["PRICE_buy:1", "PRICE_buy:2"]
    assert all(r["set"] == "PRICE" for r in rows)


def test_disyllabic_targets_second_syllable_window() -> None:
    rel = np.linspace(0.0, 1.0, 21)
    # Flat plateau only in the second-syllable window region (~0.83).
    f1 = np.array([500.0] * 21)
    f1[:14] += np.linspace(0, 60, 14)  # earlier frames vary
    df = _frames(rel, f1, [1600] * 21)
    rows = collapse_token(df, "2leTTER_butter")
    assert len(rows) == 1
    assert rows[0]["set"] == "leTTER"
    assert rows[0]["word"] == "butter"


def test_nan_f0_falls_back_to_token_mean() -> None:
    rel = np.linspace(0.0, 1.0, 5)
    f0 = [110.0, 130.0, float("nan"), 130.0, 110.0]
    df = _frames(rel, [500] * 5, [1500] * 5, f0=f0)
    rows = collapse_token(df, "KIT_bit")
    assert rows[0]["F0"] > 0  # not NaN
