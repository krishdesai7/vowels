import math
from typing import cast

import numpy as np
import polars as pl
import pytest

from vowels.aggregate import (
    classify_sets,
    collapse_token,
    points_from_trajectory,
    steady_state_index,
    token_displacement,
)


def test_picks_flat_region_in_center() -> None:
    rel = np.linspace(0.0, 1.0, 11)
    d = np.array([300, 400, 500, 500, 500, 500, 500, 500, 600, 700, 800], float)
    idx = steady_state_index(d.reshape(-1, 1), rel, 0.2, 0.8)
    assert 0.2 <= rel[idx] <= 0.8
    assert d[idx] == 500


def test_window_restricts_search() -> None:
    rel = np.linspace(0.0, 1.0, 11)
    d = np.array([500, 480, 500, 500, 500, 500, 500, 500, 500, 500, 500], float)
    idx = steady_state_index(d.reshape(-1, 1), rel, 0.55, 0.9)
    assert 0.55 <= rel[idx] <= 0.9


def test_velocity_uses_all_bark_axes() -> None:
    # Axis 0 alone is slowest at frame 4, so ignoring axis 1 would select it.
    # Axis 1 jumps there, making frame 4 the fastest overall. Selecting frame 1
    # therefore proves both axes contribute to the velocity norm.
    rel = np.linspace(0.0, 1.0, 5)
    axis0 = np.array([0.0, 1.0, 3.0, 6.0, 6.0])  # per-frame steps 1, 2, 3, 0
    axis1 = np.array([0.0, 0.0, 0.0, 0.0, 5.0])  # jump on the last frame
    dims = np.column_stack([axis0, axis1])
    assert steady_state_index(axis0.reshape(-1, 1), rel, 0.0, 1.0) == 4
    assert steady_state_index(dims, rel, 0.0, 1.0) == 1


def test_empty_window_falls_back_to_all_frames() -> None:
    rel = np.linspace(0.0, 1.0, 5)
    d = np.array([500.0, 500.0, 490.0, 500.0, 500.0], float)
    idx = steady_state_index(d.reshape(-1, 1), rel, 1.5, 2.0)
    assert 0 <= idx <= 4


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
    df = _frames(
        rel, [300, 400, 500, 500, 500, 500, 500, 500, 600, 700, 800], [2000] * 11
    )
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
    f0 = [110.0, float("nan"), 130.0, 130.0, 110.0]
    df = _frames(rel, [500] * 5, [1500] * 5, f0=f0)
    rows = collapse_token(df, "KIT_bit")
    # Formants are flat and Z0 is anchored per token, so every frame has zero
    # Bark velocity and the first candidate (index 1) wins. Its F0 is NaN, so
    # the point must fall back to the token mean of 110, 130, 130, 110.
    assert rows[0]["F0"] == pytest.approx(120.0)


def test_all_nan_f0_still_yields_a_point() -> None:
    # A wholly unvoiced token (seen in real data) has no F0 to anchor Bark to,
    # so _with_bark returns None and selection falls back to z-scored Hz. The
    # token must still produce a usable point rather than blowing up.
    rel = np.linspace(0.0, 1.0, 5)
    f1 = [500.0, 500.0, 700.0, 500.0, 500.0]
    df = _frames(rel, f1, [1500] * 5, f0=[float("nan")] * 5)
    rows = collapse_token(df, "KIT_bit")
    assert len(rows) == 1
    assert math.isnan(cast(float, rows[0]["F0"]))
    assert rows[0]["F1"] in (500.0, 700.0)


def test_points_from_trajectory_one_row_per_mono_two_per_diph() -> None:
    rel = list(np.linspace(0.0, 1.0, 11))

    def block(token_id, label, f1, f2):
        return pl.DataFrame(
            {
                "token_id": [token_id] * 11,
                "label": [label] * 11,
                "rel_time": rel,
                "F0": [120.0] * 11,
                "F1_s": [float(f1)] * 11,
                "F2_s": [float(f2)] * 11,
                "F3_s": [2500.0] * 11,
            }
        )

    traj = pl.concat(
        [
            block(0, "TRAP_cat", 700, 1600),
            block(1, "PRICE_buy", 400, 2200),
        ]
    )
    points = points_from_trajectory(traj)
    assert set(points.columns) == {"label", "set", "word", "F0", "F1", "F2", "F3"}
    labels = sorted(points["label"].to_list())
    assert labels == ["PRICE_buy:1", "PRICE_buy:2", "TRAP_cat"]


def test_displacement_small_for_monophthong() -> None:
    rel = np.linspace(0.0, 1.0, 11)
    # Flat F1/F2/F3 -> onset and offset targets coincide -> ~0 displacement.
    df = _frames(rel, [500.0] * 11, [1500.0] * 11)
    assert token_displacement(df) < 0.2


def test_displacement_large_for_diphthong() -> None:
    rel = np.linspace(0.0, 1.0, 11)
    # F2 sweeps 900 -> 2300 Hz across the token -> big Frontness change.
    f2 = np.linspace(900.0, 2300.0, 11)
    df = _frames(rel, [500.0] * 11, list(f2))
    assert token_displacement(df) > 1.0


def test_displacement_nan_without_f0() -> None:
    rel = np.linspace(0.0, 1.0, 11)
    df = _frames(rel, [500.0] * 11, [1500.0] * 11, f0=[float("nan")] * 11)
    assert math.isnan(token_displacement(df))


def _token(token_id, label, f1, f2, n=11):
    rel = list(np.linspace(0.0, 1.0, n))
    return pl.DataFrame(
        {
            "token_id": [token_id] * n,
            "label": [label] * n,
            "rel_time": rel,
            "F0": [120.0] * n,
            "F1_s": [float(x) for x in f1]
            if hasattr(f1, "__len__")
            else [float(f1)] * n,
            "F2_s": [float(x) for x in f2]
            if hasattr(f2, "__len__")
            else [float(f2)] * n,
            "F3_s": [2500.0] * n,
        }
    )


def _sweep(a, b, n=11):
    return list(np.linspace(a, b, n))


def test_classify_flips_flat_canonical_diphthong_to_mono() -> None:
    # Baseline: several flat canonical-monophthong sets. A canonical diphthong
    # (GOAT) that is also flat must flip to monophthong; a canonical diphthong
    # that sweeps (PRICE) must stay a diphthong.
    traj = pl.concat(
        [
            _token(0, "KIT_bit", 400, 2000),
            _token(1, "DRESS_bed", 550, 1900),
            _token(2, "TRAP_cat", 700, 1800),
            _token(3, "LOT_cot", 650, 1100),
            _token(4, "GOAT_goat", 500, 1200),  # flat -> mono
            _token(5, "PRICE_buy", 700, _sweep(1000, 2400)),  # sweep -> diph
        ]
    )
    result = classify_sets(traj)
    assert result["GOAT"] is False
    assert result["PRICE"] is True
    assert result["KIT"] is False


def test_classify_keeps_borderline_at_prior() -> None:
    # A canonical diphthong whose movement sits inside the hysteresis band
    # keeps its canonical (diphthong) label.
    traj = pl.concat(
        [
            _token(0, "KIT_bit", 400, 2000),
            _token(1, "DRESS_bed", 550, 1900),
            _token(2, "TRAP_cat", 700, 1800),
            _token(3, "LOT_cot", 650, 1100),
            _token(
                4, "MOUTH_out", 600, _sweep(1500, 1750)
            ),  # mild move -> stays diph by prior
        ]
    )
    result = classify_sets(traj)
    assert result["MOUTH"] is True


def test_classify_flips_moving_canonical_mono_to_diph() -> None:
    traj = pl.concat(
        [
            _token(0, "KIT_bit", 400, 2000),
            _token(1, "DRESS_bed", 550, 1900),
            _token(2, "TRAP_cat", 700, 1800),
            _token(3, "LOT_cot", 650, 1100),
            _token(4, "FLEECE_bead", 350, _sweep(900, 2600)),  # strong sweep -> diph
        ]
    )
    result = classify_sets(traj)
    assert result["FLEECE"] is True
