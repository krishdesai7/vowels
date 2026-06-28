import numpy as np

from vowels.aggregate import steady_state_index


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
