import math

import numpy as np
import polars as pl

from .labels import (
    SECOND_VOWEL_CENTER_RATIO,
    get_set_name,
    is_diphthong_set,
    is_disyllabic,
    normalize_label,
)
from .paths import session_dir


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def steady_state_index(
    f1: np.ndarray,
    f2: np.ndarray,
    rel_time: np.ndarray,
    lo: float,
    hi: float,
) -> int:
    f1z = _zscore(f1)
    f2z = _zscore(f2)
    velocity = np.full(f1.shape, np.inf)
    velocity[1:] = np.hypot(np.diff(f1z), np.diff(f2z))

    in_window = (rel_time >= lo) & (rel_time <= hi)
    if not in_window.any():
        in_window = np.ones(f1.shape, dtype=bool)

    masked = np.where(in_window, velocity, np.inf)
    return int(np.argmin(masked))


_DISYLLABIC_HALF_WINDOW = 0.15


def _point(
    token: pl.DataFrame, label: str, set_name: str, word: str, lo: float, hi: float
) -> dict[str, float | str]:
    rel = token["rel_time"].to_numpy()
    f1 = token["F1_s"].to_numpy()
    f2 = token["F2_s"].to_numpy()
    f3 = token["F3_s"].to_numpy()
    f0 = token["F0"].to_numpy()
    idx = steady_state_index(f1, f2, rel, lo, hi)
    chosen_f0 = float(f0[idx])
    if math.isnan(chosen_f0):
        finite = f0[~np.isnan(f0)]
        chosen_f0 = float(finite.mean()) if finite.size else float("nan")
    return {
        "label": label,
        "set": set_name,
        "word": word,
        "F0": chosen_f0,
        "F1": float(f1[idx]),
        "F2": float(f2[idx]),
        "F3": float(f3[idx]),
    }


def collapse_token(token: pl.DataFrame, label: str) -> list[dict[str, float | str]]:
    token = token.sort("rel_time")
    normalized, _ = normalize_label(label)
    set_name = get_set_name(normalized)
    word = normalized.split("_", 1)[1] if "_" in normalized else ""

    if is_diphthong_set(set_name):
        return [
            _point(token, f"{label}:1", set_name, word, 0.1, 0.45),
            _point(token, f"{label}:2", set_name, word, 0.55, 0.9),
        ]
    if is_disyllabic(label):
        lo = SECOND_VOWEL_CENTER_RATIO - _DISYLLABIC_HALF_WINDOW
        hi = SECOND_VOWEL_CENTER_RATIO + _DISYLLABIC_HALF_WINDOW
        return [_point(token, label, set_name, word, lo, hi)]
    return [_point(token, label, set_name, word, 0.2, 0.8)]


_POINT_COLUMNS = ["label", "set", "word", "F0", "F1", "F2", "F3"]


def points_from_trajectory(traj: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (_token_id,), token in traj.group_by("token_id", maintain_order=True):
        label = token["label"][0]
        rows.extend(collapse_token(token, label))
    return pl.DataFrame(rows, schema={c: (pl.Utf8 if c in ("label", "set", "word") else pl.Float64) for c in _POINT_COLUMNS})


def load_points(session: str) -> pl.DataFrame:
    traj = pl.read_parquet(session_dir(session) / f"{session}_formants.parquet")
    return points_from_trajectory(traj)
