import math
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .bark import add_bark_dims
from .labels import (
    SECOND_VOWEL_CENTER_RATIO,
    get_set_name,
    is_diphthong_set,
    is_disyllabic,
    normalize_label,
)
from .paths import session_dir


def _zscore[T: np.number](x: NDArray[T]) -> NDArray[np.double]:
    std = x.std()
    if std == 0:
        return np.zeros_like(x, dtype=np.double)
    return (x - x.mean()) / std


_BARK_DIMS: tuple[str, str, str] = ("Openness", "Frontness", "Roundness")
_ONSET_WINDOW: tuple[float, float] = (0.1, 0.45)
_OFFSET_WINDOW: tuple[float, float] = (0.55, 0.9)


def steady_state_index(
    dims: NDArray[np.double],
    rel_time: NDArray[np.double],
    lo: float,
    hi: float,
) -> int:
    n: int = dims.shape[0]
    velocity: NDArray[np.double] = np.full(n, np.inf)
    if n > 1:
        velocity[1:] = np.linalg.norm(np.diff(dims, axis=0), axis=1)
    in_window: NDArray[np.bool] = (rel_time >= lo) & (rel_time <= hi)
    if not in_window.any():
        in_window = np.full(n, True, dtype=np.bool)
    masked: NDArray[np.double] = np.where(in_window, velocity, np.inf)
    return int(np.argmin(masked))


def _with_bark(token: pl.DataFrame) -> pl.DataFrame | None:
    """Bark dims on the smoothed formant track; None if the token has no F0.

    NaN-F0 frames are filled with the token's mean finite F0 so Bark's F0
    reference (Z0) is defined per frame.
    """
    f0: NDArray[np.double] = token["F0"].to_numpy()
    finite: NDArray[np.double] = f0[~np.isnan(f0)]
    if finite.size == 0:
        return None
    filled: pl.DataFrame = token.with_columns(pl.col("F0").fill_nan(float(finite.mean())))
    return add_bark_dims(filled, formant_cols=("F1_s", "F2_s", "F3_s"))


def _zscore_velocity_index(
    f1: NDArray[np.double],
    f2: NDArray[np.double],
    rel_time: NDArray[np.double],
    lo: float,
    hi: float,
) -> int:
    """Fallback for tokens with no F0: z-scored-Hz velocity (pre-Bark method)."""
    dims: NDArray[np.double] = np.column_stack([_zscore(f1), _zscore(f2)])
    return steady_state_index(dims, rel_time, lo, hi)


_DISYLLABIC_HALF_WINDOW = 0.15


def _point(
    token: pl.DataFrame, label: str, set_name: str, word: str, lo: float, hi: float
) -> dict[str, np.double | str]:
    rel: NDArray[np.double] = token["rel_time"].to_numpy()
    f1: NDArray[np.double] = token["F1_s"].to_numpy()
    f2: NDArray[np.double] = token["F2_s"].to_numpy()
    f3: NDArray[np.double] = token["F3_s"].to_numpy()
    f0: NDArray[np.double] = token["F0"].to_numpy()
    barked: pl.DataFrame | None = _with_bark(token)
    if barked is not None:
        dims: NDArray[np.double] = np.column_stack(
            [barked[d].to_numpy() for d in _BARK_DIMS]
        )
        idx: int = steady_state_index(dims, rel, lo, hi)
    else:
        idx = _zscore_velocity_index(f1, f2, rel, lo, hi)
    chosen_f0: np.double = f0[idx]
    if math.isnan(chosen_f0):
        remaining: NDArray[np.double] = f0[~np.isnan(f0)]
        chosen_f0 = cast(np.double, remaining.mean() if remaining.size else np.nan)
    return {
        "label": label,
        "set": set_name,
        "word": word,
        "F0": chosen_f0,
        "F1": f1[idx],
        "F2": f2[idx],
        "F3": f3[idx],
    }


def collapse_token(token: pl.DataFrame, label: str) -> list[dict[str, np.double | str]]:
    token = token.sort("rel_time")
    normalized, _ = normalize_label(label)
    set_name: str = get_set_name(normalized)
    word: str = normalized.split("_", 1)[1] if "_" in normalized else ""

    # Diphthong routing takes precedence over disyllabic: the corpus has no
    # diphthong sets that are also 2-prefixed, so checking diphthong first is safe.
    if is_diphthong_set(set_name):
        return [
            _point(token, f"{label}:1", set_name, word, 0.1, 0.45),
            _point(token, f"{label}:2", set_name, word, 0.55, 0.9),
        ]
    if is_disyllabic(label):
        lo: float = SECOND_VOWEL_CENTER_RATIO - _DISYLLABIC_HALF_WINDOW
        hi: float = SECOND_VOWEL_CENTER_RATIO + _DISYLLABIC_HALF_WINDOW
        return [_point(token, label, set_name, word, lo, hi)]
    return [_point(token, label, set_name, word, 0.2, 0.8)]


_POINT_COLUMNS = ["label", "set", "word", "F0", "F1", "F2", "F3"]


def points_from_trajectory(traj: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, np.double | str]] = []
    for (_token_id,), token in traj.group_by("token_id", maintain_order=True):
        label: str = token["label"][0]
        rows.extend(collapse_token(token, label))
    return pl.DataFrame(
        rows,
        schema={
            c: (pl.Utf8 if c in ("label", "set", "word") else pl.Float64)
            for c in _POINT_COLUMNS
        },
    )


def load_points(session: str) -> pl.DataFrame:
    traj: pl.DataFrame = pl.read_parquet(
        session_dir(session) / f"{session}_formants.parquet"
    )
    return points_from_trajectory(traj)
