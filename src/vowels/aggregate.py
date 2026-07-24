import math
from typing import Final, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .bark import add_bark_dims
from .labels import (
    SECOND_VOWEL_CENTER_RATIO,
    get_set_name,
    is_disyllabic,
    normalize_label,
)
from .paths import session_dir
from .schema import Dialect, DialectProfile


def _zscore[T: np.number](x: NDArray[T]) -> NDArray[np.double]:
    std = x.std()
    if std == 0:
        return np.zeros_like(x, dtype=np.double)
    return (x - x.mean()) / std


_BARK_DIMS: tuple[str, str, str] = ("Openness", "Frontness", "Roundness")

# Measurement windows are confined to the middle 50% of the interval, because a
# labeled interval spans the whole word rather than the vowel alone. Adjacent
# voiced sonorants (nasals, /w/, /l/, /r/) are tracked as ordinary formant
# frames, and -- being consonants with *steady* formants -- they masquerade as
# vowel steady states, so the minimum-velocity criterion does not screen them
# out. Keeping the search away from the interval edges is the defence. Voiceless
# obstruents need no such handling: unvoiced frames are simply not tracked.
_MONO_WINDOW: tuple[float, float] = (0.25, 0.75)
_ONSET_WINDOW: tuple[float, float] = (0.25, 0.45)
_OFFSET_WINDOW: tuple[float, float] = (0.55, 0.75)

# Tukey's far-outlier distance. A canonically monophthongal set must clear it
# to be called a diphthong, so the bar always sits outside the baseline pool it
# is derived from -- the flaw in an earlier centre + k*MAD rule, where a tightly
# clustered pool drove the bar down among its own members.
_FAR_FENCE: Final[float] = 3.0
MIN_SPREAD: Final[float] = 0.1  # Bark; guards a degenerate (zero-IQR) baseline


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


def _with_bark(token: pl.LazyFrame) -> pl.LazyFrame | None:
    """Bark dims on the smoothed formant track; None if the token has no F0.

    Z0 is held at the token's mean finite F0 across every frame rather than
    tracking F0 frame by frame. In Syrdal & Gopal's Bark-difference metric F0
    is a speaker-size normalizer, not a per-frame signal: since
    ``Openness = Z1 - Z0``, a per-frame Z0 lets intonation and declination
    masquerade as articulatory movement -- and frame-to-frame movement is
    exactly what steady-state selection minimizes. Anchoring also subsumes the
    NaN-F0 fill, since unvoiced frames inherit the same reference.
    """
    f0: NDArray[np.double] = token.select("F0").collect().to_numpy()
    finite: NDArray[np.double] = f0[~np.isnan(f0)]
    if finite.size == 0:
        return None
    anchored: pl.LazyFrame = token.with_columns(
        pl.lit(float(finite.mean())).alias("F0")
    )
    return add_bark_dims(anchored, formant_cols=("F1_s", "F2_s", "F3_s"))


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


def token_displacement(token: pl.LazyFrame) -> float:
    """Onset->offset spectral displacement in Bark for one token.

    Returns NaN when the token has no finite F0 (cannot be Bark-normalized).
    """
    token = token.sort("rel_time")
    barked: pl.LazyFrame | None = _with_bark(token)
    if barked is None:
        return float("nan")
    rel: NDArray[np.double] = token.select("rel_time").collect().to_numpy()
    dims: NDArray[np.double] = np.column_stack(
        [barked.select(d).collect().to_numpy() for d in _BARK_DIMS]
    )
    onset: int = steady_state_index(dims, rel, *_ONSET_WINDOW)
    offset: int = steady_state_index(dims, rel, *_OFFSET_WINDOW)
    return float(np.linalg.norm(dims[offset] - dims[onset]))


_DISYLLABIC_HALF_WINDOW = 0.15


def _point(
    token: pl.LazyFrame, label: str, set_name: str, word: str, lo: float, hi: float
) -> dict[str, np.double | str]:
    actual: pl.DataFrame = token.collect()
    rel: NDArray[np.double] = actual.select("rel_time").to_numpy()
    f1: NDArray[np.double] = actual.select("F1_s").to_numpy()
    f2: NDArray[np.double] = actual.select("F2_s").to_numpy()
    f3: NDArray[np.double] = actual.select("F3_s").to_numpy()
    f0: NDArray[np.double] = actual.select("F0").to_numpy()
    barked: pl.LazyFrame | None = _with_bark(token)
    if barked is not None:
        dims: NDArray[np.double] = np.column_stack(
            [barked.select(d).collect().to_numpy() for d in _BARK_DIMS]
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


def collapse_token(
    token: pl.LazyFrame, label: str, is_diphthong: bool
) -> list[dict[str, np.double | str]]:
    token = token.sort("rel_time")
    normalized, _ = normalize_label(label)
    set_name: str = get_set_name(normalized)
    word: str = normalized.split("_", 1)[1] if "_" in normalized else ""

    if is_diphthong:
        return [
            _point(token, f"{label}:1", set_name, word, *_ONSET_WINDOW),
            _point(token, f"{label}:2", set_name, word, *_OFFSET_WINDOW),
        ]
    if is_disyllabic(label):
        lo: float = SECOND_VOWEL_CENTER_RATIO - _DISYLLABIC_HALF_WINDOW
        hi: float = SECOND_VOWEL_CENTER_RATIO + _DISYLLABIC_HALF_WINDOW
        return [_point(token, label, set_name, word, lo, hi)]
    return [_point(token, label, set_name, word, *_MONO_WINDOW)]


_POINT_COLUMNS: list[str] = ["label", "set", "word", "F0", "F1", "F2", "F3"]


def points_from_trajectory(traj: pl.LazyFrame, dialect: Dialect) -> pl.LazyFrame:
    classification: dict[str, bool] = classify_sets(traj, dialect)
    rows: list[dict[str, np.double | str]] = []
    for (_token_id,), token in traj.collect().group_by("token_id", maintain_order=True):
        label: str = token["label"][0]
        normalized, _ = normalize_label(label)
        set_name: str = get_set_name(normalized)
        # Sets whose tokens were all unscorable fall back to the canonical prior.
        is_diph: bool = classification.get(
            set_name, set_name in dialect.profile.diphthongs
        )
        rows.extend(collapse_token(pl.LazyFrame(token), label, is_diph))
    return pl.LazyFrame(
        rows,
        schema={
            c: (pl.Utf8 if c in ("label", "set", "word") else pl.Float64)
            for c in _POINT_COLUMNS
        },
    )


def load_points(session: str, dialect: Dialect) -> pl.LazyFrame:
    traj: pl.LazyFrame = pl.scan_parquet(
        session_dir(session) / f"{session}_formants.parquet"
    )
    return points_from_trajectory(traj, dialect)


def _score_sets(
    traj: pl.LazyFrame,
) -> tuple[dict[str, float], dict[str, int], dict[str, bool]]:
    scores: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    disyll: dict[str, bool] = {}
    for (_token_id,), token in traj.collect().group_by("token_id", maintain_order=True):
        label: str = token["label"][0]
        normalized, is_dis = normalize_label(label)
        set_name: str = get_set_name(normalized)
        counts[set_name] = counts.get(set_name, 0) + 1
        disyll[set_name] = disyll.get(set_name, False) or is_dis
        disp: float = token_displacement(pl.LazyFrame(token))
        if not math.isnan(disp):
            scores.setdefault(set_name, []).append(disp)
    set_score: dict[str, float] = {
        s: float(np.median(v)) for s, v in scores.items() if v
    }
    return set_score, counts, disyll


def _baseline(
    set_score: dict[str, float],
    disyll: dict[str, bool],
    profile: DialectProfile,
) -> tuple[float, float]:
    """Q3 and IQR of this speaker's plain-monophthong displacements.

    The pool is what a monophthong looks like *for this speaker*, so it admits
    only sets that should not glide at all: the dialect's own non-diphthongs,
    minus the disyllabics (measured on their second syllable, so not
    comparable) and minus the r-coloured sets, whose falling F3 registers as
    large spectral displacement without any gliding nucleus.
    """
    mono: list[float] = [
        sc
        for s, sc in set_score.items()
        if s not in profile.diphthongs
        and s not in profile.r_colored
        and not disyll.get(s, False)
    ]
    if not mono:
        return 0.0, MIN_SPREAD
    arr: NDArray[np.double] = np.array(mono)
    q1, q3 = (float(x) for x in np.percentile(arr, [25, 75]))
    return q3, max(q3 - q1, MIN_SPREAD)


def _decide(
    set_name: str,
    score: float,
    q3: float,
    iqr: float,
    disyll: dict[str, bool],
    profile: DialectProfile,
) -> bool:
    """Mono/diphthong for one set, with the dialect's expectation as the prior.

    The margins are deliberately asymmetric and both are anchored outside the
    bulk of the baseline. Calling a canonically monophthongal set a diphthong
    takes Tukey's far-outlier distance (Q3 + 3*IQR) -- it must be unlike this
    speaker's monophthongs by a wide margin. Demoting a canonical diphthong
    only requires it to sit inside the monophthong body (below Q3). Anything
    between keeps the dialect's expectation, which is where genuinely
    borderline sets belong.
    """
    if disyll.get(set_name, False):
        return False
    if score > q3 + _FAR_FENCE * iqr:
        return True
    if score <= q3:
        return False
    return set_name in profile.diphthongs


def classify_sets(traj: pl.LazyFrame, dialect: Dialect) -> dict[str, bool]:
    profile: DialectProfile = dialect.profile
    set_score, _counts, disyll = _score_sets(traj)
    q3, iqr = _baseline(set_score, disyll, profile)
    return {s: _decide(s, sc, q3, iqr, disyll, profile) for s, sc in set_score.items()}


def baseline_bars(session: str, dialect: Dialect) -> tuple[float, float, float, float]:
    """The session's monophthong baseline and the two decision bars.

    Returns (q3, iqr, mono_bar, diph_bar), all in Bark. A set scoring below
    mono_bar is a monophthong, above diph_bar a diphthong; in between the
    dialect's expectation wins.
    """
    traj: pl.LazyFrame = pl.scan_parquet(
        session_dir(session) / f"{session}_formants.parquet"
    )
    set_score, _counts, disyll = _score_sets(traj)
    q3, iqr = _baseline(set_score, disyll, dialect.profile)
    return q3, iqr, q3, q3 + _FAR_FENCE * iqr


def diphthong_report(session: str, dialect: Dialect) -> pl.DataFrame:
    profile: DialectProfile = dialect.profile
    traj: pl.LazyFrame = pl.scan_parquet(
        session_dir(session) / f"{session}_formants.parquet"
    )
    set_score, counts, disyll = _score_sets(traj)
    q3, iqr = _baseline(set_score, disyll, profile)
    rows: list[dict[str, object]] = []
    for set_name in sorted(set_score):
        canonical_diph: bool = set_name in profile.diphthongs
        final_diph: bool = _decide(
            set_name, set_score[set_name], q3, iqr, disyll, profile
        )
        rows.append(
            {
                "set": set_name,
                "n": counts.get(set_name, 0),
                "score": round(set_score[set_name], 3),
                "canonical": "diphthong" if canonical_diph else "monophthong",
                "final": "diphthong" if final_diph else "monophthong",
                "flipped": canonical_diph != final_diph,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "set": pl.Utf8,
            "n": pl.Int64,
            "score": pl.Float64,
            "canonical": pl.Utf8,
            "final": pl.Utf8,
            "flipped": pl.Boolean,
        },
    )
