# fasttrackpy Formant Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded measurement-point formant extraction with fasttrackpy trajectory tracking plus a data-driven steady-state (minimum-velocity) aggregator.

**Architecture:** The `formants` step runs `fasttrackpy.process_audio_file` per labeled vowel interval and writes the full smoothed trajectory (many rows/token) to `{session}_formants.parquet`. A new `aggregate.load_points` collapses each token's trajectory to the one-row-per-token point contract the plots already consume, choosing the frame of minimum (z-scored) formant velocity. The `nucleus` point-tier step is retired; its label knowledge moves to a new `labels.py`.

**Tech Stack:** Python 3.14, parselmouth, fasttrackpy 0.6.1, polars, numpy, typer, pytest.

## Global Constraints

- Python `>=3.14`; deps managed by `uv` (`uv run`, `uv sync`).
- polars for all dataframes (no pandas).
- Mixed-case Wells set names (`haPPY`, `coMMA`, `leTTER`) are case-sensitive — never uppercase label text.
- Diphthong point rows MUST carry a `:1`/`:2` suffix on `label` — plots detect diphthongs via `label.str.contains(":")`.
- The point contract the plots consume is exactly: columns `label`, `set`, `word`, `F0`, `F1`, `F2`, `F3`, one row per token (two for diphthongs).
- Gender ceiling search ranges: M → `min_max_formant=4500`, `max_max_formant=5500`, `window_length=0.025`, `pitch_floor=75`; F/C → `5000`/`6500`/`0.030`/`100`. `n_formants=4`.
- Run tests with `uv run pytest`. Lint with `uv run ruff check`.

---

### Task 1: Migrate label helpers into `labels.py`

Move the pure label logic out of the soon-to-be-deleted `nucleus.py` into a stable module that `formants.py` and `aggregate.py` will both import.

**Files:**
- Create: `src/vowels/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: `vowels.schema.DIPHTHONGS` (set of `Wells`), `Wells` enum (member `.name`).
- Produces:
  - `DISYLLABLE_PREFIX: str = "2"`
  - `CONSONANT_WEIGHT: float = 1.0`, `VOWEL_WEIGHT: float = 2.0`
  - `SECOND_VOWEL_CENTER_RATIO: float` (= 5/6)
  - `DIPHTHONG_NAMES: frozenset[str]` (the `.name` of every `Wells` in `DIPHTHONGS`)
  - `normalize_label(label: str) -> tuple[str, bool]` — strips a leading `"2"`, returns `(stripped, was_disyllabic)`
  - `get_set_name(label: str) -> str` — text before the first `_`, else whole label
  - `is_disyllabic(label: str) -> bool`
  - `is_diphthong_set(set_name: str) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labels.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vowels.labels'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/vowels/labels.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_labels.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/vowels/labels.py tests/test_labels.py
git commit -m "feat: add labels module with migrated label helpers"
```

---

### Task 2: Steady-state index detector

The pure numeric core of the aggregator: given smoothed F1/F2 tracks and each frame's relative time, return the index of minimum z-scored velocity within a relative-time window.

**Files:**
- Create: `src/vowels/aggregate.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: numpy.
- Produces:
  - `steady_state_index(f1: np.ndarray, f2: np.ndarray, rel_time: np.ndarray, lo: float, hi: float) -> int`
    - z-scores `f1` and `f2` independently over all frames (std 0 → treat as zeros),
    - velocity `v[i] = hypot(f1z[i]-f1z[i-1], f2z[i]-f2z[i-1])` assigned to frame `i` (first frame gets `+inf`),
    - considers only frames with `lo <= rel_time <= hi`; if the window is empty, falls back to all frames,
    - returns the absolute index (into the input arrays) of minimum velocity.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregate.py
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
    # Without normalization the F2 frame would dominate; with it, velocities tie
    # and the (earlier) flat frame is chosen. Assert the F2 jump is NOT selected.
    rel = np.linspace(0.0, 1.0, 5)
    f1 = np.array([500, 500, 500, 510, 500], float)
    f2 = np.array([1500, 1500, 1500, 1500, 2500], float)
    idx = steady_state_index(f1, f2, rel, 0.0, 1.0)
    assert idx != 4  # the big raw-Hz F2 jump must not automatically win
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vowels.aggregate'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/vowels/aggregate.py
import numpy as np


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/vowels/aggregate.py tests/test_aggregate.py
git commit -m "feat: add steady-state minimum-velocity index detector"
```

---

### Task 3: Token-trajectory collapse to point rows

Collapse one token's trajectory (many frames) into 1 point row (monophthong) or 2 point rows (diphthong), choosing steady-state frames within the right windows.

**Files:**
- Modify: `src/vowels/aggregate.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `steady_state_index` (Task 2); `vowels.labels.is_diphthong_set`, `is_disyllabic`, `get_set_name`, `normalize_label`, `SECOND_VOWEL_CENTER_RATIO` (Task 1); polars.
- Produces:
  - `collapse_token(token: pl.DataFrame, label: str) -> list[dict[str, float | str]]`
    - `token` has columns `rel_time`, `F0`, `F1_s`, `F2_s`, `F3_s` (one row per frame, ascending `rel_time`).
    - Returns dicts with keys `label`, `set`, `word`, `F0`, `F1`, `F2`, `F3`.
    - Monophthong → 1 dict, window `(0.2, 0.8)`; disyllabic → 1 dict, window `(RATIO-0.15, RATIO+0.15)`; diphthong → 2 dicts with `label` suffixed `:1` (window `0.1, 0.45`) and `:2` (window `0.55, 0.9`).
    - `set`/`word` derived from the normalized (prefix-stripped) label; `F1`/`F2`/`F3` are the smoothed values at the chosen frame; `F0` is that frame's F0, or the token mean F0 if it is NaN.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aggregate.py
import polars as pl

from vowels.aggregate import collapse_token


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregate.py -k collapse -v`
Expected: FAIL with `ImportError: cannot import name 'collapse_token'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/vowels/aggregate.py
import math

import polars as pl

from .labels import (
    SECOND_VOWEL_CENTER_RATIO,
    get_set_name,
    is_diphthong_set,
    is_disyllabic,
    normalize_label,
)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: PASS (all aggregate tests)

- [ ] **Step 5: Commit**

```bash
git add src/vowels/aggregate.py tests/test_aggregate.py
git commit -m "feat: collapse token trajectory to steady-state point rows"
```

---

### Task 4: `load_points` parquet loader

Read the trajectory parquet and return the full point-contract DataFrame the plots expect.

**Files:**
- Modify: `src/vowels/aggregate.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `collapse_token` (Task 3); `vowels.paths.session_dir`; polars.
- Produces:
  - `load_points(session: str) -> pl.DataFrame` — reads `{session}_formants.parquet`, groups by `token_id`, calls `collapse_token` per token using that token's `label` (first value), returns a DataFrame with columns `label, set, word, F0, F1, F2, F3`.
  - `points_from_trajectory(traj: pl.DataFrame) -> pl.DataFrame` — the pure inner function (takes the trajectory DataFrame, no I/O) so it can be unit-tested.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aggregate.py
from vowels.aggregate import points_from_trajectory


def test_points_from_trajectory_one_row_per_mono_two_per_diph() -> None:
    rel = list(np.linspace(0.0, 1.0, 11))
    def block(token_id, label, f1, f2):
        return pl.DataFrame({
            "token_id": [token_id] * 11,
            "label": [label] * 11,
            "rel_time": rel,
            "F0": [120.0] * 11,
            "F1_s": [float(f1)] * 11,
            "F2_s": [float(f2)] * 11,
            "F3_s": [2500.0] * 11,
        })
    traj = pl.concat([
        block(0, "TRAP_cat", 700, 1600),
        block(1, "PRICE_buy", 400, 2200),
    ])
    points = points_from_trajectory(traj)
    assert set(points.columns) == {"label", "set", "word", "F0", "F1", "F2", "F3"}
    labels = sorted(points["label"].to_list())
    assert labels == ["PRICE_buy:1", "PRICE_buy:2", "TRAP_cat"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregate.py -k points_from_trajectory -v`
Expected: FAIL with `ImportError: cannot import name 'points_from_trajectory'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/vowels/aggregate.py
from .paths import session_dir

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: PASS (all aggregate tests)

- [ ] **Step 5: Commit**

```bash
git add src/vowels/aggregate.py tests/test_aggregate.py
git commit -m "feat: add load_points trajectory aggregator"
```

---

### Task 5: Trajectory builder from a fasttrackpy winner

The pure mapping from a fasttrackpy winner DataFrame + per-frame F0 to our trajectory schema rows, isolated from audio I/O so it is unit-testable.

**Files:**
- Modify: `src/vowels/pipeline/formants.py`
- Test: `tests/test_formants.py`

**Interfaces:**
- Consumes: `vowels.labels.get_set_name`, `normalize_label`, `is_diphthong_set`, `is_disyllabic`; polars, numpy.
- Produces:
  - `winner_to_rows(winner_df: pl.DataFrame, f0: np.ndarray, token_id: int, label: str, t1: float, t2: float) -> list[dict]`
    - `winner_df` has columns `time`, `F1`, `F2`, `F3` (raw), `F1_s`, `F2_s`, `F3_s` (smoothed), `B1`, `B2`, `B3` (bandwidths), `max_formant`, `error` — one row per frame.
    - `f0` is the per-frame pitch aligned to `winner_df` rows (same length).
    - Emits one dict per frame with: `token_id`, `label`, `set`, `word`, `is_diphthong`, `is_disyllabic`, `time`, `rel_time` (= `(time - t1) / (t2 - t1)`), `F0`, `F1`, `F2`, `F3`, `F1_s`, `F2_s`, `F3_s`, `B1`, `B2`, `B3`, `max_formant`, `error`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_formants.py
import numpy as np
from vowels.pipeline.formants import winner_to_rows


def test_winner_to_rows_schema_and_rel_time() -> None:
    winner = pl.DataFrame({
        "time": [1.0, 1.1, 1.2],
        "F1": [500.0, 510.0, 505.0], "F2": [1500.0, 1490.0, 1495.0],
        "F3": [2500.0, 2510.0, 2505.0],
        "F1_s": [502.0, 508.0, 506.0], "F2_s": [1498.0, 1492.0, 1496.0],
        "F3_s": [2502.0, 2508.0, 2506.0],
        "B1": [50.0, 51.0, 52.0], "B2": [80.0, 81.0, 82.0], "B3": [120.0, 121.0, 122.0],
        "max_formant": [5000.0] * 3, "error": [0.1] * 3,
    })
    f0 = np.array([120.0, 121.0, 119.0])
    rows = winner_to_rows(winner, f0, token_id=7, label="2haPPY_coffee", t1=1.0, t2=1.2)
    assert len(rows) == 3
    assert rows[0]["token_id"] == 7
    assert rows[0]["set"] == "haPPY"
    assert rows[0]["word"] == "coffee"
    assert rows[0]["is_disyllabic"] is True
    assert rows[0]["is_diphthong"] is False
    assert rows[0]["rel_time"] == pytest.approx(0.0)
    assert rows[2]["rel_time"] == pytest.approx(1.0)
    assert rows[0]["F0"] == pytest.approx(120.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_formants.py -k winner_to_rows -v`
Expected: FAIL with `ImportError: cannot import name 'winner_to_rows'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/vowels/pipeline/formants.py (imports + function)
import numpy as np

from ..labels import (
    get_set_name,
    is_diphthong_set,
    is_disyllabic,
    normalize_label,
)


def winner_to_rows(
    winner_df: pl.DataFrame,
    f0: np.ndarray,
    token_id: int,
    label: str,
    t1: float,
    t2: float,
) -> list[dict]:
    normalized, disyll = normalize_label(label)
    set_name = get_set_name(normalized)
    word = normalized.split("_", 1)[1] if "_" in normalized else ""
    diph = is_diphthong_set(set_name)
    span = (t2 - t1) or 1.0
    rows: list[dict] = []
    for i, frame in enumerate(winner_df.iter_rows(named=True)):
        rows.append(
            {
                "token_id": token_id,
                "label": label,
                "set": set_name,
                "word": word,
                "is_diphthong": diph,
                "is_disyllabic": disyll,
                "time": frame["time"],
                "rel_time": (frame["time"] - t1) / span,
                "F0": float(f0[i]),
                "F1": frame["F1"], "F2": frame["F2"], "F3": frame["F3"],
                "F1_s": frame["F1_s"], "F2_s": frame["F2_s"], "F3_s": frame["F3_s"],
                "B1": frame["B1"], "B2": frame["B2"], "B3": frame["B3"],
                "max_formant": frame["max_formant"], "error": frame["error"],
            }
        )
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_formants.py -k winner_to_rows -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vowels/pipeline/formants.py tests/test_formants.py
git commit -m "feat: map fasttrackpy winner trajectory to schema rows"
```

---

### Task 6: Rewrite `extract_formants` to drive fasttrackpy

Replace the Burg single-point extraction with per-interval fasttrackpy tracking writing the trajectory parquet. This is I/O orchestration — verified by the end-to-end run in Task 9, not a unit test.

**Files:**
- Modify: `src/vowels/pipeline/formants.py`
- Modify: `src/vowels/schema.py` (no change expected; only read `Gender`)

**Interfaces:**
- Consumes: `winner_to_rows` (Task 5); `fasttrackpy.process_audio_file`; parselmouth; `vowels.paths.session_dir`; `vowels.schema.Gender`.
- Produces: `extract_formants(session: str, gender: Gender = Gender.M) -> None` — writes `{session}_formants.parquet` (trajectory). Signature unchanged from current export.

- [ ] **Step 1: Replace the function body**

Replace the existing `extract_formants` (the old Burg/`To Formant (burg)` implementation) with the version below. Keep `parse_labels` and `winner_to_rows` in the file.

```python
# src/vowels/pipeline/formants.py  (replace extract_formants)
from fasttrackpy import process_audio_file

_MIN_DURATION: float = 0.05


def _gender_params(gender: Gender) -> dict[str, float | int]:
    if gender == Gender.M:
        return {"min_max_formant": 4500, "max_max_formant": 5500,
                "window_length": 0.025, "pitch_floor": 75}
    return {"min_max_formant": 5000, "max_max_formant": 6500,
            "window_length": 0.030, "pitch_floor": 100}


def extract_formants(session: str, gender: Gender = Gender.M) -> None:
    d: Path = session_dir(session)
    wav_path: Path = d / f"{session}.wav"
    tg_path: Path = d / f"{session}_labeled.TextGrid"
    params = _gender_params(gender)

    tg: parselmouth.TextGrid = parselmouth.read(tg_path.as_posix())
    labeled_tier: int = 1
    n_intervals: int = parselmouth.praat.call(
        tg, "Get number of intervals", labeled_tier
    )

    rows: list[dict] = []
    token_id: int = 0
    for i in range(1, n_intervals + 1):
        label: str | None = parselmouth.praat.call(
            tg, "Get label of interval", labeled_tier, i
        )
        if not label or label.lower() == "silent":
            continue
        t1: float = parselmouth.praat.call(
            tg, "Get start time of interval", labeled_tier, i
        )
        t2: float = parselmouth.praat.call(
            tg, "Get end time of interval", labeled_tier, i
        )
        if t2 - t1 < _MIN_DURATION:
            continue

        candidates = process_audio_file(
            wav_path.as_posix(),
            xmin=t1,
            xmax=t2,
            n_formants=4,
            **params,
        )
        winner = candidates.winner
        # to_df(output="formants") already yields columns:
        # F1..F4, F1_s..F4_s, B1..B4, error, time, max_formant, n_formant,
        # smooth_method (+ metadata). winner_to_rows selects the keys it needs.
        winner_df: pl.DataFrame = winner.to_df(output="formants")
        # candidates.f0 is sampled at winner.time_domain -> 1:1 with winner_df rows.
        f0 = candidates.f0
        rows.extend(winner_to_rows(winner_df, f0, token_id, label, t1, t2))
        token_id += 1

    pl.DataFrame(rows).write_parquet(d / f"{session}_formants.parquet")
    print(f"Created {d / f'{session}_formants.parquet'} ({token_id} tokens)")
```

VERIFIED against `fasttrackpy/processors/outputs.py` (v0.6.1): `to_df()` names
bandwidths `B1/B2/B3/B4` directly (no rename needed), and `CandidateTracks.f0`
is built from `pitch_obj.get_value_at_time(t) for t in winner.time_domain`, so it
is already aligned 1:1 with `winner_df` rows. F0 values may be `NaN` on unvoiced
frames — that is expected and handled downstream in `collapse_token` (Task 3).

- [ ] **Step 2: Verify imports and lint**

Run: `uv run ruff check src/vowels/pipeline/formants.py`
Expected: no errors (remove now-unused imports such as the old pitch/formant Burg calls).

- [ ] **Step 3: Verify existing pure tests still pass**

Run: `uv run pytest tests/test_formants.py -v`
Expected: PASS (parse_labels + winner_to_rows tests).

- [ ] **Step 4: Commit**

```bash
git add src/vowels/pipeline/formants.py
git commit -m "feat: extract formant trajectories via fasttrackpy"
```

---

### Task 7: Point the plots at `load_points`

Switch the three plot loaders from reading the parquet directly (now a trajectory) to the aggregated point contract.

**Files:**
- Modify: `src/vowels/plots/vowel_space.py` (`save_chart`, ~line 437)
- Modify: `src/vowels/plots/bark_space.py` (`_load_formants`, ~line 29)

**Interfaces:**
- Consumes: `vowels.aggregate.load_points` (Task 4).
- Produces: no signature changes; the DataFrames handed to `build_chart` / `_add_bark_dims` now come from `load_points`.

- [ ] **Step 1: Edit `vowel_space.save_chart`**

Replace:

```python
    df: pl.DataFrame = pl.read_parquet(
        session_dir(session) / f"{session}_formants.parquet"
    )
```

with:

```python
    from ..aggregate import load_points

    df: pl.DataFrame = load_points(session)
```

- [ ] **Step 2: Edit `bark_space._load_formants`**

Replace:

```python
def _load_formants(session: str) -> pl.DataFrame:
    return (
        pl.read_parquet(session_dir(session) / f"{session}_formants.parquet")
        .pipe(_add_bark_dims)
        .filter(pl.col("F0").is_not_nan())
    )
```

with:

```python
def _load_formants(session: str) -> pl.DataFrame:
    from ..aggregate import load_points

    return (
        load_points(session)
        .pipe(_add_bark_dims)
        .filter(pl.col("F0").is_not_nan())
    )
```

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/vowels/plots/`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/vowels/plots/vowel_space.py src/vowels/plots/bark_space.py
git commit -m "feat: feed plots from aggregated steady-state points"
```

---

### Task 8: Retire the nucleus step and rewire CLI/exports

Delete `nucleus.py`, drop it from the pipeline and CLI, and update package exports and the stale `test_nucleus.py`.

**Files:**
- Delete: `src/vowels/pipeline/nucleus.py`
- Delete: `tests/test_nucleus.py`
- Modify: `src/vowels/pipeline/__init__.py`
- Modify: `src/vowels/__init__.py`
- Modify: `src/vowels/cli.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: nothing new.
- Produces: `vowels` package no longer exports `make_nucleus_points`, `nucleus_time`, `diphthong_times`, `disyllabic_time`; `SECOND_VOWEL_CENTER_RATIO` now sourced from `vowels.labels`. CLI `nucleus` command removed; `run` sequence is `silences → label → formants → plots`.

- [ ] **Step 1: Delete the retired files**

```bash
git rm src/vowels/pipeline/nucleus.py tests/test_nucleus.py
```

- [ ] **Step 2: Update `src/vowels/pipeline/__init__.py`**

```python
from typing import Final

from .formants import extract_formants, parse_labels, winner_to_rows
from .label import label_textgrid
from .silences import detect_silences

__all__: Final[list[str]] = [
    "extract_formants",
    "parse_labels",
    "winner_to_rows",
    "label_textgrid",
    "detect_silences",
]
```

- [ ] **Step 3: Update `src/vowels/__init__.py`**

Replace the `.pipeline` import block and `__all__` so the nucleus symbols are gone and `SECOND_VOWEL_CENTER_RATIO` comes from `.labels`:

```python
from .labels import SECOND_VOWEL_CENTER_RATIO
from .pipeline import (
    detect_silences,
    extract_formants,
    label_textgrid,
    parse_labels,
)
from .plots import precompute_ellipse, save_bark_chart, save_bark_projections, save_chart
from .schema import DIPHTHONGS, Gender, Wells
```

And set:

```python
__all__: Final[list[str]] = [
    "DIPHTHONGS",
    "Gender",
    "Wells",
    "project_root",
    "session_dir",
    "detect_silences",
    "extract_formants",
    "parse_labels",
    "label_textgrid",
    "SECOND_VOWEL_CENTER_RATIO",
    "precompute_ellipse",
    "save_chart",
    "save_bark_chart",
    "save_bark_projections",
]
```

(Keep the existing `from .paths import project_root, session_dir` line.)

- [ ] **Step 4: Update `src/vowels/cli.py`**

Remove `make_nucleus_points` from the imports. Delete the entire `nucleus` command:

```python
@app.command()
def nucleus(session: str) -> None:
    """Create nucleus point tier for formant extraction."""
    make_nucleus_points(session)
```

In `run`, delete the `make_nucleus_points(session)` line and update the docstring to `"""Run the full pipeline: silences → label → formants → plots."""`.

- [ ] **Step 5: Update `CLAUDE.md`**

In the "Individual Steps" block, delete the `nucleus` step lines:

```bash
# Step 3: Create nucleus point tier for formant extraction
uv run vowels nucleus <session>
```

Renumber the remaining steps. In the Data Flow section, delete the `**nucleus**` bullet (item 4) and fold its note into the `formants` bullet: "extracts F1/F2/F3 by tracking the full formant trajectory per labeled vowel interval (fasttrackpy, auto-selected ceiling) and collapsing each token to its steady-state (minimum-velocity) frame."

- [ ] **Step 6: Verify the suite and imports**

Run: `uv run python -c "import vowels; from vowels.cli import app"`
Expected: no ImportError.

Run: `uv run pytest -v`
Expected: PASS (test_labels, test_aggregate, test_formants, test_label_parsing, test_ellipse); no test_nucleus collected.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: retire nucleus point-tier step"
```

---

### Task 9: End-to-end verification on session2

Run the real pipeline on existing data and confirm the artifacts regenerate.

**Files:**
- None (verification only).

**Interfaces:**
- Consumes: everything above.
- Produces: regenerated `sessions/session2/session2_formants.parquet` (trajectory) and the three HTML plots.

- [ ] **Step 1: Run formants + plots on session2**

Run:
```bash
uv run vowels formants session2 --gender M
uv run vowels plot session2
uv run vowels bark session2
uv run vowels projections session2
```
Expected: each prints its "Created …" line with no traceback; `formants` reports a token count in the same ballpark as the number of non-silent labels.

- [ ] **Step 2: Sanity-check the parquet shape**

Run:
```bash
uv run python -c "import polars as pl; df=pl.read_parquet('sessions/session2/session2_formants.parquet'); print(df.columns); print('rows', df.height, 'tokens', df['token_id'].n_unique()); print(df.select(['F1_s','F2_s','F3_s','max_formant']).describe())"
```
Expected: columns include `token_id, label, set, word, is_diphthong, is_disyllabic, time, rel_time, F0, F1..F3, F1_s..F3_s, B1..B3, max_formant, error`; many rows per token; `max_formant` values vary across tokens (confirming per-token ceiling selection); F1/F2/F3 in plausible Hz ranges.

- [ ] **Step 3: Sanity-check the aggregated points**

Run:
```bash
uv run python -c "from vowels.aggregate import load_points; p=load_points('session2'); print(p.columns); print('points', p.height); print(p.filter(p['label'].str.contains(':')).select('label').unique().sort('label'))"
```
Expected: columns exactly `label, set, word, F0, F1, F2, F3`; one row per monophthong token, two (`:1`/`:2`) per diphthong token; F1/F2/F3 in plausible ranges.

- [ ] **Step 4: Open the plots (optional manual check)**

Open `sessions/session2/session2_vowel_space.html` and confirm the vowel space looks sane (monophthong clusters separated, diphthong arrows present). Compare against the pre-change plot if one was kept.

- [ ] **Step 5: Final lint + commit**

Run: `uv run ruff check`
Expected: no errors.

```bash
git add -A
git commit -m "test: verify fasttrackpy pipeline end-to-end on session2"
```

---

## Self-Review

**Spec coverage:**
- §1 pipeline shape / nucleus retired → Task 8.
- §2 formants rewrite + trajectory schema → Tasks 5 (mapping) + 6 (orchestration).
- §3 gender ceiling ranges → Task 6 `_gender_params` + Global Constraints.
- §4 steady-state aggregation + plot rewiring → Tasks 2, 3, 4 (aggregate) + 7 (plots).
- §5 tests → Tasks 1–5 carry their own tests; Task 8 removes `test_nucleus.py`; Task 9 is end-to-end.
- Label-knowledge migration (DIPHTHONGS membership, disyllabic prefix, ratio) → Task 1.

**Placeholder scan:** None. The one fasttrackpy-internal uncertainty (bandwidth column names, `f0` alignment) was resolved against `fasttrackpy/processors/outputs.py` and `tracks.py` v0.6.1 and baked into Task 6 as verified fact. All steps contain complete code.

**Type consistency:** `steady_state_index`, `collapse_token`, `points_from_trajectory`, `load_points`, `winner_to_rows`, `_gender_params` names match across tasks. Point contract columns `label, set, word, F0, F1, F2, F3` are identical in Tasks 3, 4, 7 and the Global Constraints. `SECOND_VOWEL_CENTER_RATIO` is defined once (Task 1) and consumed in Task 3.
