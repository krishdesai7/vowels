# Per-speaker Diphthong Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide per speaker, from the trajectory data, whether each Wells set is realized as a monophthong or diphthong, instead of the hardcoded `schema.DIPHTHONGS` list.

**Architecture:** The canonical `DIPHTHONGS` list becomes a *prior*. For each token we measure onset→offset spectral displacement in speaker-normalized Bark space `(Openness, Frontness, Roundness)`; the per-set median is compared against a baseline built from this speaker's canonical-monophthong sets, and the canonical label is overturned only when the score clears an asymmetric margin. All formant work happens in Bark.

**Tech Stack:** Python 3.13, polars (no pandas), numpy, fasttrackpy, parselmouth, typer, pytest, ruff, uv.

## Global Constraints

- Work in **Bark** space, never raw Hz, for every metric/distance/threshold (project principle).
- Bark transform: `Z_i = 26.81·F_i/(1960+F_i) − 0.53`; `Openness=Z1−Z0`, `Frontness=Z2−Z1`, `Roundness=Z3−Z2` (Z0 from F0).
- Onset window `(0.1, 0.45)`, offset window `(0.55, 0.9)` — the same windows the diphthong branch already uses for `:1`/`:2`.
- Prior source: a set is a diphthong iff its name is in `labels.DIPHTHONG_NAMES` (derived from `schema.DIPHTHONGS`).
- Disyllabic sets (`2`-prefix) are always monophthongs and are excluded from the baseline and from flipping.
- polars only; use `math`/`numpy` for scalars; run `uv run pytest` and `uv run ruff check` — both must be clean.
- Every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Stage ONLY the paths listed in each task's commit step. Never `git add -A`; never stage anything under `sessions/`.

---

### Task 1: Shared `bark.py` transform

Move the Bark helpers out of the plots layer into a shared module so aggregation and plotting share one transform, and generalize it to accept smoothed formant columns.

**Files:**
- Create: `src/vowels/bark.py`
- Modify: `src/vowels/plots/bark_space.py:16-32` (remove `_bark`/`_add_bark_dims`, import from `..bark`)
- Test: `tests/test_bark.py`

**Interfaces:**
- Produces: `add_bark_dims(df: pl.DataFrame, *, f0_col: str = "F0", formant_cols: tuple[str, str, str] = ("F1", "F2", "F3")) -> pl.DataFrame` — adds `Z0..Z3`, `Openness`, `Frontness`, `Roundness`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bark.py
import polars as pl
import pytest

from vowels.bark import add_bark_dims


def _z(f: float) -> float:
    return (26.81 * f) / (1960 + f) - 0.53


def test_add_bark_dims_default_columns() -> None:
    df = pl.DataFrame({"F0": [120.0], "F1": [500.0], "F2": [1500.0], "F3": [2500.0]})
    out = add_bark_dims(df)
    z0, z1, z2, z3 = _z(120.0), _z(500.0), _z(1500.0), _z(2500.0)
    assert out["Openness"][0] == pytest.approx(z1 - z0)
    assert out["Frontness"][0] == pytest.approx(z2 - z1)
    assert out["Roundness"][0] == pytest.approx(z3 - z2)


def test_add_bark_dims_accepts_smoothed_columns() -> None:
    df = pl.DataFrame({"F0": [120.0], "F1_s": [500.0], "F2_s": [1500.0], "F3_s": [2500.0]})
    out = add_bark_dims(df, formant_cols=("F1_s", "F2_s", "F3_s"))
    assert out["Frontness"][0] == pytest.approx(_z(1500.0) - _z(500.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vowels.bark'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/vowels/bark.py
import polars as pl


def _z_expr(col: str, i: int) -> pl.Expr:
    c: pl.Expr = pl.col(col)
    return ((26.81 * c) / (1960 + c) - 0.53).alias(f"Z{i}")


def add_bark_dims(
    df: pl.DataFrame,
    *,
    f0_col: str = "F0",
    formant_cols: tuple[str, str, str] = ("F1", "F2", "F3"),
) -> pl.DataFrame:
    cols: tuple[str, ...] = (f0_col, *formant_cols)
    return df.with_columns(
        _z_expr(c, i) for i, c in enumerate(cols)
    ).with_columns(
        (pl.col("Z1") - pl.col("Z0")).alias("Openness"),
        (pl.col("Z2") - pl.col("Z1")).alias("Frontness"),
        (pl.col("Z3") - pl.col("Z2")).alias("Roundness"),
    )
```

- [ ] **Step 4: Point `bark_space.py` at the shared module**

In `src/vowels/plots/bark_space.py`, delete the `_bark` function (lines 16-18) and the `_add_bark_dims` function (lines 21-26). Add to the imports near the top (after `from ..paths import ...`):

```python
from ..bark import add_bark_dims
```

Change `_load_formants` (line 29-32) so the pipe uses the imported name:

```python
def _load_formants(session: str) -> pl.DataFrame:
    from ..aggregate import load_points

    return load_points(session).pipe(add_bark_dims).filter(pl.col("F0").is_not_nan())
```

- [ ] **Step 5: Run tests + ruff to verify green**

Run: `uv run pytest tests/test_bark.py tests/ -q && uv run ruff check src/ tests/`
Expected: all pass, ruff clean. (The full suite confirms `bark_space` still imports.)

- [ ] **Step 6: Commit**

```bash
git add src/vowels/bark.py src/vowels/plots/bark_space.py tests/test_bark.py
git commit -m "refactor: extract shared Bark transform into vowels.bark"
```

---

### Task 2: Steady-state selection in Bark

Rewrite `steady_state_index` to take a generic Bark-dims matrix and compute velocity as the Euclidean norm of frame-to-frame movement (no z-scoring). Rewrite `_point` to select the frame using Bark velocity, with a z-scored-Hz fallback for the rare token that has no F0 at all.

**Files:**
- Modify: `src/vowels/aggregate.py` (`steady_state_index`, `_point`; add `_with_bark`, `_zscore_velocity_index`)
- Test: `tests/test_aggregate.py` (rewrite the four `steady_state_index` tests)

**Interfaces:**
- Consumes: `add_bark_dims` (Task 1).
- Produces:
  - `steady_state_index(dims: NDArray[np.double], rel_time: NDArray[np.double], lo: float, hi: float) -> int`
  - `_with_bark(token: pl.DataFrame) -> pl.DataFrame | None` (Bark dims on smoothed formants; `None` if no finite F0)
  - `_BARK_DIMS: tuple[str, str, str] = ("Openness", "Frontness", "Roundness")`
  - `_ONSET_WINDOW = (0.1, 0.45)`, `_OFFSET_WINDOW = (0.55, 0.9)`

- [ ] **Step 1: Rewrite the failing tests**

Replace the four tests `test_picks_flat_region_in_center`, `test_window_restricts_search`, `test_normalization_balances_f1_f2`, `test_empty_window_falls_back_to_all_frames` (lines 8-47) with:

```python
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
    # Movement only on the second axis (e.g. Frontness) must still register.
    rel = np.linspace(0.0, 1.0, 5)
    axis0 = np.zeros(5)
    axis1 = np.array([0.0, 0.0, 0.0, 0.0, 5.0])  # jump on the last frame
    dims = np.column_stack([axis0, axis1])
    idx = steady_state_index(dims, rel, 0.0, 1.0)
    assert idx != 4  # the moving frame is not the steady state


def test_empty_window_falls_back_to_all_frames() -> None:
    rel = np.linspace(0.0, 1.0, 5)
    d = np.array([500.0, 500.0, 490.0, 500.0, 500.0], float)
    idx = steady_state_index(d.reshape(-1, 1), rel, 1.5, 2.0)
    assert 0 <= idx <= 4
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_aggregate.py -k "steady or velocity or window or flat" -v`
Expected: FAIL — `steady_state_index` still takes `(f1, f2, rel_time, lo, hi)`, so passing a 2-D `dims` positional mismatches / raises.

- [ ] **Step 3: Rewrite `steady_state_index` and add helpers**

In `src/vowels/aggregate.py`, replace the current `steady_state_index` (lines 25-42) with the version below and add the two helpers + constants. Keep `_zscore` (lines 18-22) — it is reused by the fallback.

```python
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
```

Add the import at the top of `aggregate.py` (with the other `from .` imports):

```python
from .bark import add_bark_dims
```

- [ ] **Step 4: Rewrite `_point` to select in Bark**

Replace `_point` (lines 48-69) with:

```python
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
```

- [ ] **Step 5: Run tests + ruff**

Run: `uv run pytest tests/test_aggregate.py -q && uv run ruff check src/ tests/`
Expected: all pass (the `collapse_token`/`points_from_trajectory` tests still call the old `collapse_token(df, label)` signature and stay green — they exercise `_point` through it), ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/vowels/aggregate.py tests/test_aggregate.py
git commit -m "feat: select vowel steady-state frame in Bark space"
```

---

### Task 3: Onset→offset displacement metric

Add the per-token diphthongness score: the Bark distance between the onset and offset steady-state targets.

**Files:**
- Modify: `src/vowels/aggregate.py` (add `token_displacement`)
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `_with_bark`, `steady_state_index`, `_BARK_DIMS`, `_ONSET_WINDOW`, `_OFFSET_WINDOW` (Task 2).
- Produces: `token_displacement(token: pl.DataFrame) -> float` (NaN if the token has no finite F0).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_aggregate.py` (the `_frames` helper already exists at line 50):

```python
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
```

Add `import math` at the top of the test file if not present (it is not currently imported).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_aggregate.py -k displacement -v`
Expected: FAIL — `token_displacement` is not defined.

- [ ] **Step 3: Implement `token_displacement`**

Add to `src/vowels/aggregate.py` (after `_zscore_velocity_index`):

```python
def token_displacement(token: pl.DataFrame) -> float:
    """Onset->offset spectral displacement in Bark for one token.

    Returns NaN when the token has no finite F0 (cannot be Bark-normalized).
    """
    token = token.sort("rel_time")
    barked: pl.DataFrame | None = _with_bark(token)
    if barked is None:
        return float("nan")
    rel: NDArray[np.double] = token["rel_time"].to_numpy()
    dims: NDArray[np.double] = np.column_stack([barked[d].to_numpy() for d in _BARK_DIMS])
    onset: int = steady_state_index(dims, rel, *_ONSET_WINDOW)
    offset: int = steady_state_index(dims, rel, *_OFFSET_WINDOW)
    return float(np.linalg.norm(dims[offset] - dims[onset]))
```

- [ ] **Step 4: Run tests + ruff**

Run: `uv run pytest tests/test_aggregate.py -k displacement -q && uv run ruff check src/`
Expected: PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/vowels/aggregate.py tests/test_aggregate.py
git commit -m "feat: add onset-offset Bark displacement metric"
```

---

### Task 4: Speaker-relative classifier

Aggregate per-token displacements to per-set medians, build the monophthong baseline, and apply the asymmetric-margin decision with the canonical prior.

**Files:**
- Modify: `src/vowels/aggregate.py` (add constants + `_score_sets`, `_baseline`, `_decide`, `classify_sets`; import `DIPHTHONG_NAMES`)
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `token_displacement` (Task 3); `normalize_label`, `get_set_name`, `DIPHTHONG_NAMES` from `.labels`.
- Produces:
  - `classify_sets(traj: pl.DataFrame) -> dict[str, bool]` (True = diphthong)
  - `_score_sets(traj) -> tuple[dict[str, float], dict[str, int], dict[str, bool]]` → (per-set median score, per-set token count, per-set is-disyllabic)
  - `_baseline(set_score, disyll) -> tuple[float, float]` → (center, spread)
  - `K_LOW = 2.0`, `K_HIGH = 4.0`, `MIN_SPREAD = 0.1`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_aggregate.py`:

```python
def _token(token_id, label, f1, f2, n=11):
    rel = list(np.linspace(0.0, 1.0, n))
    return pl.DataFrame({
        "token_id": [token_id] * n,
        "label": [label] * n,
        "rel_time": rel,
        "F0": [120.0] * n,
        "F1_s": [float(x) for x in f1] if hasattr(f1, "__len__") else [float(f1)] * n,
        "F2_s": [float(x) for x in f2] if hasattr(f2, "__len__") else [float(f2)] * n,
        "F3_s": [2500.0] * n,
    })


def _sweep(a, b, n=11):
    return list(np.linspace(a, b, n))


def test_classify_flips_flat_canonical_diphthong_to_mono() -> None:
    # Baseline: several flat canonical-monophthong sets. A canonical diphthong
    # (GOAT) that is also flat must flip to monophthong; a canonical diphthong
    # that sweeps (PRICE) must stay a diphthong.
    traj = pl.concat([
        _token(0, "KIT_bit", 400, 2000),
        _token(1, "DRESS_bed", 550, 1900),
        _token(2, "TRAP_cat", 700, 1800),
        _token(3, "LOT_cot", 650, 1100),
        _token(4, "GOAT_goat", 500, 1200),               # flat -> mono
        _token(5, "PRICE_buy", 700, _sweep(1000, 2400)),  # sweep -> diph
    ])
    result = classify_sets(traj)
    assert result["GOAT"] is False
    assert result["PRICE"] is True
    assert result["KIT"] is False


def test_classify_keeps_borderline_at_prior() -> None:
    # A canonical diphthong whose movement sits inside the hysteresis band
    # keeps its canonical (diphthong) label.
    traj = pl.concat([
        _token(0, "KIT_bit", 400, 2000),
        _token(1, "DRESS_bed", 550, 1900),
        _token(2, "TRAP_cat", 700, 1800),
        _token(3, "LOT_cot", 650, 1100),
        _token(4, "MOUTH_out", 600, _sweep(1500, 1750)),  # mild move -> stays diph by prior
    ])
    result = classify_sets(traj)
    assert result["MOUTH"] is True


def test_classify_flips_moving_canonical_mono_to_diph() -> None:
    traj = pl.concat([
        _token(0, "KIT_bit", 400, 2000),
        _token(1, "DRESS_bed", 550, 1900),
        _token(2, "TRAP_cat", 700, 1800),
        _token(3, "LOT_cot", 650, 1100),
        _token(4, "FLEECE_bead", 350, _sweep(900, 2600)),  # strong sweep -> diph
    ])
    result = classify_sets(traj)
    assert result["FLEECE"] is True
```

(Import `classify_sets` in the existing `from vowels.aggregate import ...` line.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_aggregate.py -k classify -v`
Expected: FAIL — `classify_sets` not defined.

- [ ] **Step 3: Implement the classifier**

Add the `Final` import (`from typing import Final, cast` — `cast` is already imported) and `DIPHTHONG_NAMES` to the `.labels` import in `aggregate.py`:

```python
from .labels import (
    DIPHTHONG_NAMES,
    SECOND_VOWEL_CENTER_RATIO,
    get_set_name,
    is_diphthong_set,
    is_disyllabic,
    normalize_label,
)
```

Add constants near the top (after `_OFFSET_WINDOW`):

```python
K_LOW: Final[float] = 2.0
K_HIGH: Final[float] = 4.0
MIN_SPREAD: Final[float] = 0.1  # Bark; guards a degenerate (zero-MAD) baseline
```

Add the functions:

```python
def _score_sets(
    traj: pl.DataFrame,
) -> tuple[dict[str, float], dict[str, int], dict[str, bool]]:
    scores: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    disyll: dict[str, bool] = {}
    for (_token_id,), token in traj.group_by("token_id", maintain_order=True):
        label: str = token["label"][0]
        normalized, is_dis = normalize_label(label)
        set_name: str = get_set_name(normalized)
        counts[set_name] = counts.get(set_name, 0) + 1
        disyll[set_name] = disyll.get(set_name, False) or is_dis
        disp: float = token_displacement(token)
        if not math.isnan(disp):
            scores.setdefault(set_name, []).append(disp)
    set_score: dict[str, float] = {s: float(np.median(v)) for s, v in scores.items() if v}
    return set_score, counts, disyll


def _baseline(set_score: dict[str, float], disyll: dict[str, bool]) -> tuple[float, float]:
    mono: list[float] = [
        sc
        for s, sc in set_score.items()
        if s not in DIPHTHONG_NAMES and not disyll.get(s, False)
    ]
    if not mono:
        return 0.0, MIN_SPREAD
    arr: NDArray[np.double] = np.array(mono)
    center: float = float(np.median(arr))
    spread: float = max(float(np.median(np.abs(arr - center))), MIN_SPREAD)
    return center, spread


def _decide(
    set_name: str, score: float, center: float, spread: float, disyll: dict[str, bool]
) -> bool:
    if disyll.get(set_name, False):
        return False
    if score > center + K_HIGH * spread:
        return True
    if score < center + K_LOW * spread:
        return False
    return set_name in DIPHTHONG_NAMES


def classify_sets(traj: pl.DataFrame) -> dict[str, bool]:
    set_score, _counts, disyll = _score_sets(traj)
    center, spread = _baseline(set_score, disyll)
    return {
        s: _decide(s, sc, center, spread, disyll) for s, sc in set_score.items()
    }
```

- [ ] **Step 4: Run tests + ruff**

Run: `uv run pytest tests/test_aggregate.py -k classify -q && uv run ruff check src/`
Expected: PASS, ruff clean. If a borderline test lands on the wrong side, do NOT change `K_LOW`/`K_HIGH` here (Task 7 calibrates them) — instead widen the synthetic sweep so the intent (clearly-moving vs clearly-flat) is unambiguous.

- [ ] **Step 5: Commit**

```bash
git add src/vowels/aggregate.py tests/test_aggregate.py
git commit -m "feat: classify sets as mono/diphthong from Bark displacement"
```

---

### Task 5: Wire classification into aggregation

Make `collapse_token` take the data-driven decision and have `points_from_trajectory` classify once per session and thread it through. `load_points`'s external contract is unchanged.

**Files:**
- Modify: `src/vowels/aggregate.py` (`collapse_token`, `points_from_trajectory`)
- Test: `tests/test_aggregate.py` (update `collapse_token` / `points_from_trajectory` tests)

**Interfaces:**
- Consumes: `classify_sets` (Task 4).
- Produces: `collapse_token(token: pl.DataFrame, label: str, is_diphthong: bool) -> list[dict[str, np.double | str]]`.

- [ ] **Step 1: Update the failing tests**

In `tests/test_aggregate.py`, update every `collapse_token(...)` call to pass the decision explicitly:

- `test_monophthong_yields_one_point`: `collapse_token(df, "TRAP_cat", is_diphthong=False)`
- `test_diphthong_yields_two_suffixed_points`: `collapse_token(df, "PRICE_buy", is_diphthong=True)`
- `test_disyllabic_targets_second_syllable_window`: `collapse_token(df, "2leTTER_butter", is_diphthong=False)`
- `test_nan_f0_falls_back_to_token_mean`: `collapse_token(df, "KIT_bit", is_diphthong=False)`

Replace `test_points_from_trajectory_one_row_per_mono_two_per_diph` (lines 105-124) with a version where the diphthong actually moves (so it survives classification) and there are flat monophthongs for the baseline:

```python
def test_points_from_trajectory_classifies_then_collapses() -> None:
    n = 11
    rel = list(np.linspace(0.0, 1.0, n))

    def block(token_id, label, f1, f2):
        f2col = [float(x) for x in f2] if hasattr(f2, "__len__") else [float(f2)] * n
        return pl.DataFrame({
            "token_id": [token_id] * n,
            "label": [label] * n,
            "rel_time": rel,
            "F0": [120.0] * n,
            "F1_s": [float(f1)] * n,
            "F2_s": f2col,
            "F3_s": [2500.0] * n,
        })

    traj = pl.concat([
        block(0, "KIT_bit", 400, 2000),
        block(1, "DRESS_bed", 550, 1900),
        block(2, "TRAP_cat", 700, 1800),
        block(3, "LOT_cot", 650, 1100),
        block(4, "PRICE_buy", 700, list(np.linspace(1000, 2400, n))),  # moves -> diph
    ])
    points = points_from_trajectory(traj)
    assert set(points.columns) == {"label", "set", "word", "F0", "F1", "F2", "F3"}
    labels = points["label"].to_list()
    assert "PRICE_buy:1" in labels and "PRICE_buy:2" in labels
    assert "TRAP_cat" in labels and "TRAP_cat:1" not in labels
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_aggregate.py -k "collapse or monophthong or diphthong or disyllabic or nan_f0 or classifies" -v`
Expected: FAIL — `collapse_token` still has the 2-arg signature.

- [ ] **Step 3: Update `collapse_token` and `points_from_trajectory`**

Replace `collapse_token` (lines 72-89) with:

```python
def collapse_token(
    token: pl.DataFrame, label: str, is_diphthong: bool
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
    return [_point(token, label, set_name, word, 0.2, 0.8)]
```

Replace `points_from_trajectory` (lines 95-106) with:

```python
def points_from_trajectory(traj: pl.DataFrame) -> pl.DataFrame:
    classification: dict[str, bool] = classify_sets(traj)
    rows: list[dict[str, np.double | str]] = []
    for (_token_id,), token in traj.group_by("token_id", maintain_order=True):
        label: str = token["label"][0]
        normalized, _ = normalize_label(label)
        set_name: str = get_set_name(normalized)
        # Sets whose tokens were all unscorable fall back to the canonical prior.
        is_diph: bool = classification.get(set_name, is_diphthong_set(set_name))
        rows.extend(collapse_token(token, label, is_diph))
    return pl.DataFrame(
        rows,
        schema={
            c: (pl.Utf8 if c in ("label", "set", "word") else pl.Float64)
            for c in _POINT_COLUMNS
        },
    )
```

- [ ] **Step 4: Run the full aggregate suite + ruff**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/`
Expected: all pass, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/vowels/aggregate.py tests/test_aggregate.py
git commit -m "feat: drive diphthong point extraction from per-speaker classification"
```

---

### Task 6: `diphthongs` transparency report

Add a report that shows each set's score, the baseline, canonical vs. final label, and any flips; expose it as `vowels diphthongs <session>`.

**Files:**
- Modify: `src/vowels/aggregate.py` (add `diphthong_report`)
- Modify: `src/vowels/__init__.py` (export `diphthong_report`)
- Modify: `src/vowels/cli.py` (add `diphthongs` command)
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `_score_sets`, `_baseline`, `_decide`, `DIPHTHONG_NAMES` (Task 4).
- Produces: `diphthong_report(session: str) -> pl.DataFrame` with columns `set, n, score, canonical, final, flipped`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_aggregate.py`:

```python
def test_diphthong_report_columns_and_flip(tmp_path, monkeypatch) -> None:
    import vowels.aggregate as agg

    n = 11
    rel = list(np.linspace(0.0, 1.0, n))

    def block(token_id, label, f1, f2):
        f2col = [float(x) for x in f2] if hasattr(f2, "__len__") else [float(f2)] * n
        return pl.DataFrame({
            "token_id": [token_id] * n, "label": [label] * n, "rel_time": rel,
            "F0": [120.0] * n, "F1_s": [float(f1)] * n, "F2_s": f2col,
            "F3_s": [2500.0] * n,
        })

    traj = pl.concat([
        block(0, "KIT_bit", 400, 2000),
        block(1, "DRESS_bed", 550, 1900),
        block(2, "TRAP_cat", 700, 1800),
        block(3, "LOT_cot", 650, 1100),
        block(4, "GOAT_goat", 500, 1200),  # flat canonical diphthong -> flips
    ])
    d = tmp_path / "sX"
    d.mkdir()
    traj.write_parquet(d / "sX_formants.parquet")
    monkeypatch.setattr(agg, "session_dir", lambda s: d)

    report = agg.diphthong_report("sX")
    assert set(report.columns) == {"set", "n", "score", "canonical", "final", "flipped"}
    goat = report.filter(pl.col("set") == "GOAT").row(0, named=True)
    assert goat["canonical"] == "diphthong"
    assert goat["final"] == "monophthong"
    assert goat["flipped"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_aggregate.py -k report -v`
Expected: FAIL — `diphthong_report` not defined.

- [ ] **Step 3: Implement `diphthong_report`**

Add to `src/vowels/aggregate.py`:

```python
def diphthong_report(session: str) -> pl.DataFrame:
    traj: pl.DataFrame = pl.read_parquet(
        session_dir(session) / f"{session}_formants.parquet"
    )
    set_score, counts, disyll = _score_sets(traj)
    center, spread = _baseline(set_score, disyll)
    rows: list[dict[str, object]] = []
    for set_name in sorted(set_score):
        canonical_diph: bool = set_name in DIPHTHONG_NAMES
        final_diph: bool = _decide(
            set_name, set_score[set_name], center, spread, disyll
        )
        rows.append({
            "set": set_name,
            "n": counts.get(set_name, 0),
            "score": round(set_score[set_name], 3),
            "canonical": "diphthong" if canonical_diph else "monophthong",
            "final": "diphthong" if final_diph else "monophthong",
            "flipped": canonical_diph != final_diph,
        })
    return pl.DataFrame(
        rows,
        schema={
            "set": pl.Utf8, "n": pl.Int64, "score": pl.Float64,
            "canonical": pl.Utf8, "final": pl.Utf8, "flipped": pl.Boolean,
        },
    )
```

- [ ] **Step 4: Export and add the CLI command**

In `src/vowels/__init__.py`, add `from .aggregate import diphthong_report` (new import line) and add `"diphthong_report"` to `__all__`.

In `src/vowels/cli.py`, add `diphthong_report` to the `from . import (...)` block, then add this command (e.g. after `projections`):

```python
@app.command()
def diphthongs(session: str) -> None:
    """Report per-set mono/diphthong classification and the flips from canonical."""
    report = diphthong_report(session)
    with pl.Config(tbl_rows=-1):
        print(report)
```

Add `import polars as pl` to `cli.py` if not already imported.

- [ ] **Step 5: Run tests + ruff + smoke the CLI**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run vowels diphthongs --help`
Expected: tests pass, ruff clean, help text prints.

- [ ] **Step 6: Commit**

```bash
git add src/vowels/aggregate.py src/vowels/__init__.py src/vowels/cli.py tests/test_aggregate.py
git commit -m "feat: add vowels diphthongs classification report command"
```

---

### Task 7: Calibrate against session4 and validate

Tune `K_LOW`/`K_HIGH` so session4's known outcomes land correctly, freeze the constants, and confirm the plots.

**Files:**
- Modify (if needed): `src/vowels/aggregate.py` (`K_LOW`/`K_HIGH` values only)

**Interfaces:**
- Consumes: `diphthong_report`, the `formants`/`plot`/`bark`/`projections` CLI.

- [ ] **Step 1: Regenerate session4's trajectory and inspect the report**

```bash
uv run vowels formants session4 --gender M
uv run vowels diphthongs session4
```

Expected outcome (the ground truth from the speaker):
- FACE, GOAT → `final = monophthong`, `flipped = True`
- PRICE, CHOICE, NEAR → `final = diphthong`
- MOUTH → `final = diphthong` (by evidence or by prior)
- SQUARE, CURE → canonical unless the score is decisive

- [ ] **Step 2: Adjust constants only if a known case is wrong**

If FACE/GOAT do not flip to mono, or a true diphthong collapses, edit `K_LOW`/`K_HIGH` in `aggregate.py` and re-run Step 1. Record the chosen values and the resulting report in the commit message. Rule of thumb: raise `K_HIGH` to make monophthong→diphthong flips harder; lower `K_LOW` to make diphthong→monophthong reverts harder (more prior stickiness).

- [ ] **Step 3: Regenerate and eyeball the plots**

```bash
uv run vowels plot session4
uv run vowels bark session4
uv run vowels projections session4
```

Confirm FACE and GOAT now render as single tight clusters (no random short arrows) and the true diphthongs keep sensible onset→offset arrows. Do NOT commit anything under `sessions/`.

- [ ] **Step 4: Run the full suite + ruff**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/`
Expected: all pass, ruff clean.

- [ ] **Step 5: Commit (only if constants changed)**

```bash
git add src/vowels/aggregate.py
git commit -m "chore: calibrate diphthong-detection thresholds on session4"
```

If Step 2 made no change, skip this commit.

---

## Self-Review

**Spec coverage:**
- Metric (Bark onset→offset displacement, F1/F2/F3, median) → Tasks 1, 3, 4. ✓
- Steady-state velocity moved to Bark → Task 2. ✓
- Speaker-relative baseline + asymmetric prior bands → Task 4. ✓
- Disyllabic exclusion + canonical-mono baseline → Task 4 (`_baseline`, `_decide`). ✓
- Shared `bark.py`, plots unchanged → Task 1. ✓
- Pipeline integration (`collapse_token`/`points_from_trajectory`/`load_points`) → Task 5. ✓
- Transparency report `vowels diphthongs` → Task 6. ✓
- NaN-F0 handling (fill for Bark; exclude fully-unvoiced from scoring; z-scored-Hz fallback point) → Tasks 2, 3, 4. ✓
- Validation on session4 + constant calibration → Task 7. ✓
- Tests (test_bark, aggregate updates) → Tasks 1-6. ✓

**Placeholder scan:** No TBD/"handle edge cases"/bare "write tests" — every code step shows full code. ✓

**Type consistency:** `steady_state_index(dims, rel_time, lo, hi)` used consistently in Tasks 2-4; `collapse_token(token, label, is_diphthong)` defined in Task 5 and every test call updated; `_BARK_DIMS`/`_ONSET_WINDOW`/`_OFFSET_WINDOW` defined in Task 2 and reused in 3-5; `classify_sets`/`_score_sets`/`_baseline`/`_decide` signatures match across Tasks 4 and 6. ✓

**Edge guard:** `MIN_SPREAD` prevents a degenerate zero-MAD baseline when monophthong scores are identical (Task 4). ✓
