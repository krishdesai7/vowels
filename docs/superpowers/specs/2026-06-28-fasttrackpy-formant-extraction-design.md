# fasttrackpy-based formant extraction — design

**Date:** 2026-06-28
**Status:** Approved (pending spec review)

## Motivation

The current pipeline measures formants at hardcoded points: `t=0.5` for
monophthongs, `t=0.25`/`t=0.75` for diphthongs (`pipeline/nucleus.py:19-25`),
sampling a single raw Burg frame per point (`pipeline/formants.py:74-82`) with a
fixed gender→ceiling guess. Two weaknesses:

1. The measurement *timepoint* is assumed, not derived from the acoustics.
2. The formant ceiling is a fixed per-gender guess, and a single raw frame is
   noisy.

`fasttrackpy` (already added, v0.6.1) addresses both: it tracks the **whole
formant trajectory** across each vowel and **auto-selects the best max-formant
ceiling** per token (tries `nstep` ceilings between `min_max_formant` and
`max_max_formant`, keeps the smoothest-tracking analysis via DCT smoothing). It
does **not** decide where the vowel target is — so we add a **data-driven
steady-state (minimum-velocity) detector** to pick the representative point,
which is the systematic answer to weakness (1).

## Key facts about fasttrackpy (verified from installed source)

- `process_audio_file(path, xmin, xmax, …)` → one `CandidateTracks` for one
  interval. Fits our data model: drive it off our labeled vowel intervals.
- `process_audio_textgrid(…)` expects **MFA-style Word/Phone tiers** with regex
  `[AEIOU]` label matching — our custom labeled/nucleus tiers do not fit it.
  **Not used.**
- `CandidateTracks.winner` is a `OneTrack` exposing `.to_df(output="formants")` →
  polars with columns `F1..F4` (raw), `F1_s..F4_s` (DCT-smoothed), smoothed
  bandwidths, `error` (smooth error), `time`, `max_formant` (chosen ceiling),
  plus `label`/`id`/`group` when an interval is attached.
- `CandidateTracks` also computes `.f0` (per-frame pitch). So we get F0–F3 +
  bandwidths + chosen ceiling for free.

## Design

### 1. Pipeline shape

The `nucleus` point-tier step is **retired**. Its linguistic knowledge migrates
into the new extraction/aggregation code rather than disappearing:

- which Wells sets are diphthongs (`schema.DIPHTHONGS`),
- the `2`-prefix disyllabic rule (`normalize_label`),
- the second-syllable weighting ratio (`SECOND_VOWEL_CENTER_RATIO`).

New `run` sequence: `silences → label → formants → plot/bark/projections`. The
`formants` step reads `*_labeled.TextGrid` + wav directly (no nucleus tier).

The `nucleus` CLI command is **removed outright** (not kept as a deprecated
no-op), along with `make_nucleus_points` and its export.

### 2. `formants` step (rewrite of `pipeline/formants.py`)

For each interval on the labeled tier that is non-silent, non-empty, and longer
than `min_duration` (0.05 s):

- Call `process_audio_file(wav, xmin=t1, xmax=t2, …)` → `CandidateTracks`; take
  `.winner`.
- Emit the winner's **full smoothed trajectory** as many rows.

Output schema (`{session}_formants.parquet`, the single source of truth — a
trajectory, many rows per token):

```
token_id, label, set, word, is_diphthong, is_disyllabic,
time, rel_time,            # rel_time = (time - t1) / (t2 - t1)
F0,                        # from CandidateTracks.f0 (per frame)
F1, F2, F3,                # raw winner formants
F1_s, F2_s, F3_s,          # DCT-smoothed
B1, B2, B3,                # smoothed bandwidths
max_formant, error         # fasttrack's chosen ceiling + smooth error (diagnostics)
```

`parse_labels` (set/word extraction from `LEXICAL_SET_word`, stripping `:N`
suffixes and the `2` prefix) is preserved.

### 3. Gender → ceiling search range

Instead of one fixed ceiling, gender maps to fasttrack's search bounds; it
auto-picks the best per token:

| Gender | min_max_formant | max_max_formant | window_length | pitch_floor |
|--------|-----------------|-----------------|---------------|-------------|
| M      | 4500            | 5500            | 0.025         | 75          |
| F/C    | 5000            | 6500            | 0.030         | 100         |

`n_formants = 4` tracked, F1–F3 reported. These bracket the old fixed
5000/5500 ceilings; widen if fasttrack clips against a bound frequently.

### 4. New `pipeline/aggregate.py` — steady-state collapse

`load_points(session) -> pl.DataFrame` reads the trajectory parquet and returns
the **one-row-per-token point contract** the plots already expect:

```
label   # diphthong points carry :1 / :2 suffix
set, word, F0, F1, F2, F3
```

Per token:

- Compute frame-to-frame velocity on the smoothed track. **Normalize first**
  (z-score `F1_s`/`F2_s` across the token's frames) so F2's larger Hz range does
  not dominate — raw Euclidean velocity would be almost entirely F2:
  `v[i] = sqrt(ΔF1_z² + ΔF2_z²)`.
- **Monophthong:** pick the min-velocity frame within central
  `rel_time ∈ [0.2, 0.8]` (avoids edge artifacts); emit one row, `F1–F3` =
  smoothed values at that frame.
- **Diphthong:** min-velocity in `[0.1, 0.45]` → `…:1`, and `[0.55, 0.9]` →
  `…:2` (two rows, matching the plot contract).
- **Disyllabic (`2`-prefix, ~9 tokens):** restrict the search window to ±0.15
  around `SECOND_VOWEL_CENTER_RATIO` (= 5/6 ≈ 0.833) so the target lands on the
  second syllable as today. *Deliberately approximate:* a 2-syllable interval
  has two steady states; we pick the second.

`F0` for the point is the smoothed-track value at the chosen frame (or the
per-token mean F0 if the chosen frame's F0 is NaN).

The three plot loaders switch from reading `_formants.parquet` directly to
calling `load_points(session)`:

- `plots/vowel_space.py` `save_chart`
- `plots/bark_space.py` `_load_formants`
- `plots/bark_space.py` projections loader

Everything downstream of the loader (Bark transform, ellipses, means, diphthong
arrows) is untouched because the contract is preserved.

### 5. Tests

- `tests/test_nucleus.py` → retired, replaced by `tests/test_aggregate.py`:
  - synthetic monophthong trajectory (flat middle, moving edges) → assert
    min-velocity frame lands in the flat central region;
  - synthetic diphthong → two distinct targets in the two halves;
  - synthetic disyllabic → target lands in the second-syllable window;
  - velocity normalization: a track moving only in F2 (large Hz) vs only in F1
    (small Hz) by equal *perceptual* amounts yields comparable velocities.
- `tests/test_formants.py`: keep `parse_labels` coverage (adapt import if the
  function moves).
- `tests/test_label_parsing.py`, `tests/test_ellipse.py`: unaffected.

## Decisions (resolved)

1. **Output model:** full trajectory stored; plots aggregate in-memory.
2. **Aggregation:** steady-state (min smoothed-track velocity).
3. **Velocity normalization:** z-score per token.
4. **Gender ceiling ranges:** as tabled above.
5. **`nucleus` CLI command:** removed outright.

## Known approximations / future work

- Disyllabic handling picks the second steady state via a fixed window — fine
  for the current scripted word lists; revisit if multi-syllable coverage grows.
- `process_audio_file` re-reads the wav per interval (~100 tokens → acceptable).
  Could slice a single `Sound` and call `CandidateTracks` directly for speed.
- Edge frames of each extracted interval can be unreliable; the central
  `rel_time` search windows mitigate this.
- Trajectory storage unlocks future trajectory plots / alternative aggregation
  methods without re-extracting.
