# Per-speaker diphthong detection — design

**Date:** 2026-06-30
**Status:** Approved (pending spec review)

## Motivation

Whether a Wells lexical set is realized as a monophthong or a diphthong is
currently hardcoded: `schema.DIPHTHONGS` lists eight sets (FACE, GOAT, PRICE,
CHOICE, MOUTH, NEAR, SQUARE, CURE), and `is_diphthong_set` makes
`collapse_token` emit two points (`:1`/`:2`) for exactly those sets and one
point for everything else.

But this varies by speaker. In session4, FACE and GOAT are realized as
**monophthongs** — forcing them into the diphthong branch produces essentially
random short onset→offset arrows. Other speakers may diphthongize a canonically
monophthongal set. Now that fasttrackpy gives us the full smoothed trajectory
per token, we can decide this **from the data, per speaker**, instead of
hardcoding it.

## Approach (resolved in brainstorming)

A Bayesian-flavored rule: the canonical classification is the **prior**, and the
trajectory data must provide **sufficiently strong evidence** to overturn it.
Borderline sets keep their canonical label.

1. **Metric (per token):** onset→offset spectral displacement, measured in the
   speaker-normalized Bark-difference space `(Openness, Frontness, Roundness)`.
2. **Per-set score:** the **median** of the set's per-token displacements.
3. **Boundary:** **speaker-relative**, anchored to this speaker's own
   canonical-monophthong sets (the control group).
4. **Prior:** asymmetric margins around that baseline — a set keeps its
   canonical label unless its score clears the far bar.

All formant work is done in **Bark**, never raw Hz (project-wide principle).

## Bark space

Reuse the existing transform from `plots/bark_space.py`:

- `Z_i = 26.81 · F_i / (1960 + F_i) − 0.53` for `i ∈ {0,1,2,3}` (Traunmüller),
  where `F_0` is F0.
- `Openness = Z1 − Z0`, `Frontness = Z2 − Z1`, `Roundness = Z3 − Z2`
  (Syrdal & Gopal Bark-difference metric — F0-normalized, hence
  speaker-normalized, and it folds in all of F1/F2/F3).

`_bark` and `_add_bark_dims` **move** from `plots/bark_space.py` into a new
shared module `src/vowels/bark.py`; `bark_space.py` imports them from there.
Both aggregation and plotting then use one transform.

## The metric

For one token's trajectory (already sorted by `rel_time`, with Bark dims added):

- **Onset target:** the steady-state frame within `[0.1, 0.45]`.
- **Offset target:** the steady-state frame within `[0.55, 0.9]`.
  (Same windows the diphthong branch already uses to place `:1`/`:2`, so the
  score is literally the distance between the two points we would plot.)
- **Displacement:** Euclidean distance between onset and offset in
  `(Openness, Frontness, Roundness)` Bark space.

### Steady-state detection moves to Bark

`steady_state_index` currently computes frame-to-frame velocity on z-scored
`F1_s`/`F2_s`. It is rewritten to compute velocity as the frame-to-frame
Euclidean distance in `(Openness, Frontness, Roundness)` Bark space, with **no
z-scoring** — the three Bark-difference axes are already in comparable
perceptual units, so the z-scoring (which only existed to stop raw-Hz F2 from
dominating F1) is removed. The min-velocity-in-window selection, the
empty-window fallback (use all frames), and the `[0,1]` `rel_time` contract are
unchanged.

## Classifier

```
score(set)        = median over the set's tokens of onset→offset Bark displacement
baseline center   = median over canonical-MONOPHTHONG sets of score(set)
baseline spread   = MAD   over canonical-MONOPHTHONG sets of score(set)

score < center + k_low  · spread   →  MONOPHTHONG   (overrides a diphthong prior)
score > center + k_high · spread   →  DIPHTHONG     (overrides a monophthong prior)
otherwise                          →  keep canonical label (prior wins)
```

with `k_low < k_high`. The middle band is the hysteresis encoding the prior:
neither a canonical monophthong nor a canonical diphthong flips unless the
evidence clears the far bar. Anchoring on monophthongs means canonical
diphthongs that behave monophthongally for this speaker (FACE/GOAT) cannot
contaminate the baseline.

`k_low` and `k_high` are named constants, calibrated against session4 (see
Validation). Starting values: `k_low = 2.0`, `k_high = 4.0`.

### Scope of the classification

- Only **monosyllabic** sets are classified. **Disyllabic** sets (`2`-prefix:
  haPPY, coMMA, leTTER) are always monophthongs (unchanged) and are excluded
  from both the baseline and from flipping — their measurement window is the
  second syllable, so a full-interval onset/offset displacement would not be
  comparable.
- "Canonical monophthong" = any non-disyllabic set whose name is **not** in
  `schema.DIPHTHONGS` and which is present in the session.

## Pipeline integration

`aggregate.py` gains:

- `classify_sets(traj: pl.DataFrame) -> dict[str, bool]` — computes per-set
  scores, the speaker baseline, and the final mono/diph map for the session.
- `collapse_token` takes the classification map and consults
  `is_diphthong[set_name]` instead of `is_diphthong_set(set_name)`.
- `points_from_trajectory` / `load_points` compute the map once per session and
  thread it through.

`schema.DIPHTHONGS` and `labels.is_diphthong_set` remain — they now express the
**prior**, consumed by `classify_sets`.

Plots are unchanged: `vowel_space` / `bark_space` detect diphthong points via
the `:` suffix on labels, which now reflects the per-speaker decision
automatically.

## Transparency report

New CLI command `vowels diphthongs <session>` prints, per set:

```
set     n   score   canonical   final    flipped?
FACE    3   0.41    diphthong   mono     ← flipped
GOAT    3   0.55    diphthong   mono     ← flipped
PRICE   4   3.812   diphthong   diph
MOUTH   3   1.95    diphthong   diph
...
baseline: center=0.62  spread=0.21  (k_low=2.0 → 1.04, k_high=4.0 → 1.46)
```

so borderline calls and flips are inspectable. It reads the existing parquet;
it does not re-extract.

## NaN-F0 handling

Bark dims need F0 (`Z0`). For frames with NaN F0, substitute the token's mean
finite F0 before computing Bark dims (consistent with the existing point-F0
fallback). A token with **no** finite F0 anywhere is excluded from the baseline
and from its set's score; for its own point extraction it falls back to the old
z-scored-Hz steady-state path (the rare, unavoidable raw-Hz exception). This
mirrors the single all-unvoiced token already seen in session2.

## Validation

session4 is a labeled ground truth for the speaker who motivated this:

- FACE, GOAT → must flip to **monophthong**.
- PRICE, CHOICE, NEAR → must stay **diphthong**.
- MOUTH, NEAR → must stay **diphthong** (debatable, kept by the prior).
- SQUARE, CURE → keep canonical unless data is decisive.

Calibrate `k_low`/`k_high` so these land correctly, then freeze the constants.
Re-run session4's `formants`→plots and eyeball that FACE/GOAT collapse to single
tight clusters and the true diphthongs keep sensible arrows.

## Tests

- `tests/test_bark.py` (new): `_bark`/`add_bark_dims` numeric correctness on a
  known frame; round-trip of the three difference dims.
- `tests/test_aggregate.py`:
  - steady-state velocity in Bark: synthetic monophthong (flat Bark middle,
    moving edges) → min-velocity frame in the flat region; diphthong → two
    distinct targets; disyllabic → second-syllable window.
  - displacement metric: synthetic moving vs. flat token → large vs. small
    score.
  - `classify_sets`: synthetic trajectory where one canonical-diphthong set is
    flat (flips to mono), one canonical-monophthong set glides (flips to diph),
    and a borderline set stays at its prior.
  - NaN-F0 token excluded from scoring but still yields a point.
- `tests/test_formants.py`: unchanged (`rel_time` contract still holds).

## Known approximations / future work

- Sets with very few tokens get a less stable median; the report shows `n` so
  thin sets are visible. No special-casing.
- `k_low`/`k_high` are calibrated on one speaker (session4); revisit if more
  labeled speakers become available.
- Per-token classification (a set that is diphthongal in some words but not
  others) is out of scope — the decision is per set, per speaker.
