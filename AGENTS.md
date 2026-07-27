# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a phonetic analysis toolkit for extracting and visualizing vowel formants from speech recordings using Praat (via the parselmouth Python library). It processes audio files with TextGrid annotations to generate vowel space plots based on lexical sets (Wells' standard lexical sets like FLEECE, KIT, TRAP, etc.).

## Commands

### Full Pipeline

Run the complete analysis for a session:

```zsh
uv run vowels run <session>
uv run vowels run <session> --gender F
uv run vowels run <session> --min-sounding-interval 0.15
```

### Individual Steps

```zsh
# Step 1: Detect silences and create initial TextGrid
uv run vowels silences <session>

# Step 2: Label TextGrid intervals with vowel annotations
uv run vowels label <session>

# Step 3: Extract formants and save to parquet
uv run vowels formants <session> --gender M

# Step 4: Generate plots from existing parquet
uv run vowels plot <session>
uv run vowels bark <session>
uv run vowels projections <session>

# Inspect the per-speaker monophthong/diphthong classification
uv run vowels diphthongs <session>
uv run vowels diphthongs <session> --dialect RP
```

### Dialect

The plot and report commands take `--dialect` / `-d` (`GA` or `RP`, default
`GA`) — see [Per-speaker diphthong detection](#per-speaker-diphthong-detection).
It sets the classification prior and baseline, so it belongs on any command that
reads the parquet into points: `plot`, `bark`, `projections`, `diphthongs`, and
`run`. `silences`/`label`/`formants` do not take it.

```zsh
uv run vowels run <session> --dialect RP
uv run vowels plot <session> -d GA
```

### Dependencies

Uses `uv` for package management. Install dependencies with:

```zsh
uv sync --frozen
```

## Architecture

### Data Flow

1. **Input**: `sessions/<session>/<session>.wav` (audio) + `labels.txt` (looked up first at `sessions/<session>/labels.txt`, falling back to `data/labels.txt`)

2. **`silences`**: Runs Praat's "To TextGrid (silences)" on the audio to create `<session>.TextGrid` with "silent" and "sounding" intervals. Tune `--min-sounding-interval` if the detected interval count doesn't match the number of labels.

3. **`label`**: Reads `labels.txt` and assigns labels to "sounding" intervals in the TextGrid, producing `<session>_labeled.TextGrid`. If the label count doesn't match the interval count, writes a diagnostic CSV (`<session>_intervals.csv`) for manual correction and exits with an error.

4. **`formants`**: Extracts F1/F2/F3 by tracking the full formant trajectory per labeled vowel interval (fasttrackpy, auto-selected ceiling) and saves the whole trajectory (many rows per token) to `<session>_formants.parquet` — the single source of truth. It does **not** generate plots. Steady-state (minimum-velocity) collapse to one representative point per token happens later, in-memory, when a plot command loads the parquet via `aggregate.load_points`.

5. **`plot` / `bark` / `projections`**: Each reads `<session>_formants.parquet`, collapses to steady-state points, and writes one interactive HTML plot:
   - `plot` → `<session>_vowel_space.html` — F1/F2 scatter plot with ellipses and mean markers
   - `bark` → `<session>_bark_space.html` — 3D Bark Z vowel space (Openness × Frontness × Roundness)
   - `projections` → `<session>_bark_projections.html` — three 2D Bark projections

   Whether a set is drawn as one point or a `:1`→`:2` diphthong arrow is decided per speaker (see below), not hardcoded, so each of these accepts `--dialect`.

6. **`diphthongs`**: Reads the parquet and prints the per-set classification table (`set, n, score, canonical, final, flipped`) plus the baseline line, so the mono/diphthong calls and any flips from the dialect prior are inspectable. It does not re-extract or write plots. Takes `--dialect`.

The `run` command chains `silences → label → formants → plot → bark → projections`.

### Per-speaker diphthong detection

Whether a Wells set is realized as a monophthong or a diphthong varies by speaker, so it is decided **from the trajectory data per session** rather than from a fixed list. `aggregate.classify_sets` scores each token by its onset→offset spectral displacement in Bark `(Openness, Frontness, Roundness)` space, medians per set, and compares against a **speaker baseline** built from that speaker's own plain-monophthong sets (Tukey Q3 / IQR). The decision is Bayesian-flavored: the **dialect's** canonical classification is the prior, and the data must clear Tukey's far-outlier fence (`Q3 + 3·IQR`) to promote a monophthong to diphthong, or sit inside the baseline body (`≤ Q3`) to demote a diphthong; anything between keeps the prior. `schema.DIPHTHONGS` remains as the dialect-agnostic default prior.

`schema.Dialect` (`GA`, `RP`) carries a `DialectProfile` with the dialect's rhoticity, expected IPA per set, gliding prior, and r-colored sets. The prior and baseline are dialect-specific because the same sets differ for real phonetic reasons: RP realizes NEAR/SQUARE/CURE as centering diphthongs (they glide), whereas GA realizes them as vowel + rhotic, whose falling F3 reads as large Bark displacement with no gliding nucleus — so GA excludes its eight r-colored sets from the baseline pool and from the gliding prior. Pass the wrong dialect and a rhotic speaker's r-colored sets will inflate the baseline and swallow the genuine diphthongs.

### Label Format

Labels follow the pattern `LEXICAL_SET_word` (e.g., `FLEECE_beat`, `TRAP_cat`). Diphthong measurements are marked with `:1` and `:2` suffixes. Disyllabic words are prefixed with `2` (e.g., `2haPPY_coffee`, `2leTTER_butter`).

Mixed-case set names (`haPPY`, `coMMA`, `leTTER`) must be entered with exact case — the parser does not normalise case, and uppercasing would corrupt these names.

### Key Parameters

- **Gender**: Sets fasttrackpy's max-formant **search range** (it auto-picks the best ceiling per token) plus window length and pitch floor — M: 4500–5500 Hz, 25 ms, 75 Hz; F/C: 5000–6500 Hz, 30 ms, 100 Hz
- **Dialect**: `GA` (default) or `RP`; sets the diphthong prior and the baseline exclusion (see [Per-speaker diphthong detection](#per-speaker-diphthong-detection)). Passed to `plot`/`bark`/`projections`/`diphthongs`/`run` via `--dialect`.
- **`data/standards/male_standard.parquet`**: IPA reference vowel positions (Openness/Frontness/Roundness in Bark) overlaid on plots for comparison

### Schema notes

- **`Wells` values are their own names** (`Wells.PRICE == "PRICE"`), so a member is interchangeable with the parsed label text. Colors are **not** on the value — they live in `schema.COLORS` behind the `Wells.color` property. Do not put presentation or IPA data on the enum value.
