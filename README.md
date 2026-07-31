# vowels

A command-line toolkit for extracting and visualizing vowel formants from speech recordings. Give it a `.wav` file and a list of word labels; get back interactive vowel space plots and a Parquet file of F1/F2/F3 formant trajectories.

## What you get

Running the full pipeline on a session produces four outputs:

- **`<session>_vowel_space.html`** — F1/F2 scatter plot with per-set confidence ellipses and mean markers, overlaid on IPA reference positions
- **`<session>_bark_space.html`** — interactive 3D Bark Z vowel space (Openness × Frontness × Roundness)
- **`<session>_bark_projections.html`** — three 2D Bark projections (Frontness×Openness, Frontness×Roundness, Openness×Roundness)
- **`<session>_formants.parquet`** — the full F1/F2/F3 trajectory for every token (many rows per token), the single source of truth for every plot

All plots are interactive HTML files with toggles for lexical sets, display modes, and vowel types.

The parquet holds whole trajectories, not one point per token. Collapsing each token to a single steady-state (minimum-velocity) measurement happens in memory when a plot command loads the parquet — so you can re-plot, re-classify, or switch dialects without re-measuring the audio.

## Installation

Requires Python 3.14+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd vowels
uv sync --frozen
```

## Quickstart

```bash
uv run vowels run <session>
```

This runs the full pipeline — silence detection, labeling, formant tracking, and all three plots — in one command.

Options:

```bash
uv run vowels run <session> --gender F                 # speaker gender: M (default), F, or C
uv run vowels run <session> --dialect RP               # speaker dialect: GA (default) or RP
uv run vowels run <session> --min-sounding-interval 0.15
```

`./run.sh <session> [gender] [dialect]` is a thin wrapper over the same command.

**Gender** sets the max-formant *search range* handed to [fasttrackpy](https://fasttrackiverse.github.io/fasttrackpy/), which then auto-picks the best ceiling per token, along with the analysis window and pitch floor:

| Gender | Max-formant range | Window | Pitch floor |
| ------ | ----------------- | ------ | ----------- |
| M      | 4500–5500 Hz      | 25 ms  | 75 Hz       |
| F / C  | 5000–6500 Hz      | 30 ms  | 100 Hz      |

**Dialect** sets the prior and baseline for monophthong/diphthong classification — see [Per-speaker diphthong detection](#per-speaker-diphthong-detection).

## Input

Place your session files under `sessions/<session>/`:

```text
sessions/
  session1/
    session1.wav
    labels.csv
```

`labels.csv` lists one row per speech interval detected in the recording, in order. It is looked up first at `sessions/<session>/labels.csv`, then at `data/labels.csv`.

```csv
syllable,set,word
1,FLEECE,bleed
1,TRAP,cat
1,STRUT,cup
1,GOOSE,food
2,haPPY,coffee
2,leTTER,butter
```

- **`set`** — the Wells lexical set (see below), spelled exactly
- **`word`** — the word as read aloud
- **`syllable`** — which syllable carries the target vowel, `1` or `2`

Each row becomes the internal label `LEXICAL_SET_word`, with a `2` prefix when `syllable` is 2 (`2haPPY_coffee`). The nucleus finder uses a weighted center calculation for a second-syllable target rather than the word midpoint.

**Label format:** `LEXICAL_SET_word`

The lexical set must be one of the [Wells (1982)](https://en.wikipedia.org/wiki/Lexical_set) keywords supported by the toolkit (**exact case required** — `haPPY`, `coMMA`, and `leTTER` are parsed literally and are corrupted by uppercasing):

| Monophthongs        | Diphthongs           | Schwa/Reduced        |
| ------------------- | -------------------- | -------------------- |
| FLEECE, KIT, haPPY  | FACE, GOAT           | coMMA, leTTER, NURSE |
| DRESS               | PRICE, MOUTH, CHOICE |                      |
| TRAP, BATH, PALM    | NEAR, SQUARE, CURE   |                      |
| LOT, THOUGHT, CLOTH |                      |                      |
| FOOT, GOOSE         |                      |                      |
| STRUT, START        |                      |                      |
| NORTH, FORCE        |                      |                      |

This table is the canonical (RP-flavoured) grouping. Which sets are *actually* plotted as diphthongs is decided per speaker from the data, not from this table — NEAR/SQUARE/CURE, for instance, are r-colored rather than gliding for a General American speaker.

**Diphthong endpoints** may be marked with `:1` and `:2` suffixes; the suffix is stripped when the label is parsed and drives the onset→offset arrow in the plots.

The number of rows must match the number of speech intervals detected in the recording. If they don't match, the `label` step writes a diagnostic CSV (`<session>_intervals.csv`) showing detected vs. expected labels — edit the `expected_label` column and re-run `vowels label` to fix mismatches without re-running silence detection.

### Reading list

The same `labels.csv` feeds the recording prompter, which pages through the `word` column one word at a time (space/enter/→ next, ← back, esc quit) so a speaker can read the list while you record:

```bash
uv run python -m vowels.record_audio            # defaults to data/labels.csv
uv run python -m vowels.record_audio sessions/session5/labels.csv
```

The prompter is a standalone module, not a `vowels` subcommand.

## Pipeline steps

The `run` command chains these steps. You can also run them individually if you need to inspect or adjust intermediate outputs:

```bash
# 1. Detect speech intervals and write a TextGrid
uv run vowels silences <session>
uv run vowels silences <session> --min-sounding-interval 0.15  # tune if interval count is off

# 2. Assign labels from labels.csv to speech intervals
uv run vowels label <session>

# 3. Track formant trajectories per labeled vowel and write the parquet
uv run vowels formants <session> --gender M

# 4. Generate plots from the existing parquet (re-run without re-measuring)
uv run vowels plot <session>
uv run vowels bark <session>
uv run vowels projections <session>

# Inspect the mono/diphthong calls behind the plots
uv run vowels diphthongs <session>
uv run vowels diphthongs <session> --dialect RP
```

`--dialect` / `-d` belongs on every command that reads the parquet into points — `plot`, `bark`, `projections`, `diphthongs`, and `run`. `silences`, `label`, and `formants` do not take it.

## Per-speaker diphthong detection

Whether a Wells set is realized as a monophthong or a diphthong varies by speaker, so the toolkit decides it **from that session's trajectory data** rather than from a fixed list. Each token is scored by its onset→offset spectral displacement in Bark (Openness, Frontness, Roundness) space; scores are medianed per set and compared against a **speaker baseline** built from that speaker's own plain-monophthong sets (Tukey Q3 / IQR).

The decision is Bayesian-flavoured: the dialect's canonical classification is the prior, and the data must clear Tukey's far-outlier fence (`Q3 + 3·IQR`) to promote a monophthong to a diphthong, or sit inside the baseline body (`≤ Q3`) to demote a diphthong. Anything in between keeps the prior.

Dialect matters because the same sets differ for real phonetic reasons. RP realizes NEAR/SQUARE/CURE as centring diphthongs (`ɪə, ɛə, ʊə`) — they genuinely glide. GA realizes them as vowel + rhotic (`ɪr, ɛr, ʊr`), whose falling F3 reads as large Bark displacement with no gliding nucleus, so GA excludes its eight r-colored sets from both the gliding prior and the baseline pool. Pass the wrong dialect to a rhotic speaker and the r-colored sets inflate the baseline, swallowing the genuine diphthongs.

`vowels diphthongs` prints the whole decision table (`set, n, score, canonical, final, flipped`) plus the baseline thresholds, so every call and every flip from canonical is inspectable.

## CLI reference

```
vowels run <session>          Run the full pipeline
vowels silences <session>     Detect speech intervals
vowels label <session>        Assign labels to intervals
vowels formants <session>     Track F1/F2/F3 trajectories and write parquet
vowels plot <session>         Generate F1/F2 vowel space HTML
vowels bark <session>         Generate 3D Bark Z vowel space HTML
vowels projections <session>  Generate 2D Bark projection HTMLs
vowels diphthongs <session>   Report per-set mono/diphthong classification
```

Run `uv run vowels <command> --help` for full option details.

## Development

```bash
uv run just m   # mutable: sync, type-infer, format, autofix lint, test
uv run just i   # immutable: frozen sync, type-check, format check, lint, audit, test
```

CI runs `just i` on every push and pull request to `main`.

## Dependencies

- [fasttrackpy](https://fasttrackiverse.github.io/fasttrackpy/) — formant trajectory tracking with automatic ceiling selection
- [parselmouth](https://github.com/YannickJadoul/Parselmouth) — Python bindings for Praat (silence detection, TextGrids)
- [Polars](https://pola.rs) — data manipulation
- [Altair](https://altair-viz.github.io) — interactive 2D plots (vowel space, projections)
- [Plotly](https://plotly.com/python/) — interactive 3D plot (Bark space)
- [Typer](https://typer.tiangolo.com) — CLI
