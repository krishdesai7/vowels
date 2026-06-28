# vowels

A command-line toolkit for extracting and visualizing vowel formants from speech recordings. Give it a `.wav` file and a list of word labels; get back interactive vowel space plots and a Parquet file of F1/F2/F3 measurements.

## What you get

Running the full pipeline on a session produces four outputs:

- **`<session>_vowel_space.html`** — F1/F2 scatter plot with per-set confidence ellipses and mean markers, overlaid on IPA reference positions
- **`<session>_bark_space.html`** — interactive 3D Bark Z vowel space (Openness × Frontness × Roundness)
- **`<session>_bark_projections.html`** — three 2D Bark projections (Frontness×Openness, Frontness×Roundness, Openness×Roundness)
- **`<session>_formants.parquet`** — raw F1/F2/F3 measurements at each nucleus point, one row per token

All plots are interactive HTML files with toggles for lexical sets, display modes, and vowel types.

## Installation

Requires Python 3.14+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd vowels
uv sync
```

## Quickstart

```bash
uv run vowels run <session>
```

This runs the full pipeline — silence detection, labeling, nucleus extraction, formant measurement, and all three plots — in one command.

Options:

```bash
uv run vowels run <session> --gender F           # speaker gender: M (default), F, or C
uv run vowels run <session> --min-sounding-interval 0.15
```

Gender affects the formant ceiling used by Praat's Burg algorithm (M: 5000 Hz, F/C: 5500 Hz) and the analysis window length.

## Input

Place your session files under `sessions/<session>/`:

```text
sessions/
  session1/
    session1.wav
```

Then create a `labels.txt` file listing one label per speech interval detected in the recording, in order. The file is looked up first at `sessions/<session>/labels.txt`, then at `data/labels.txt`.

```text
FLEECE_bleed
TRAP_cat
STRUT_cup
GOOSE_food
2haPPY_coffee
2leTTER_butter
```

**Label format:** `LEXICAL_SET_word`

The lexical set must be one of the [Wells (1982)](https://en.wikipedia.org/wiki/Lexical_set) keywords supported by the toolkit (exact case required):

| Monophthongs                          | Diphthongs                            | Schwa/Reduced        |
| ------------------------------------- | ------------------------------------- | -------------------- |
| FLEECE, KIT, haPPY                    | FACE, GOAT                            | coMMA, leTTER, NURSE |
| DRESS                                 | PRICE, MOUTH, CHOICE                  |                      |
| TRAP, BATH, PALM                      | NEAR, SQUARE, CURE                    |                      |
| LOT, THOUGHT, CLOTH                   |                                       |                      |
| FOOT, GOOSE                           |                                       |                      |
| STRUT, START                          |                                       |                      |
| NORTH, FORCE                          |                                       |                      |

**Disyllabic words** (where the target vowel falls in the second syllable) are prefixed with `2`, e.g. `2haPPY_coffee`, `2leTTER_butter`. The nucleus finder uses a weighted center calculation for the second syllable rather than the word midpoint.

The number of labels must match the number of speech intervals detected in the recording. If they don't match, the `label` step writes a diagnostic CSV (`<session>_intervals.csv`) showing detected vs. expected labels — edit the `expected_label` column and re-run to fix mismatches without re-running silence detection.

## Pipeline steps

The `run` command chains these steps. You can also run them individually if you need to inspect or adjust intermediate outputs:

```bash
# 1. Detect speech intervals and write a TextGrid
uv run vowels silences <session>
uv run vowels silences <session> --min-sounding-interval 0.15  # tune if interval count is off

# 2. Assign labels from labels.txt to speech intervals
uv run vowels label <session>

# 3. Mark nucleus points within each labeled interval
uv run vowels nucleus <session>

# 4. Extract F1/F2/F3 at nucleus points and write parquet
uv run vowels formants <session> --gender M

# 5. Generate plots from existing parquet (re-run without re-measuring)
uv run vowels plot <session>
uv run vowels bark <session>
uv run vowels projections <session>
```

## CLI reference

```
vowels run <session>          Run the full pipeline
vowels silences <session>     Detect speech intervals
vowels label <session>        Assign labels to intervals
vowels nucleus <session>      Mark nucleus measurement points
vowels formants <session>     Extract F1/F2/F3 and write parquet
vowels plot <session>         Generate F1/F2 vowel space HTML
vowels bark <session>         Generate 3D Bark Z vowel space HTML
vowels projections <session>  Generate 2D Bark projection HTMLs
```

Run `uv run vowels <command> --help` for full option details.

## Dependencies

- [parselmouth](https://github.com/YannickJadoul/Parselmouth) — Python bindings for Praat
- [Polars](https://pola.rs) — data manipulation
- [Altair](https://altair-viz.github.io) — interactive 2D plots (vowel space, projections)
- [Plotly](https://plotly.com/python/) — interactive 3D plot (Bark space)
- [Typer](https://typer.tiangolo.com) — CLI
