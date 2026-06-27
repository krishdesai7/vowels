# vowels

A command-line toolkit for extracting and visualizing vowel formants from speech recordings. Give it a `.wav` file and a list of word labels; get back an interactive vowel space plot and a CSV of F1/F2/F3 measurements.

## What you get

Running the full pipeline on a session produces three outputs:

- **Vowel space scatter plot** — every token plotted by F1/F2, labeled with its word, overlaid on IPA reference positions
- **Lexical set plot** — tokens for a single target set (e.g. STRUT) with a confidence ellipse
- **Means plot** — mean F1/F2 position for every lexical set in the recording
- **`<session>_formants.csv`** — raw F1/F2/F3 measurements at each nucleus point, one row per token

All plots are interactive HTML files (Altair/Vega-Lite).

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

This runs the full pipeline — silence detection, labeling, nucleus extraction, formant measurement, and plotting — in one command.

Options:

```bash
uv run vowels run <session> --gender F          # speaker gender: M (default), F, or C
uv run vowels run <session> --mode diph         # include diphthong measurements
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

Then create a `labels.txt` file at the project root listing one label per speech interval detected in the recording, in order:

```text
FLEECE_bleed
TRAP_cat
STRUT_cup
GOOSE_food
2HAPPY_coffee
2LETTER_butter
```

**Label format:** `LEXICAL_SET_word`

The lexical set must be one of the [Wells (1982)](https://en.wikipedia.org/wiki/Lexical_set) keywords supported by the toolkit:

| Monophthongs       | Diphthongs         | Reduced |
| ------------------ | ------------------ | ------- |
| FLEECE, KIT, happY | FACE, GOAT         | commA   |
| GOOSE, FOOT        | PRICE, MOUTH       | lettER  |
| DRESS              | CHOICE             | NURSE   |
| THOUGHT, CLOTH     | NEAR, SQUARE, CURE |         |
| TRAP, BATH, PALM   |                    |         |
| LOT, START, STRUT  |                    |         |
| FORCE, NORTH       |                    |         |

**Disyllabic words** (where the target vowel falls in the second syllable) are prefixed with `2`, e.g. `2HAPPY_coffee`, `2LETTER_butter`. The nucleus finder uses a weighted center calculation for the second syllable rather than the word midpoint.

The number of labels must match the number of speech intervals detected in the recording. If they don't match, re-run the silence detector with a tuned `--min-sounding-interval` value.

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
uv run vowels nucleus <session> --mode diph   # two points per diphthong (25% and 75%)
uv run vowels nucleus <session> --mode all    # monophthongs + diphthongs

# 4. Extract F1/F2/F3 at nucleus points and write CSV
uv run vowels formants <session> --gender M

# 5. Generate plots from existing CSV (re-run without re-measuring)
uv run vowels plot <session> --mode mono
```

## CLI reference

```bash
vowels run <session>       Run the full pipeline
vowels silences <session>  Detect speech intervals
vowels label <session>     Assign labels to intervals
vowels nucleus <session>   Mark nucleus measurement points
vowels formants <session>  Extract F1/F2/F3 and write CSV
vowels plot <session>      Generate plots from existing CSV
```

Run `uv run vowels <command> --help` for full option details.

## Dependencies

- [parselmouth](https://github.com/YannickJadoul/Parselmouth) — Python bindings for Praat
- [Polars](https://pola.rs) — data manipulation
- [Altair](https://altair-viz.github.io) — interactive plots
- [Typer](https://typer.tiangolo.com) — CLI
