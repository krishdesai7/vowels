# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a phonetic analysis toolkit for extracting and visualizing vowel formants from speech recordings using Praat (via the parselmouth Python library). It processes audio files with TextGrid annotations to generate vowel space plots based on lexical sets (Wells' standard lexical sets like FLEECE, KIT, TRAP, etc.).

## Commands

### Full Pipeline

Run the complete analysis for a session:
```bash
./run.sh <session> [gender] [target_lexical_set] [show_diphthongs]
# Examples:
./run.sh session2
./run.sh session2 F
./run.sh session2 M STRUT
./run.sh session2 M STRUT true
```

### Individual Scripts

```bash
# Step 1: Detect silences and create initial TextGrid
uv run detect_silences <session>

# Step 2: Label TextGrid intervals with vowel annotations
uv run label_textgrid <session>

# Step 3: Create nucleus point tier for formant extraction
uv run make_nucleus_points <session>

# Step 4: Extract formants and generate plots
uv run extract_formants <session> --Gender=M --target_lexical_set=STRUT --show_diphthongs=false
```

### Dependencies

Uses `uv` for package management. Install dependencies with:
```bash
uv sync
```

## Architecture

### Data Flow

1. **Input**: `sessions/<session>/<session>.wav` (audio) + `labels.txt` (copied to session directory on first run)

2. **detect_silences.py**: Runs Praat's "To TextGrid (silences)" on the audio to create `<session>.TextGrid` with "silent" and "sounding" intervals. Tune `--min_sounding_interval` if the detected interval count doesn't match labels.txt.

3. **label_textgrid.py**: Reads labels.txt and assigns them to "sounding" intervals in the TextGrid, producing `<session>_labeled.TextGrid`

4. **make_nucleus_points.py**: Creates a "nucleus" point tier marking vowel measurement points. Monophthongs get one midpoint; diphthongs (when enabled) get two points at 25% and 75% of interval duration. Disyllabic words (prefixed with `2`) use weighted center calculation for the second syllable.

5. **extract_formants.py**: Extracts F1/F2/F3 at nucleus points using Praat's Burg algorithm, saves to CSV, and generates three types of vowel plots:
   - Word-labeled scatter plot (all tokens)
   - Single lexical set plot with enclosing ellipse
   - Mean values plot across all sets

### Label Format

Labels follow the pattern `LEXICAL_SET_word` (e.g., `FLEECE_beat`, `TRAP_cat`). Diphthong measurements are marked with `:1` and `:2` suffixes. Disyllabic words are prefixed with `2` (e.g., `2HAPPY_coffee`, `2LETTER_butter`).

### Key Parameters

- **Gender**: Affects formant ceiling (M: 5000Hz, F/C: 5500Hz) and window length (M: 25ms, F: 30ms)
- **standard.csv**: Reference vowel positions (IPA symbols with F1/F2/F3 values) overlaid on plots for comparison
