import parselmouth
from parselmouth.praat import call
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Literal
from fire import Fire

# -----------------------------------------------------------------------------
# Color palette: vowels grouped by phonetic quality for semantic meaning
# -----------------------------------------------------------------------------
# High front vowels (FLEECE, KIT, HAPPY)
# High back vowels (GOOSE, FOOT)
# Mid front vowels (FACE, DRESS)
# Mid back vowels (GOAT, THOUGHT, FORCE, NORTH, CLOTH)
# Low vowels (TRAP, BATH, PALM, LOT, START, STRUT)
# Diphthongs (PRICE, MOUTH, CHOICE, NEAR, SQUARE, CURE)
# Reduced vowels (COMMA, LETTER, NURSE)

VOWEL_COLORS: dict[str, str] = {
    # High front - blue tones
    "FLEECE": "#2166ac",
    "KIT": "#67a9cf",
    "HAPPY": "#92c5de",
    # High back - purple tones
    "GOOSE": "#762a83",
    "FOOT": "#9970ab",
    # Mid front - green tones
    "FACE": "#1b7837",
    "DRESS": "#5aae61",
    # Mid back - orange/brown tones
    "GOAT": "#d95f02",
    "THOUGHT": "#e6ab02",
    "FORCE": "#b35806",
    "NORTH": "#d8b365",
    "CLOTH": "#bf812d",
    # Low vowels - red tones
    "TRAP": "#b2182b",
    "BATH": "#d6604d",
    "PALM": "#c51b7d",
    "LOT": "#e08214",
    "START": "#f46d43",
    "STRUT": "#fdae61",
    # Diphthongs - teal/cyan
    "PRICE": "#01665e",
    "MOUTH": "#35978f",
    "CHOICE": "#80cdc1",
    "NEAR": "#018571",
    "SQUARE": "#66c2a5",
    "CURE": "#5ab4ac",
    # Reduced - gray tones
    "COMMA": "#636363",
    "LETTER": "#969696",
    "NURSE": "#525252",
}

def get_color(lexical_set: str) -> str:
    """Get color for a lexical set, with fallback."""
    return VOWEL_COLORS.get(lexical_set.upper(), "#404040")

def extract_formants(dir: Path, session: str, Gender: Literal['M', 'F', 'C'] = 'M') -> None:
    wav_path: Path = dir / f"{session}.wav"
    tg_path: Path = dir / f"{session}_nucleus.TextGrid"
    tier_name: str = "nucleus"
    formant_ceiling: int = 5000 if Gender == 'M' else 5500
    window_length: float = 0.025 if Gender == 'M' else 0.03

    sound: parselmouth.Sound = parselmouth.Sound(str(wav_path))
    tg: parselmouth.Data = parselmouth.read(str(tg_path))

    n_tiers: int = call(tg, "Get number of tiers")
    tier_index: int | None = None
    for i in range(1, n_tiers + 1):
        if call(tg, "Get tier name", i) == tier_name:
            tier_index: int = i
            break

    n_points: int = call(tg, "Get number of points", tier_index)
    print(f"Found {n_points} vowel nuclei")

    formant_obj: parselmouth.Formant = call(sound, "To Formant (burg)", 0.0, 5, formant_ceiling, window_length, 50)
    records: list[dict[str, float | str]] = []
    for i in range(1, n_points + 1):
        time: float = call(tg, "Get time of point", tier_index, i)
        label: str = call(tg, "Get label of point", tier_index, i)
        if not label:
            continue
        f1: float = call(formant_obj, "Get value at time", 1, time, "Hertz", "Linear")
        f2: float = call(formant_obj, "Get value at time", 2, time, "Hertz", "Linear")
        f3: float = call(formant_obj, "Get value at time", 3, time, "Hertz", "Linear")
        records.append({"time": time, "label": label, "F1": f1, "F2": f2, "F3": f3})

    df: pd.DataFrame = pd.DataFrame(records)
    df[["set", "word"]] = df["label"].str.split(pat="_", n=1, expand=True)
    df["set"] = df["set"].str.replace(r":\d+", "", regex=True)
    df["set"] = df["set"].str.replace(r"^2(?=[A-Za-z])", "", regex=True)
    df["word"] = df["word"].str.replace(r":\d+", "", regex=True)
    df.to_csv(dir / f"{session}_formants.csv", index=False)

def confidence_ellipse(ax: Axes, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64],
                       n_std: float = 2.0, **kwargs) -> Ellipse | None:
    """
    Draw a covariance-based confidence ellipse.

    n_std: number of standard deviations (2.0 ≈ 95% confidence for bivariate normal)
    """
    if len(x) < 3:
        return None

    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)

    # Eigenvalue decomposition for ellipse orientation and size
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Angle of rotation (first eigenvector)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Width and height (2 * n_std * sqrt(eigenvalue))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse((mean_x, mean_y), width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def draw_vowel_quadrilateral(ax: Axes, std_df: pd.DataFrame) -> None:
    """
    Draw the IPA vowel quadrilateral connecting cardinal vowels.
    This provides a reference frame for the vowel space.
    """
    # Cardinal vowels forming the quadrilateral (approximate positions)
    # Top-left: [i], Top-right: [u], Bottom-left: [a], Bottom-right: [ɑ/ɒ]
    cardinals = {
        '[i]': None, '[u]': None, '[a]': None, '[ɑ]': None, '[ɒ]': None,
        '[e]': None, '[o]': None, '[ɛ]': None, '[ɔ]': None, '[æ]': None
    }

    for _, row in std_df.iterrows():
        label = row['label']
        if label in cardinals:
            cardinals[label] = (row['F2'], row['F1'])

    # Draw the outer quadrilateral if we have the corner vowels
    corners = []
    for v in ['[i]', '[a]', '[ɒ]', '[u]']:
        if v == '[ɒ]' and cardinals.get('[ɒ]') is None:
            v = '[ɑ]'  # fallback
        if cardinals.get(v):
            corners.append(cardinals[v])

    if len(corners) == 4:
        quad = Polygon(corners, fill=False, edgecolor='#cccccc',
                      linewidth=1.5, linestyle='-', zorder=0, closed=True)
        ax.add_patch(quad)

    # Draw horizontal lines for vowel height reference
    height_lines = [
        ('[i]', '[u]'),      # close
        ('[e]', '[o]'),      # close-mid
        ('[ɛ]', '[ɔ]'),      # open-mid
    ]
    for left, right in height_lines:
        if cardinals.get(left) and cardinals.get(right):
            ax.plot([cardinals[left][0], cardinals[right][0]],
                   [cardinals[left][1], cardinals[right][1]],
                   color='#e0e0e0', linewidth=1, linestyle='--', zorder=0)


def plot_vowel_space(dir: Path, session: str, show_diphthongs: bool = True) -> None:
    """
    Create a clean vowel space plot showing all tokens with confidence ellipses.
    """
    output_path = dir / f"{session}_vowel_space.png"
    df = pd.read_csv(dir / f"{session}_formants.csv")

    df['is_diphthong'] = df['label'].str.contains(':')
    if not show_diphthongs:
        df = df[~df['is_diphthong']]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    # Load and draw reference vowels
    std_path = Path(__file__).parent / 'standard.csv'
    if std_path.exists():
        std_df = pd.read_csv(std_path)
        std_df.columns = std_df.columns.str.strip()
        std_df = std_df.dropna(subset=['F1', 'F2'])

        draw_vowel_quadrilateral(ax, std_df)

        # Plot reference IPA symbols in light gray
        for _, row in std_df.iterrows():
            ax.annotate(row['label'], xy=(row['F2'], row['F1']),
                       fontsize=11, ha='center', va='center',
                       color='#b0b0b0', zorder=1, fontweight='bold')

        # Invisible points to set axis limits
        ax.scatter(std_df['F2'], std_df['F1'], alpha=0)

    # Plot each lexical set with confidence ellipse
    for lexical_set in sorted(df['set'].unique()):
        subset = df[df['set'] == lexical_set]
        color = get_color(lexical_set)

        # Plot individual tokens
        ax.scatter(subset['F2'], subset['F1'],
                  c=color, s=50, alpha=0.6, edgecolors='white',
                  linewidths=0.5, zorder=3)

        # Draw confidence ellipse (95%)
        if len(subset) >= 3:
            confidence_ellipse(ax, subset['F2'].values, subset['F1'].values,
                             n_std=2.0, facecolor=color, alpha=0.15,
                             edgecolor=color, linewidth=1.5, zorder=2)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.margins(0.08)

    ax.set_xlabel('F2 (Hz)', fontsize=12)
    ax.set_ylabel('F1 (Hz)', fontsize=12)
    ax.set_title(f'Vowel Space — {session}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', color='#e0e0e0', zorder=0)

    # Minimal spine styling
    for spine in ax.spines.values():
        spine.set_color('#cccccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    

def plot_single_lexical_set(dir: Path, session: str, lexical_set: str) -> None:
    """
    Detailed view of a single lexical set with all tokens labeled.
    """
    target_set = lexical_set.upper()
    output_path = dir / f"{session}_{target_set}.png"
    df = pd.read_csv(dir / f"{session}_formants.csv")

    df['is_diphthong'] = df['label'].str.contains(':')
    subset = df[df['set'] == target_set].copy()
    if subset.empty:
        print(f"No data for lexical set '{target_set}'")
        return

    color = get_color(target_set)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    # Load reference vowels for context
    std_path = Path(__file__).parent / 'standard.csv'
    if std_path.exists():
        std_df = pd.read_csv(std_path)
        std_df.columns = std_df.columns.str.strip()
        std_df = std_df.dropna(subset=['F1', 'F2'])
        draw_vowel_quadrilateral(ax, std_df)
        for _, row in std_df.iterrows():
            ax.annotate(row['label'], xy=(row['F2'], row['F1']),
                       fontsize=10, ha='center', va='center',
                       color='#c0c0c0', zorder=1)
        ax.scatter(std_df['F2'], std_df['F1'], alpha=0)

    # Plot tokens
    mono = subset[~subset['is_diphthong']]
    if not mono.empty:
        ax.scatter(mono['F2'], mono['F1'], c=color, s=100, alpha=0.8,
                  edgecolors='white', linewidths=1, zorder=4)

        # Confidence ellipse
        if len(mono) >= 3:
            confidence_ellipse(ax, mono['F2'].values, mono['F1'].values,
                             n_std=2.0, facecolor=color, alpha=0.2,
                             edgecolor=color, linewidth=2, zorder=2)

        # Label each word
        for _, row in mono.iterrows():
            ax.annotate(row['word'], xy=(row['F2'], row['F1']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color='#333333', zorder=5)

    # Plot mean with larger marker
    if not mono.empty:
        mean_f1, mean_f2 = mono['F1'].mean(), mono['F2'].mean()
        ax.scatter([mean_f2], [mean_f1], c=color, s=250, marker='*',
                  edgecolors='white', linewidths=1.5, zorder=5)
        ax.annotate(f'{target_set}\n(mean)', xy=(mean_f2, mean_f1),
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=10, fontweight='bold', color=color, zorder=5)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.margins(0.1)

    ax.set_xlabel('F2 (Hz)', fontsize=12)
    ax.set_ylabel('F1 (Hz)', fontsize=12)
    ax.set_title(f'{target_set} vowel — {session}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', color='#e0e0e0', zorder=0)

    for spine in ax.spines.values():
        spine.set_color('#cccccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_vowel_means(dir: Path, session: str) -> None:
    """
    Clean plot showing mean F1/F2 for each lexical set with confidence ellipses.
    This is the primary summary visualization.
    """
    output_path = dir / f"{session}_means.png"
    df = pd.read_csv(dir / f"{session}_formants.csv")

    df['is_diphthong'] = df['label'].str.contains(':')
    mono_df = df[~df['is_diphthong']]

    # Calculate means and standard deviations
    means = mono_df.groupby('set').agg({
        'F1': ['mean', 'std', 'count'],
        'F2': ['mean', 'std']
    }).reset_index()
    means.columns = ['set', 'F1_mean', 'F1_std', 'n', 'F2_mean', 'F2_std']

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    # Load and draw reference vowels
    std_path = Path(__file__).parent / 'standard.csv'
    if std_path.exists():
        std_df = pd.read_csv(std_path)
        std_df.columns = std_df.columns.str.strip()
        std_df = std_df.dropna(subset=['F1', 'F2'])
        draw_vowel_quadrilateral(ax, std_df)
        for _, row in std_df.iterrows():
            ax.annotate(row['label'], xy=(row['F2'], row['F1']),
                       fontsize=11, ha='center', va='center',
                       color='#b0b0b0', zorder=1, fontweight='bold')
        ax.scatter(std_df['F2'], std_df['F1'], alpha=0)

    # Plot confidence ellipses for each set, then means
    for lexical_set in sorted(mono_df['set'].unique()):
        subset = mono_df[mono_df['set'] == lexical_set]
        color = get_color(lexical_set)

        if len(subset) >= 3:
            confidence_ellipse(ax, subset['F2'].values, subset['F1'].values,
                             n_std=1.5, facecolor=color, alpha=0.2,
                             edgecolor=color, linewidth=1.5, zorder=2)

    # Plot means as points with labels
    for _, row in means.iterrows():
        color = get_color(row['set'])
        ax.scatter(row['F2_mean'], row['F1_mean'], c=color, s=120,
                  edgecolors='white', linewidths=1.5, zorder=4)
        ax.annotate(row['set'], xy=(row['F2_mean'], row['F1_mean']),
                   xytext=(6, 4), textcoords='offset points',
                   fontsize=9, fontweight='bold', color=color, zorder=5)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.margins(0.08)

    ax.set_xlabel('F2 (Hz)', fontsize=12)
    ax.set_ylabel('F1 (Hz)', fontsize=12)
    ax.set_title(f'Vowel Means — {session}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', color='#e0e0e0', zorder=0)

    for spine in ax.spines.values():
        spine.set_color('#cccccc')

    # Add a legend grouping vowels by category
    legend_elements = []
    categories = [
        ('High front', ['FLEECE', 'KIT', 'HAPPY']),
        ('High back', ['GOOSE', 'FOOT']),
        ('Mid front', ['FACE', 'DRESS']),
        ('Mid back', ['GOAT', 'THOUGHT', 'FORCE', 'NORTH', 'CLOTH']),
        ('Low', ['TRAP', 'BATH', 'PALM', 'LOT', 'START', 'STRUT']),
        ('Reduced', ['COMMA', 'LETTER', 'NURSE']),
    ]
    present_sets = set(means['set'].values)
    for cat_name, sets in categories:
        present = [s for s in sets if s in present_sets]
        if present:
            # Use first color as representative
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=get_color(present[0]),
                      markersize=8, label=cat_name)
            )

    ax.legend(handles=legend_elements, loc='upper left', frameon=True,
             fontsize=9, title='Categories')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_vowel_comparison(dir: Path, session: str) -> None:
    """
    Grid comparison of vowel categories: front vs back, high vs low.
    """
    output_path = dir / f"{session}_comparison.png"
    df = pd.read_csv(dir / f"{session}_formants.csv")
    df['is_diphthong'] = df['label'].str.contains(':')
    mono_df = df[~df['is_diphthong']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('white')

    # Load reference vowels
    std_path = Path(__file__).parent / 'standard.csv'
    std_df = None
    if std_path.exists():
        std_df = pd.read_csv(std_path)
        std_df.columns = std_df.columns.str.strip()
        std_df = std_df.dropna(subset=['F1', 'F2'])

    comparisons = [
        ('High vowels', ['FLEECE', 'KIT', 'HAPPY', 'GOOSE', 'FOOT']),
        ('Low vowels', ['TRAP', 'BATH', 'PALM', 'LOT', 'START', 'STRUT']),
        ('Front vowels', ['FLEECE', 'KIT', 'FACE', 'DRESS', 'TRAP', 'BATH']),
        ('Back vowels', ['GOOSE', 'FOOT', 'GOAT', 'THOUGHT', 'LOT', 'CLOTH']),
    ]

    for ax, (title, sets) in zip(axes.flat, comparisons):
        ax.set_facecolor('#fafafa')

        if std_df is not None:
            draw_vowel_quadrilateral(ax, std_df)
            for _, row in std_df.iterrows():
                ax.annotate(row['label'], xy=(row['F2'], row['F1']),
                           fontsize=9, ha='center', va='center',
                           color='#d0d0d0', zorder=1)
            ax.scatter(std_df['F2'], std_df['F1'], alpha=0)

        for lexical_set in sets:
            subset = mono_df[mono_df['set'] == lexical_set]
            if subset.empty:
                continue
            color = get_color(lexical_set)

            ax.scatter(subset['F2'], subset['F1'], c=color, s=40, alpha=0.5,
                      edgecolors='white', linewidths=0.5, zorder=3)

            if len(subset) >= 3:
                confidence_ellipse(ax, subset['F2'].values, subset['F1'].values,
                                 n_std=1.5, facecolor=color, alpha=0.15,
                                 edgecolor=color, linewidth=1.5, zorder=2)

            # Plot mean
            mean_f2, mean_f1 = subset['F2'].mean(), subset['F1'].mean()
            ax.scatter([mean_f2], [mean_f1], c=color, s=80,
                      edgecolors='white', linewidths=1, zorder=4)
            ax.annotate(lexical_set, xy=(mean_f2, mean_f1),
                       xytext=(4, 3), textcoords='offset points',
                       fontsize=8, fontweight='bold', color=color, zorder=5)

        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.margins(0.1)
        ax.set_xlabel('F2 (Hz)', fontsize=10)
        ax.set_ylabel('F1 (Hz)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', color='#e0e0e0', zorder=0)
        for spine in ax.spines.values():
            spine.set_color('#cccccc')

    plt.suptitle(f'Vowel Comparisons — {session}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def run(session: str,
        Gender: Literal['M', 'F', 'C'] = 'M',
        target_lexical_set: str = None,
        show_diphthongs: bool = False) -> None:
    dir: Path = Path(__file__).parent / "sessions" / session

    # Extract formants
    extract_formants(dir, session, Gender=Gender)

    # Generate visualizations
    plot_vowel_space(dir, session, show_diphthongs=show_diphthongs)
    plot_vowel_means(dir, session)
    plot_vowel_comparison(dir, session)

    # If a specific lexical set is requested, generate detailed view
    if target_lexical_set:
        plot_single_lexical_set(dir, session, target_lexical_set)


def main() -> None:
    Fire(run)


if __name__ == "__main__":
    main()