from enum import StrEnum, auto
from typing import Final


class Wells(StrEnum):
    # --- MONOPHTHONGS ---
    # Distributed across a full 21-color qualitative spectrum

    # High Front (High Contrast)
    FLEECE = "#d62728"  # Red
    KIT = "#1f77b4"  # Blue
    haPPY = "#7b4173"  # Dark Purple

    # Mid Front
    FACE = "#2ca02c"  # Green
    DRESS = "#ff7f0e"  # Orange

    # Low Front/Central
    TRAP = "#9467bd"  # Purple
    BATH = "#8c564b"  # Brown
    PALM = "#e377c2"  # Pink

    # Low Back (Max contrast for cot/caught)
    LOT = "#17becf"  # Cyan
    THOUGHT = "#bcbd22"  # Olive
    CLOTH = "#f7b6d2"  # Light Pink

    # Mid/High Back
    FOOT = "#9edae5"  # Light Blue
    GOOSE = "#c5b0d5"  # Light Purple
    GOAT = "#c49c94"  # Light Brown

    # Central
    STRUT = "#ffbb78"  # Light Orange
    START = "#7f7f7f"  # Grey

    # R-Colored (Max contrast for horse/hoarse)
    NORTH = "#dbdb8d"  # Khaki
    FORCE = "#393b79"  # Dark Blue

    # Schwa/Reduced
    NURSE = "#637939"  # Dark Green
    coMMA = "#8c6d31"  # Dark Yellow
    leTTER = "#843c39"  # Dark Red

    # --- DIPHTHONGS ---
    # Distributed across a distinct 6-color spectrum

    PRICE = "#e41a1c"  # Crimson
    MOUTH = "#377eb8"  # Cobalt
    CHOICE = "#4daf4a"  # Emerald
    NEAR = "#984ea3"  # Amethyst
    SQUARE = "#ff7f00"  # Tangerine
    CURE = "#e6ab02"  # Goldenrod


GROUPS: Final[dict[str, list[str]]] = {
    "High Front": ["FLEECE", "KIT", "haPPY"],
    "Mid Front": ["DRESS"],
    "Low Front/Central": ["TRAP", "BATH", "PALM"],
    "Low Back": ["LOT", "THOUGHT", "CLOTH"],
    "Mid/High Back": ["FOOT", "GOOSE"],
    "Central": ["STRUT", "START"],
    "R-Colored": ["NORTH", "FORCE"],
    "Schwa/Reduced": ["NURSE", "coMMA", "leTTER"],
    "Diphthongs": [
        "FACE",
        "GOAT",
        "PRICE",
        "CHOICE",
        "MOUTH",
        "NEAR",
        "SQUARE",
        "CURE",
    ],
}

DIPHTHONGS: Final[set[Wells]] = {
    Wells.FACE,
    Wells.GOAT,
    Wells.PRICE,
    Wells.CHOICE,
    Wells.MOUTH,
    Wells.NEAR,
    Wells.SQUARE,
    Wells.CURE,
}


class Gender(StrEnum):
    M = auto()
    F = auto()
    C = auto()
