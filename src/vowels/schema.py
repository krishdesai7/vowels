from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping


class Wells(StrEnum):
    """A Wells standard lexical set.

    The value is the set's own name, so a member is interchangeable with the
    label text built from `labels.csv`. Presentation (colour) and dialectal
    realisation (IPA) are looked up separately -- see `Wells.color` and
    `DialectProfile.expected` -- rather than overloaded onto the value.
    """

    # --- MONOPHTHONGS ---
    FLEECE = "FLEECE"
    KIT = "KIT"
    haPPY = "haPPY"
    DRESS = "DRESS"
    TRAP = "TRAP"
    BATH = "BATH"
    PALM = "PALM"
    LOT = "LOT"
    THOUGHT = "THOUGHT"
    CLOTH = "CLOTH"
    FOOT = "FOOT"
    GOOSE = "GOOSE"
    STRUT = "STRUT"
    START = "START"
    NORTH = "NORTH"
    FORCE = "FORCE"
    NURSE = "NURSE"
    coMMA = "coMMA"
    leTTER = "leTTER"

    # --- DIPHTHONGS ---
    FACE = "FACE"
    GOAT = "GOAT"
    PRICE = "PRICE"
    MOUTH = "MOUTH"
    CHOICE = "CHOICE"
    NEAR = "NEAR"
    SQUARE = "SQUARE"
    CURE = "CURE"

    @property
    def color(self) -> str:
        """The set's plotting colour."""
        return COLORS[self]


# A 21-colour qualitative spectrum for the monophthongs plus a distinct
# 6-colour spectrum for the diphthongs. Grouped to maximise contrast within
# each region of the vowel space (notably cot/caught and horse/hoarse).
COLORS: Final[Mapping[Wells, str]] = {
    # High Front
    Wells.FLEECE: "#d62728",  # Red
    Wells.KIT: "#1f77b4",  # Blue
    Wells.haPPY: "#7b4173",  # Dark Purple
    # Mid Front
    Wells.FACE: "#2ca02c",  # Green
    Wells.DRESS: "#ff7f0e",  # Orange
    # Low Front/Central
    Wells.TRAP: "#9467bd",  # Purple
    Wells.BATH: "#8c564b",  # Brown
    Wells.PALM: "#e377c2",  # Pink
    # Low Back
    Wells.LOT: "#17becf",  # Cyan
    Wells.THOUGHT: "#bcbd22",  # Olive
    Wells.CLOTH: "#f7b6d2",  # Light Pink
    # Mid/High Back
    Wells.FOOT: "#9edae5",  # Light Blue
    Wells.GOOSE: "#c5b0d5",  # Light Purple
    Wells.GOAT: "#c49c94",  # Light Brown
    # Central
    Wells.STRUT: "#ffbb78",  # Light Orange
    Wells.START: "#7f7f7f",  # Grey
    # R-Colored
    Wells.NORTH: "#dbdb8d",  # Khaki
    Wells.FORCE: "#393b79",  # Dark Blue
    # Schwa/Reduced
    Wells.NURSE: "#637939",  # Dark Green
    Wells.coMMA: "#8c6d31",  # Dark Yellow
    Wells.leTTER: "#843c39",  # Dark Red
    # Diphthongs
    Wells.PRICE: "#e41a1c",  # Crimson
    Wells.MOUTH: "#377eb8",  # Cobalt
    Wells.CHOICE: "#4daf4a",  # Emerald
    Wells.NEAR: "#984ea3",  # Amethyst
    Wells.SQUARE: "#ff7f00",  # Tangerine
    Wells.CURE: "#e6ab02",  # Goldenrod
}


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


@dataclass(frozen=True, slots=True)
class DialectProfile:
    """How one dialect realises the lexical sets.

    `diphthongs` is the classifier's PRIOR: the sets this dialect is expected
    to realise with a gliding nucleus. `r_colored` are sets realised with a
    following rhotic; their F3 falls sharply across the vowel, which the
    Bark `Roundness` dimension reads as large spectral displacement, so they
    are not representative of a plain monophthong and are excluded from the
    speaker baseline.
    """

    rhotic: bool
    expected: Mapping[Wells, str]
    diphthongs: frozenset[Wells]
    r_colored: frozenset[Wells]


class Dialect(StrEnum):
    GA = auto()
    RP = auto()

    @property
    def profile(self) -> DialectProfile:
        return PROFILES[self]


DEFAULT_DIALECT: Final[Dialect] = Dialect.GA

_RP_EXPECTED: Final[Mapping[Wells, str]] = {
    Wells.KIT: "ɪ",
    Wells.DRESS: "e",
    Wells.TRAP: "æ",
    Wells.LOT: "ɒ",
    Wells.STRUT: "ʌ",
    Wells.FOOT: "ʊ",
    Wells.BATH: "ɑː",
    Wells.CLOTH: "ɒ",
    Wells.NURSE: "ɜː",
    Wells.FLEECE: "iː",
    Wells.FACE: "eɪ",
    Wells.PALM: "ɑː",
    Wells.THOUGHT: "ɔː",
    Wells.GOAT: "əʊ",
    Wells.GOOSE: "uː",
    Wells.PRICE: "aɪ",
    Wells.CHOICE: "ɔɪ",
    Wells.MOUTH: "aʊ",
    Wells.NEAR: "ɪə",
    Wells.SQUARE: "ɛə",
    Wells.START: "ɑː",
    Wells.NORTH: "ɔː",
    Wells.FORCE: "ɔː",
    Wells.CURE: "ʊə",
    Wells.haPPY: "ɪ",
    Wells.leTTER: "ə",
    Wells.coMMA: "ə",
}

_GA_EXPECTED: Final[Mapping[Wells, str]] = {
    Wells.KIT: "ɪ",
    Wells.DRESS: "ɛ",
    Wells.TRAP: "æ",
    Wells.LOT: "ɑ",
    Wells.STRUT: "ʌ",
    Wells.FOOT: "ʊ",
    Wells.BATH: "æ",
    Wells.CLOTH: "ɔ",
    Wells.NURSE: "ɜr",
    Wells.FLEECE: "i",
    Wells.FACE: "eɪ",
    Wells.PALM: "ɑ",
    Wells.THOUGHT: "ɔ",
    Wells.GOAT: "oʊ",
    Wells.GOOSE: "u",
    Wells.PRICE: "aɪ",
    Wells.CHOICE: "ɔɪ",
    Wells.MOUTH: "aʊ",
    Wells.NEAR: "ɪr",
    Wells.SQUARE: "ɛr",
    Wells.START: "ɑr",
    Wells.NORTH: "ɔr",
    Wells.FORCE: "or",
    Wells.CURE: "ʊr",
    Wells.haPPY: "ɪ",
    Wells.leTTER: "ər",
    Wells.coMMA: "ə",
}

PROFILES: Final[Mapping[Dialect, DialectProfile]] = {
    Dialect.RP: DialectProfile(
        rhotic=False,
        expected=_RP_EXPECTED,
        # RP's NEAR/SQUARE/CURE are centring diphthongs (ɪə, ɛə, ʊə): the
        # historical /r/ survives as a schwa offglide, so they genuinely glide.
        diphthongs=frozenset(
            {
                Wells.FACE,
                Wells.GOAT,
                Wells.PRICE,
                Wells.CHOICE,
                Wells.MOUTH,
                Wells.NEAR,
                Wells.SQUARE,
                Wells.CURE,
            }
        ),
        r_colored=frozenset(),
    ),
    Dialect.GA: DialectProfile(
        rhotic=True,
        expected=_GA_EXPECTED,
        # GA realises NEAR/SQUARE/CURE as vowel + rhotic (ɪr, ɛr, ʊr), not as
        # centring diphthongs, so they are not part of the gliding prior.
        diphthongs=frozenset(
            {
                Wells.FACE,
                Wells.GOAT,
                Wells.PRICE,
                Wells.CHOICE,
                Wells.MOUTH,
            }
        ),
        r_colored=frozenset(
            {
                Wells.NURSE,
                Wells.NEAR,
                Wells.SQUARE,
                Wells.START,
                Wells.NORTH,
                Wells.FORCE,
                Wells.CURE,
                Wells.leTTER,
            }
        ),
    ),
}


# The canonical Wells classification, retained as the dialect-agnostic default.
DIPHTHONGS: Final[set[Wells]] = set(PROFILES[Dialect.RP].diphthongs)


class Gender(StrEnum):
    M = auto()
    F = auto()
    C = auto()
