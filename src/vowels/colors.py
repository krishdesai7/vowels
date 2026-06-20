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
    return VOWEL_COLORS.get(lexical_set.upper(), "#404040")
