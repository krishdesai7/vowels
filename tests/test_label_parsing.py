import polars as pl

from vowels import parse_labels


def test_simple_label() -> None:
    df: pl.LazyFrame = parse_labels(pl.LazyFrame({"label": ["CURE_your"]}))
    assert df.select("set", "word").collect().row(0) == ("CURE", "your")


def test_disyllabic_prefix_stripped() -> None:
    df: pl.LazyFrame = parse_labels(
        pl.LazyFrame({"label": ["2coMMA_comma", "2haPPY_coffee", "2leTTER_butter"]})
    )
    actual: list[tuple[str, str]] = df.select("set", "word").collect().rows()
    assert actual == [("coMMA", "comma"), ("haPPY", "coffee"), ("leTTER", "butter")]


def test_diphthong_suffix_stripped() -> None:
    df: pl.LazyFrame = parse_labels(
        pl.LazyFrame({"label": ["PRICE_try:1", "PRICE_try:2"]})
    )
    actual: list[tuple[str, str]] = df.select("set", "word").collect().rows()
    assert actual == [("PRICE", "try"), ("PRICE", "try")]


def test_multiple_labels() -> None:
    labels: list[str] = [
        "FLEECE_beat",
        "TRAP_bad",
        "2coMMA_sofa",
        "PRICE_try:1",
        "PRICE_try:2",
    ]
    df: pl.LazyFrame = parse_labels(pl.LazyFrame({"label": labels}))
    actual: list[tuple[str, str]] = df.select("set", "word").collect().rows()
    assert actual == [
        ("FLEECE", "beat"),
        ("TRAP", "bad"),
        ("coMMA", "sofa"),
        ("PRICE", "try"),
        ("PRICE", "try"),
    ]
