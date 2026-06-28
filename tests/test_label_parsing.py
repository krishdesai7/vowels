import polars as pl

from vowels import parse_labels


def _parse(labels: list[str]) -> pl.LazyFrame:
    return parse_labels(pl.LazyFrame({"label": labels}))


def test_simple_label() -> None:
    df: pl.LazyFrame = _parse(["CURE_your"])
    assert df.select(pl.col("set")).unique().collect().get_column("set").to_list() == [
        "CURE"
    ]
    assert df.select(pl.col("word")).unique().collect().get_column(
        "word"
    ).to_list() == ["your"]


def test_disyllabic_prefix_stripped() -> None:
    df: pl.LazyFrame = _parse(["2coMMA_comma", "2haPPY_coffee", "2leTTER_butter"])
    assert df.select(pl.col("set")).unique().collect().get_column("set").to_list() == [
        "coMMA",
        "haPPY",
        "leTTER",
    ]
    assert df.select(pl.col("word")).unique().collect().get_column(
        "word"
    ).to_list() == ["comma", "coffee", "butter"]


def test_diphthong_suffix_stripped() -> None:
    df: pl.LazyFrame = _parse(["PRICE_try:1", "PRICE_try:2"])
    assert df.select(pl.col("set")).unique().collect().get_column("set").to_list() == [
        "PRICE",
        "PRICE",
    ]
    assert df.select(pl.col("word")).unique().collect().get_column(
        "word"
    ).to_list() == ["try", "try"]


def test_set_uppercased() -> None:
    df: pl.LazyFrame = _parse(["fleece_beat"])
    assert df.select(pl.col("set")).unique().collect().get_column("set").to_list() == [
        "FLEECE"
    ]


def test_multiple_labels() -> None:
    labels: list[str] = [
        "FLEECE_beat",
        "TRAP_bad",
        "2coMMA_sofa",
        "PRICE_try:1",
        "PRICE_try:2",
    ]
    df: pl.LazyFrame = _parse(labels)
    assert df.select(pl.col("set")).unique().collect().get_column("set").to_list() == [
        "FLEECE",
        "TRAP",
        "coMMA",
        "PRICE",
        "PRICE",
    ]
    assert df.select(pl.col("word")).unique().collect().get_column(
        "word"
    ).to_list() == ["beat", "bad", "sofa", "try", "try"]
