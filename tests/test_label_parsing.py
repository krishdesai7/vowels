import polars as pl

from vowels import parse_labels


def _parse(labels: list[str]) -> pl.DataFrame:
    return parse_labels(pl.LazyFrame({"label": labels})).collect()


def test_simple_label() -> None:
    df = _parse(["CURE_your"])
    assert df["set"].to_list() == ["CURE"]
    assert df["word"].to_list() == ["your"]


def test_disyllabic_prefix_stripped() -> None:
    df = _parse(["2coMMA_comma", "2haPPY_coffee", "2leTTER_butter"])
    assert df["set"].to_list() == ["coMMA", "haPPY", "leTTER"]
    assert df["word"].to_list() == ["comma", "coffee", "butter"]


def test_diphthong_suffix_stripped() -> None:
    df = _parse(["PRICE_try:1", "PRICE_try:2"])
    assert df["set"].to_list() == ["PRICE", "PRICE"]
    assert df["word"].to_list() == ["try", "try"]


def test_multiple_labels() -> None:
    labels = [
        "FLEECE_beat",
        "TRAP_bad",
        "2coMMA_sofa",
        "PRICE_try:1",
        "PRICE_try:2",
    ]
    df = _parse(labels)
    assert df["set"].to_list() == ["FLEECE", "TRAP", "coMMA", "PRICE", "PRICE"]
    assert df["word"].to_list() == ["beat", "bad", "sofa", "try", "try"]
