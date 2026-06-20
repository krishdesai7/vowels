import polars as pl

from vowels.pipeline.formants import parse_labels


def _parse(labels: list[str]) -> pl.DataFrame:
    return parse_labels(pl.DataFrame({"label": labels}))


def test_simple_label():
    df = _parse(["CURE_your"])
    assert df["set"].to_list() == ["CURE"]
    assert df["word"].to_list() == ["your"]


def test_disyllabic_prefix_stripped():
    df = _parse(["2COMMA_comma", "2HAPPY_coffee", "2LETTER_butter"])
    assert df["set"].to_list() == ["COMMA", "HAPPY", "LETTER"]
    assert df["word"].to_list() == ["comma", "coffee", "butter"]


def test_diphthong_suffix_stripped():
    df = _parse(["PRICE_try:1", "PRICE_try:2"])
    assert df["set"].to_list() == ["PRICE", "PRICE"]
    assert df["word"].to_list() == ["try", "try"]


def test_set_uppercased():
    df = _parse(["fleece_beat"])
    assert df["set"].to_list() == ["FLEECE"]


def test_multiple_labels():
    labels = [
        "FLEECE_beat",
        "TRAP_bad",
        "2COMMA_sofa",
        "PRICE_try:1",
        "PRICE_try:2",
    ]
    df = _parse(labels)
    assert df["set"].to_list() == ["FLEECE", "TRAP", "COMMA", "PRICE", "PRICE"]
    assert df["word"].to_list() == ["beat", "bad", "sofa", "try", "try"]
