from pathlib import Path

import altair as alt
import polars as pl

from ..paths import project_root, session_dir
from ..schema import Wells
from . import precompute_ellipse


def build_chart(session: str) -> alt.VConcatChart:
    df: pl.DataFrame = pl.read_csv(session_dir(session) / f"{session}_formants.csv")

    is_diph: pl.Expr = pl.col("label").str.contains(":")
    mono_df: pl.DataFrame = df.filter(~is_diph)
    diph_df: pl.DataFrame = df.filter(is_diph)
    has_diph: bool = len(diph_df) > 0

    all_sets: list[str] = sorted(df["set"].unique().to_list())
    color_scale: alt.Scale = alt.Scale(
        domain=all_sets,
        range=[Wells[s].value for s in all_sets],
    )

    legend_sel: alt.Parameter = alt.selection_point(
        name="legend_sel", fields=["set"], bind="legend"
    )

    # Independent single-item selections — empty store = layer ON, item in store = layer OFF.
    words_sel: alt.Parameter = alt.selection_point(
        name="words_sel", fields=["label"], toggle=True, empty=False
    )
    means_sel: alt.Parameter = alt.selection_point(
        name="means_sel", fields=["label"], toggle=True, empty=False
    )
    mono_sel: alt.Parameter = alt.selection_point(
        name="mono_sel", fields=["label"], toggle=True, empty=False
    )
    diph_sel: alt.Parameter = alt.selection_point(
        name="diph_sel", fields=["label"], toggle=True, empty=False
    )

    # Vega expression fragments — negated: true = layer visible
    _words = '!vlSelectionTest("words_sel_store", {"label": "Words"})'
    _means = '!vlSelectionTest("means_sel_store", {"label": "Set averages"})'
    _mono = '!vlSelectionTest("mono_sel_store", {"label": "Monophthongs"})'
    _diph = '!vlSelectionTest("diph_sel_store", {"label": "Diphthongs"})'
    _legend = '!length(data("legend_sel_store")) || vlSelectionTest("legend_sel_store", datum)'

    mono_high: str = f"({_words}) && ({_mono}) && ({_legend})"
    mono_mid: str = f"({_words}) && ({_mono})"
    means_high: str = f"({_means}) && ({_mono}) && ({_legend})"
    means_mid: str = f"({_means}) && ({_mono})"
    ellipse_high: str = f"({_mono}) && ({_legend})"
    ellipse_mid: str = _mono
    diph_high: str = f"({_words}) && ({_diph}) && ({_legend})"
    diph_mid: str = f"({_words}) && ({_diph})"

    # Control panel helpers — transparent click_target on top so every click registers
    # on the selection regardless of whether the pointer is over text or the rect margin.
    def make_button(label: str, sel: alt.Parameter, width: int = 140) -> alt.LayerChart:
        data = pl.DataFrame([{"label": label}])
        bg = (
            alt.Chart(data)
            .mark_rect(cornerRadius=6)
            .encode(
                color=alt.condition(sel, alt.value("#c8d8e8"), alt.value("#2c7bb6"))
            )
        )
        txt = (
            alt.Chart(data)
            .mark_text(fontSize=13, fontWeight="bold")
            .encode(
                text="label:N",
                color=alt.condition(sel, alt.value("#5a7a99"), alt.value("white")),
            )
        )
        click_target = (
            alt.Chart(data).mark_rect(cornerRadius=6, opacity=0).add_params(sel)
        )
        return (bg + txt + click_target).properties(width=width, height=38)

    granularity_row: alt.HConcatChart = alt.hconcat(
        make_button("Words", words_sel),
        make_button("Set averages", means_sel, width=180),
        spacing=8,
    )
    type_buttons = [make_button("Monophthongs", mono_sel, width=180)]
    if has_diph:
        type_buttons.append(make_button("Diphthongs", diph_sel, width=160))
    type_row: alt.HConcatChart = alt.hconcat(*type_buttons, spacing=8)
    control_panel: alt.VConcatChart = alt.vconcat(granularity_row, type_row, spacing=6)

    # Layer 1: IPA reference text
    std_path: Path = project_root() / "male_standard.parquet"
    std_df: pl.DataFrame = pl.read_parquet(std_path).drop_nulls(subset=["F1", "F2"])
    ref_layer: alt.Chart = (
        alt.Chart(std_df)
        .mark_text(color="#c0c0c0", fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("F2:Q", scale=alt.Scale(reverse=True), title="F2 (Hz)"),
            y=alt.Y("F1:Q", scale=alt.Scale(reverse=True), title="F1 (Hz)"),
            text="label:N",
        )
    )

    # Layer 2: Confidence ellipses (monophthongs only)
    ellipse_records: list[dict] = []
    for s in sorted(mono_df["set"].unique().to_list()):
        subset: pl.DataFrame = mono_df.filter(pl.col("set") == s)
        if len(subset) < 3:
            continue
        pts: list[dict] | None = precompute_ellipse(
            subset["F2"].to_numpy(), subset["F1"].to_numpy()
        )
        if pts:
            for k, pt in enumerate(pts):
                ellipse_records.append(
                    {"set": s, "F2": pt["F2"], "F1": pt["F1"], "idx": k}
                )

    layers: list[alt.Chart] = [ref_layer]

    if ellipse_records:
        ellipse_df: pl.DataFrame = pl.DataFrame(ellipse_records)
        ellipse_layer: alt.Chart = (
            alt.Chart(ellipse_df)
            .mark_line(filled=True, fillOpacity=0.12, strokeWidth=1.5)
            .encode(
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                detail="set:N",
                order=alt.Order("idx:O"),
                opacity=(
                    alt.when(ellipse_high)
                    .then(alt.value(0.7))
                    .when(ellipse_mid)
                    .then(alt.value(0.15))
                    .otherwise(alt.value(0.0))
                ),
            )
        )
        layers.append(ellipse_layer)

    # Layer 3: Monophthong tokens
    token_layer: alt.Chart = (
        alt.Chart(mono_df)
        .add_params(legend_sel)
        .mark_circle(size=60, stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("F2:Q", scale=alt.Scale(reverse=True), title="F2 (Hz)"),
            y=alt.Y("F1:Q", scale=alt.Scale(reverse=True), title="F1 (Hz)"),
            color=alt.Color(
                "set:N", scale=color_scale, legend=alt.Legend(title="Lexical set")
            ),
            opacity=(
                alt.when(mono_high)
                .then(alt.value(0.85))
                .when(mono_mid)
                .then(alt.value(0.2))
                .otherwise(alt.value(0.0))
            ),
            tooltip=[
                alt.Tooltip("word:N", title="Word"),
                alt.Tooltip("set:N", title="Set"),
                alt.Tooltip("F1:Q", title="F1 (Hz)", format=".0f"),
                alt.Tooltip("F2:Q", title="F2 (Hz)", format=".0f"),
                alt.Tooltip("F3:Q", title="F3 (Hz)", format=".0f"),
            ],
        )
    )
    layers.append(token_layer)

    # Layer 4: Per-set means (monophthongs only)
    means_df: pl.DataFrame = mono_df.group_by("set").agg(
        pl.col("F1").mean().alias("F1"),
        pl.col("F2").mean().alias("F2"),
    )
    means_layer: alt.Chart = (
        alt.Chart(means_df)
        .mark_point(
            shape="diamond", size=200, filled=True, stroke="white", strokeWidth=1.5
        )
        .encode(
            x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
            y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
            color=alt.Color("set:N", scale=color_scale, legend=None),
            opacity=(
                alt.when(means_high)
                .then(alt.value(1.0))
                .when(means_mid)
                .then(alt.value(0.3))
                .otherwise(alt.value(0.0))
            ),
            tooltip=[
                alt.Tooltip("set:N", title="Set"),
                alt.Tooltip("F1:Q", title="F1 mean (Hz)", format=".0f"),
                alt.Tooltip("F2:Q", title="F2 mean (Hz)", format=".0f"),
            ],
        )
    )
    layers.append(means_layer)

    # Layer 5: Diphthong tokens
    if has_diph:
        diph_layer: alt.Chart = (
            alt.Chart(diph_df)
            .mark_point(
                shape="triangle-up",
                size=60,
                filled=True,
                stroke="white",
                strokeWidth=0.5,
            )
            .encode(
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                opacity=(
                    alt.when(diph_high)
                    .then(alt.value(0.65))
                    .when(diph_mid)
                    .then(alt.value(0.2))
                    .otherwise(alt.value(0.0))
                ),
                tooltip=[
                    alt.Tooltip("word:N", title="Word"),
                    alt.Tooltip("set:N", title="Set"),
                    alt.Tooltip("F1:Q", title="F1 (Hz)", format=".0f"),
                    alt.Tooltip("F2:Q", title="F2 (Hz)", format=".0f"),
                ],
            )
        )
        layers.append(diph_layer)

    main_chart: alt.LayerChart = (
        alt.layer(*layers)
        .resolve_scale(x="shared", y="shared")
        .properties(width=650, height=550, title=f"Vowel Space — {session}")
    )

    return alt.vconcat(control_panel, main_chart).configure_view(strokeWidth=0)


def save_chart(session: str) -> None:
    out_path: Path = session_dir(session) / f"{session}_vowel_space.html"
    build_chart(session).save(str(out_path))
    print(f"Created {out_path}")
