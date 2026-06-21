from pathlib import Path

import altair as alt
import polars as pl

from .. import Mode, Wells, project_root, session_dir
from . import precompute_ellipse


def build_chart(
    session: str, mode: Mode = Mode.MONO
) -> alt.LayerChart | alt.FacetChart:
    d: Path = session_dir(session)
    df: pl.DataFrame = pl.read_csv(d / f"{session}_formants.csv")

    is_diph: pl.Expr = pl.col("label").str.contains(":")
    if mode == Mode.MONO:
        df = df.filter(~is_diph)
    elif mode == Mode.DIPH:
        df = df.filter(is_diph)
    # "all" keeps everything

    mono_df: pl.DataFrame = df.filter(~pl.col("label").str.contains(":"))
    diph_df: pl.DataFrame = df.filter(pl.col("label").str.contains(":"))

    all_sets: list[Wells] = sorted(df["set"].unique().to_list())
    color_scale: alt.Scale = alt.Scale(
        domain=all_sets,
        range=[s.value for s in all_sets],
    )

    legend_sel: alt.Parameter = alt.selection_point(fields=["set"], bind="legend")

    # Layer 1: IPA reference text
    std_path: Path = project_root() / "standard.csv"
    std_df: pl.DataFrame = pl.read_csv(std_path).drop_nulls(subset=["F1", "F2"])
    ref_layer: alt.Chart = (
        alt.Chart(std_df)
        .mark_text(color="#c0c0c0", fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("F2:Q", scale=alt.Scale(reverse=True), title="F2 (Hz)"),
            y=alt.Y("F1:Q", scale=alt.Scale(reverse=True), title="F1 (Hz)"),
            text="label:N",
        )
    )

    # Layer 2: Confidence ellipses (monophthong tokens only)
    ellipse_records: list[dict[str, float]] = []
    for s in sorted(mono_df["set"].unique().to_list()):
        subset: pl.DataFrame = mono_df.filter(pl.col("set") == s)
        if len(subset) < 3:
            continue
        pts: list[dict[str, float]] | None = precompute_ellipse(
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
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True), axis=None),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True), axis=None),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                detail="set:N",
                order=alt.Order("idx:O"),
                opacity=alt.condition(legend_sel, alt.value(0.7), alt.value(0.15)),
            )
        )
        layers.append(ellipse_layer)

    # Layer 3: Monophthong tokens
    token_layer: alt.Chart = (
        alt.Chart(mono_df)
        .mark_circle(size=60, stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("F2:Q", scale=alt.Scale(reverse=True), title="F2 (Hz)"),
            y=alt.Y("F1:Q", scale=alt.Scale(reverse=True), title="F1 (Hz)"),
            color=alt.Color(
                "set:N",
                scale=color_scale,
                legend=alt.Legend(title="Lexical set"),
            ),
            opacity=alt.condition(legend_sel, alt.value(0.85), alt.value(0.2)),
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

    # Layer 4: Per-set means
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
            x=alt.X("F2:Q", scale=alt.Scale(reverse=True), axis=None),
            y=alt.Y("F1:Q", scale=alt.Scale(reverse=True), axis=None),
            color=alt.Color("set:N", scale=color_scale, legend=None),
            opacity=alt.condition(legend_sel, alt.value(1.0), alt.value(0.3)),
            tooltip=[
                alt.Tooltip("set:N", title="Set"),
                alt.Tooltip("F1:Q", title="F1 mean (Hz)", format=".0f"),
                alt.Tooltip("F2:Q", title="F2 mean (Hz)", format=".0f"),
            ],
        )
    )
    layers.append(means_layer)

    # Layer 5: Diphthong tokens (toggle-controlled)
    if len(diph_df) > 0:
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
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True), axis=None),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True), axis=None),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                opacity=alt.condition(legend_sel, alt.value(0.65), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("word:N", title="Word"),
                    alt.Tooltip("set:N", title="Set"),
                    alt.Tooltip("F1:Q", title="F1 (Hz)", format=".0f"),
                    alt.Tooltip("F2:Q", title="F2 (Hz)", format=".0f"),
                ],
            )
        )
        layers.append(diph_layer)

    return (
        alt.layer(*layers)
        .add_params(legend_sel)
        .resolve_scale(x="shared", y="shared")
        .properties(width=650, height=550, title=f"Vowel Space — {session}")
    )


def save_chart(session: str, mode: Mode = Mode.MONO) -> None:
    d: Path = session_dir(session)
    out_path: Path = d / f"{session}_vowel_space.html"
    build_chart(session, mode).save(str(out_path))
    print(f"Created {out_path}")
