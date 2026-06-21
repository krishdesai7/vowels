import altair as alt
import polars as pl

from vowels.colors import VOWEL_COLORS
from vowels.paths import project_root, session_dir
from vowels.plots.ellipse import precompute_ellipse


def build_chart(session: str, mode: str = "mono") -> alt.LayerChart:
    d = session_dir(session)
    df = pl.read_csv(d / f"{session}_formants.csv")

    is_diph = pl.col("label").str.contains(":")
    if mode == "mono":
        df = df.filter(~is_diph)
    elif mode == "diphthongs":
        df = df.filter(is_diph)
    # "all" keeps everything

    mono_df = df.filter(~pl.col("label").str.contains(":"))
    diph_df = df.filter(pl.col("label").str.contains(":"))

    all_sets = sorted(df["set"].unique().to_list())
    color_scale = alt.Scale(
        domain=all_sets,
        range=[VOWEL_COLORS.get(s, "#404040") for s in all_sets],
    )

    legend_sel = alt.selection_point(fields=["set"], bind="legend")

    # Layer 1: IPA reference text
    std_path = project_root() / "standard.csv"
    std_df = pl.read_csv(std_path).drop_nulls(subset=["F1", "F2"])
    ref_layer = (
        alt.Chart(std_df)
        .mark_text(color="#c0c0c0", fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("F2:Q", scale=alt.Scale(reverse=True), title="F2 (Hz)"),
            y=alt.Y("F1:Q", scale=alt.Scale(reverse=True), title="F1 (Hz)"),
            text="label:N",
        )
    )

    # Layer 2: Confidence ellipses (monophthong tokens only)
    ellipse_records: list[dict] = []
    for s in sorted(mono_df["set"].unique().to_list()):
        subset = mono_df.filter(pl.col("set") == s)
        if len(subset) < 3:
            continue
        pts = precompute_ellipse(subset["F2"].to_numpy(), subset["F1"].to_numpy())
        if pts:
            for k, pt in enumerate(pts):
                ellipse_records.append({"set": s, "F2": pt["F2"], "F1": pt["F1"], "idx": k})

    layers: list[alt.Chart] = [ref_layer]

    if ellipse_records:
        ellipse_df = pl.DataFrame(ellipse_records)
        ellipse_layer = (
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
    token_layer = (
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
    means_df = mono_df.group_by("set").agg(
        pl.col("F1").mean().alias("F1"),
        pl.col("F2").mean().alias("F2"),
    )
    means_layer = (
        alt.Chart(means_df)
        .mark_point(shape="diamond", size=200, filled=True, stroke="white", strokeWidth=1.5)
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
        diph_layer = (
            alt.Chart(diph_df)
            .mark_point(shape="triangle-up", size=60, filled=True, stroke="white", strokeWidth=0.5)
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


def save_chart(session: str, mode: str = "mono") -> None:
    d = session_dir(session)
    out_path = d / f"{session}_vowel_space.html"
    build_chart(session, mode=mode).save(str(out_path))
    print(f"Created {out_path}")
