from pathlib import Path

import altair as alt
import polars as pl

from ..paths import project_root, session_dir
from ..schema import Wells
from .ellipse import precompute_ellipse
from .vowel_space import _inject_controls


def _bark(i: int) -> pl.Expr:
    col = pl.col(f"F{i}")
    return ((26.81 * col) / (1960 + col) - 0.53).alias(f"Z{i}")


def _add_bark_differences(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(_bark(i) for i in range(1, 4))
    return df.with_columns(
        (pl.col("Z2") - pl.col("Z1")).alias("Frontness"),
        (pl.col("Z3") - pl.col("Z2")).alias("Roundness"),
    )


def build_bark_chart(session: str) -> alt.LayerChart:
    df = pl.read_csv(session_dir(session) / f"{session}_formants.csv")
    df = _add_bark_differences(df)

    is_diph = pl.col("label").str.contains(":")
    mono_df = df.filter(~is_diph)
    diph_df = df.filter(is_diph)
    has_diph = len(diph_df) > 0

    all_sets = sorted(df["set"].unique().to_list())
    color_scale = alt.Scale(domain=all_sets, range=[Wells[s].value for s in all_sets])

    words_param = alt.param(name="showWords", value=True)
    means_param = alt.param(name="showMeans", value=True)
    mono_param = alt.param(name="showMono", value=True)
    diph_param = alt.param(name="showDiph", value=True)
    set_params = {s: alt.param(name=f"show_{s}", value=True) for s in all_sets}

    _sets = " && ".join(
        f'(datum.set === "{s}" ? show_{s} : true)' for s in all_sets
    )
    mono_vis = f"showWords && showMono && ({_sets})"
    means_vis = f"showMeans && showMono && ({_sets})"
    ellipse_vis = f"showMono && ({_sets})"
    diph_vis = f"showWords && showDiph && ({_sets})"
    diph_means_vis = f"showMeans && showDiph && ({_sets})"

    # Layer 1: IPA reference — parquet already has Frontness/Roundness columns
    std_path: Path = project_root() / "male_standard.parquet"
    std_df = pl.read_parquet(std_path).drop_nulls(subset=["Frontness", "Roundness"])
    ref_layer = (
        alt.Chart(std_df)
        .mark_text(color="#c0c0c0", fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("Frontness:Q", title="Frontness — Z2 − Z1 (Bark)"),
            y=alt.Y("Roundness:Q", title="Roundness — Z3 − Z2 (Bark)"),
            text="label:N",
        )
    )

    # Layer 2: Confidence ellipses (monophthongs)
    ellipse_records: list[dict] = []
    for s in sorted(mono_df["set"].unique().to_list()):
        subset = mono_df.filter(pl.col("set") == s)
        if len(subset) < 3:
            continue
        # precompute_ellipse expects (x_array, y_array) — returns {"F2": x, "F1": y}
        pts = precompute_ellipse(
            subset["Frontness"].to_numpy(), subset["Roundness"].to_numpy()
        )
        if pts:
            for k, pt in enumerate(pts):
                ellipse_records.append(
                    {"set": s, "Frontness": pt["F2"], "Roundness": pt["F1"], "idx": k}
                )

    layers: list[alt.Chart] = [ref_layer]

    if ellipse_records:
        ellipse_df = pl.DataFrame(ellipse_records)
        ellipse_layer = (
            alt.Chart(ellipse_df)
            .mark_line(filled=True, fillOpacity=0.12, strokeWidth=1.5)
            .encode(
                x=alt.X("Frontness:Q"),
                y=alt.Y("Roundness:Q"),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                detail="set:N",
                order=alt.Order("idx:O"),
                opacity=alt.when(ellipse_vis).then(alt.value(0.7)).otherwise(alt.value(0.0)),
            )
        )
        layers.append(ellipse_layer)

    # Layer 3: Monophthong tokens
    token_layer = (
        alt.Chart(mono_df)
        .mark_circle(size=60, stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("Frontness:Q", title="Frontness — Z2 − Z1 (Bark)"),
            y=alt.Y("Roundness:Q", title="Roundness — Z3 − Z2 (Bark)"),
            color=alt.Color("set:N", scale=color_scale, legend=alt.Legend(title="Lexical set")),
            opacity=alt.when(mono_vis).then(alt.value(0.85)).otherwise(alt.value(0.0)),
            tooltip=[
                alt.Tooltip("word:N", title="Word"),
                alt.Tooltip("set:N", title="Set"),
                alt.Tooltip("Frontness:Q", title="Frontness (Z2−Z1)", format=".2f"),
                alt.Tooltip("Roundness:Q", title="Roundness (Z3−Z2)", format=".2f"),
            ],
        )
    )
    layers.append(token_layer)

    # Layer 4: Per-set means
    means_df = mono_df.group_by("set").agg(
        pl.col("Frontness").mean(), pl.col("Roundness").mean()
    )
    means_layer = (
        alt.Chart(means_df)
        .mark_point(shape="diamond", size=200, filled=True, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("Frontness:Q"),
            y=alt.Y("Roundness:Q"),
            color=alt.Color("set:N", scale=color_scale, legend=None),
            opacity=alt.when(means_vis).then(alt.value(1.0)).otherwise(alt.value(0.0)),
            tooltip=[
                alt.Tooltip("set:N", title="Set"),
                alt.Tooltip("Frontness:Q", title="Frontness mean (Z2−Z1)", format=".2f"),
                alt.Tooltip("Roundness:Q", title="Roundness mean (Z3−Z2)", format=".2f"),
            ],
        )
    )
    layers.append(means_layer)

    # Layers 5-7: Diphthong tokens, mean trajectories, mean endpoint markers
    if has_diph:
        diph_df = diph_df.with_columns(
            pl.col("label").str.split(":").list.first().alias("token"),
            pl.col("label").str.split(":").list.last().cast(pl.Int32).alias("point_num"),
        )
        diph_means_df = (
            diph_df.group_by(["set", "point_num"])
            .agg(pl.col("Frontness").mean(), pl.col("Roundness").mean())
            .sort(["set", "point_num"])
        )

        diph_trail_layer = (
            alt.Chart(diph_df)
            .mark_trail()
            .encode(
                x=alt.X("Frontness:Q"),
                y=alt.Y("Roundness:Q"),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                size=alt.Size(
                    "point_num:Q",
                    scale=alt.Scale(domain=[1, 2], range=[1, 4]),
                    legend=None,
                ),
                detail="token:N",
                order=alt.Order("point_num:O"),
                opacity=alt.when(diph_vis).then(alt.value(0.6)).otherwise(alt.value(0.0)),
                tooltip=[
                    alt.Tooltip("word:N", title="Word"),
                    alt.Tooltip("set:N", title="Set"),
                    alt.Tooltip("Frontness:Q", title="Frontness (Z2−Z1)", format=".2f"),
                    alt.Tooltip("Roundness:Q", title="Roundness (Z3−Z2)", format=".2f"),
                ],
            )
        )
        layers.append(diph_trail_layer)

        diph_mean_trail_layer = (
            alt.Chart(diph_means_df)
            .mark_trail()
            .encode(
                x=alt.X("Frontness:Q"),
                y=alt.Y("Roundness:Q"),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                size=alt.Size(
                    "point_num:Q",
                    scale=alt.Scale(domain=[1, 2], range=[3, 9]),
                    legend=None,
                ),
                detail="set:N",
                order=alt.Order("point_num:O"),
                opacity=alt.when(diph_means_vis).then(alt.value(0.9)).otherwise(alt.value(0.0)),
            )
        )
        layers.append(diph_mean_trail_layer)

        diph_mean_pts_layer = (
            alt.Chart(diph_means_df)
            .mark_point(shape="diamond", filled=True, stroke="white", strokeWidth=1.5)
            .encode(
                x=alt.X("Frontness:Q"),
                y=alt.Y("Roundness:Q"),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                size=alt.Size(
                    "point_num:Q",
                    scale=alt.Scale(domain=[1, 2], range=[80, 200]),
                    legend=None,
                ),
                opacity=alt.when(diph_means_vis).then(alt.value(1.0)).otherwise(alt.value(0.0)),
                tooltip=[
                    alt.Tooltip("set:N", title="Set"),
                    alt.Tooltip("point_num:O", title="Point"),
                    alt.Tooltip("Frontness:Q", title="Frontness mean (Z2−Z1)", format=".2f"),
                    alt.Tooltip("Roundness:Q", title="Roundness mean (Z3−Z2)", format=".2f"),
                ],
            )
        )
        layers.append(diph_mean_pts_layer)

    return (
        alt.layer(*layers)
        .add_params(words_param, means_param, mono_param, diph_param, *set_params.values())
        .resolve_scale(x="shared", y="shared")
        .properties(width=650, height=550, title=f"Bark Z Vowel Space — {session}")
        .configure_view(strokeWidth=0)
    )


def save_bark_chart(session: str) -> None:
    df = pl.read_csv(session_dir(session) / f"{session}_formants.csv")
    has_diph = df["label"].str.contains(":").any()
    all_sets = sorted(df["set"].unique().to_list())
    set_colors = {s: Wells[s].value for s in all_sets}

    out_path: Path = session_dir(session) / f"{session}_bark_space.html"
    html = build_bark_chart(session).to_html()
    html = _inject_controls(html, has_diph=has_diph, set_colors=set_colors)
    out_path.write_text(html)
    print(f"Created {out_path}")
