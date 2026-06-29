from pathlib import Path
from typing import Final

import altair as alt
import plotly.graph_objects as go
import polars as pl

from ..paths import data_dir, session_dir
from ..schema import GROUPS, Wells
from .ellipse import precompute_ellipse
from .vowel_space import _inject_controls, _text_color

DIV_ID: Final[str] = "bark-plot"


def _bark(i: int) -> pl.Expr:
    col: pl.Expr = pl.col(f"F{i}")
    return ((26.81 * col) / (1960 + col) - 0.53).alias(f"Z{i}")


def _add_bark_dims(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_bark(i) for i in range(0, 4)).with_columns(
        (pl.col("Z1") - pl.col("Z0")).alias("Openness"),
        (pl.col("Z2") - pl.col("Z1")).alias("Frontness"),
        (pl.col("Z3") - pl.col("Z2")).alias("Roundness"),
    )


def _load_formants(session: str) -> pl.DataFrame:
    from ..aggregate import load_points

    return load_points(session).pipe(_add_bark_dims).filter(pl.col("F0").is_not_nan())


def _proj_angle_expr(
    x_col: str,
    y_col: str,
    x_rng: float,
    y_rng: float,
    x_rev: bool,
    y_rev: bool,
) -> pl.Expr:
    """Arrowhead angle expression for Vega-Lite (degrees clockwise from North).

    Maps data-space deltas into normalised screen-space deltas accounting for
    reversed axes, then computes atan2d(screen_dx, screen_dy_upward).
    """
    x_sign: float = -1.0 if x_rev else 1.0
    y_sign_up: float = -1.0 if y_rev else 1.0
    screen_dx: pl.Expr = x_sign * (pl.col(x_col) - pl.col(f"{x_col}_s")) / x_rng
    screen_dy_up: pl.Expr = y_sign_up * (pl.col(y_col) - pl.col(f"{y_col}_s")) / y_rng
    return pl.arctan2(screen_dx, screen_dy_up).degrees().alias("angle")


def _load_std(drop_nulls: list[str]) -> pl.DataFrame:
    df: pl.DataFrame = pl.read_parquet(data_dir / "standards" / "male_standard.parquet")
    if "Closeness" in df.columns:
        df = df.rename({"Closeness": "Openness"})
    return df.drop_nulls(subset=drop_nulls)


def build_bark_chart(df: pl.DataFrame, session: str) -> go.Figure:
    is_diph: pl.Expr = pl.col("label").str.contains(":")
    mono_df: pl.DataFrame = df.filter(~is_diph)
    diph_df: pl.DataFrame = df.filter(is_diph)
    has_diph: bool = not diph_df.is_empty()

    all_sets: list[str] = sorted(df["set"].unique().to_list())
    color_map: dict[str, str] = {s: Wells[s].value for s in all_sets}

    std_df: pl.DataFrame = _load_std(["Openness", "Frontness", "Roundness"])
    traces: list[go.Scatter3d] = [
        go.Scatter3d(
            x=std_df["Frontness"].to_list(),
            y=std_df["Openness"].to_list(),
            z=std_df["Roundness"].to_list(),
            mode="text",
            text=std_df["label"].to_list(),
            textfont=dict(color="#c0c0c0", size=10),
            meta={},
            showlegend=False,
            hoverinfo="skip",
            name="IPA reference",
        )
    ]

    for s in all_sets:
        c: str = color_map[s]
        sub: pl.DataFrame = mono_df.filter(pl.col("set") == s)
        if sub.is_empty():
            continue

        traces.append(
            go.Scatter3d(
                x=sub["Frontness"].to_list(),
                y=sub["Openness"].to_list(),
                z=sub["Roundness"].to_list(),
                mode="markers",
                marker=dict(size=5, color=c, opacity=0.8, symbol="circle"),
                name=s,
                legendgroup=s,
                showlegend=False,
                meta={"set": s, "vtype": "mono", "kind": "tokens"},
                customdata=list(
                    zip(
                        sub["word"].to_list(),
                        sub["F0"].to_list(),
                        sub["F1"].to_list(),
                        sub["F2"].to_list(),
                        sub["F3"].to_list(),
                        strict=False,
                    )
                ),
                hovertemplate=(
                    f"<b>%{{customdata[0]}}</b> ({s})<br>"
                    "Openness: %{y:.2f}<br>"
                    "Frontness: %{x:.2f}<br>"
                    "Roundness: %{z:.2f}<br>"
                    "F0: %{customdata[1]:.0f} Hz<br>"
                    "F1: %{customdata[2]:.0f} Hz<br>"
                    "F2: %{customdata[3]:.0f} Hz<br>"
                    "F3: %{customdata[4]:.0f} Hz"
                    "<extra></extra>"
                ),
            )
        )

        means: tuple[float, float, float] = sub.select(
            pl.col("Frontness").mean(),
            pl.col("Openness").mean(),
            pl.col("Roundness").mean(),
        ).row(0)
        traces.append(
            go.Scatter3d(
                x=[means[0]],
                y=[means[1]],
                z=[means[2]],
                mode="markers",
                marker=dict(
                    size=10,
                    color=c,
                    opacity=1.0,
                    symbol="diamond",
                    line=dict(color="white", width=1),
                ),
                name=s,
                legendgroup=s,
                showlegend=False,
                meta={"set": s, "vtype": "mono", "kind": "means"},
                hovertemplate=(
                    f"<b>{s}</b> mean<br>"
                    "Openness: %{y:.2f}<br>"
                    "Frontness: %{x:.2f}<br>"
                    "Roundness: %{z:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    if has_diph:
        diph_df = diph_df.with_columns(
            pl.col("label").str.split(":").list.first().alias("token"),
            pl.col("label")
            .str.split(":")
            .list.last()
            .cast(pl.Int32)
            .alias("point_num"),
        )

        for s in all_sets:
            c = color_map[s]
            sub = diph_df.filter(pl.col("set") == s)
            if sub.is_empty():
                continue

            tokens: list[str] = sub["token"].unique().to_list()
            x_segs: list[float] = []
            y_segs: list[float] = []
            z_segs: list[float] = []
            for tok in tokens:
                pts: pl.DataFrame = sub.filter(pl.col("token") == tok).sort("point_num")
                if len(pts) < 2:
                    continue
                x_segs += pts["Frontness"].to_list() + [None]
                y_segs += pts["Openness"].to_list() + [None]
                z_segs += pts["Roundness"].to_list() + [None]

            if x_segs:
                traces.append(
                    go.Scatter3d(
                        x=x_segs,
                        y=y_segs,
                        z=z_segs,
                        mode="lines+markers",
                        line=dict(color=c, width=2),
                        marker=dict(size=3, color=c, opacity=0.6),
                        name=s,
                        legendgroup=s,
                        showlegend=False,
                        meta={"set": s, "vtype": "diph", "kind": "tokens"},
                        hovertemplate=(
                            f"({s})<br>"
                            "Openness: %{y:.2f}<br>"
                            "Frontness: %{x:.2f}<br>"
                            "Roundness: %{z:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

            dm: pl.DataFrame = (
                diph_df.filter(pl.col("set") == s)
                .group_by("point_num")
                .agg(
                    pl.col("Frontness").mean(),
                    pl.col("Openness").mean(),
                    pl.col("Roundness").mean(),
                )
                .sort("point_num")
            )
            if len(dm) >= 2:
                traces.append(
                    go.Scatter3d(
                        x=dm["Frontness"].to_list(),
                        y=dm["Openness"].to_list(),
                        z=dm["Roundness"].to_list(),
                        mode="lines+markers",
                        line=dict(color=c, width=5),
                        marker=dict(
                            size=8,
                            color=c,
                            opacity=1.0,
                            symbol="diamond",
                            line=dict(color="white", width=1),
                        ),
                        name=s,
                        legendgroup=s,
                        showlegend=False,
                        meta={"set": s, "vtype": "diph", "kind": "means"},
                        hovertemplate=(
                            f"<b>{s}</b> mean<br>"
                            "Openness: %{y:.2f}<br>"
                            "Frontness: %{x:.2f}<br>"
                            "Roundness: %{z:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

    return go.Figure(
        data=traces,
        layout=go.Layout(
            title=f"Bark Z Vowel Space — {session}",
            showlegend=False,
            scene=dict(
                xaxis=dict(title="Frontness (Z2 - Z1)", autorange=True),
                yaxis=dict(title="Openness (Z1 - Z0)", autorange=True),
                zaxis=dict(title="Roundness (Z3 - Z2)", autorange=True),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        ),
    )


def _inject_bark_controls(
    html: str, *, has_diph: bool, set_colors: dict[str, str]
) -> str:
    def btn(sig: str, label: str) -> str:
        return f'<button class="vt-btn active" data-signal="{sig}">{label}</button>'

    def set_btn(name: str, color: str) -> str:
        tc: str = _text_color(color)
        return (
            f'<button class="vt-set-btn active" data-set="{name}" '
            f'style="background:{color};color:{tc}">{name}</button>'
        )

    display_html: str = btn("showWords", "Words") + btn("showMeans", "Set averages")
    type_html: str = btn("showMono", "Monophthongs")
    if has_diph:
        type_html += btn("showDiph", "Diphthongs")

    groups_with_data: list[tuple[str, str]] = [
        (
            grp_name,
            "".join(set_btn(s, set_colors[s]) for s in grp_sets if s in set_colors),
        )
        for grp_name, grp_sets in GROUPS.items()
    ]
    groups_with_data = [(g, b) for g, b in groups_with_data if b]
    mid: int = (len(groups_with_data) + 1) // 2
    col1_html: str = "".join(
        f'<div class="vt-group-hdr">{g}</div>{b}' for g, b in groups_with_data[:mid]
    )
    col2_html: str = "".join(
        f'<div class="vt-group-hdr">{g}</div>{b}' for g, b in groups_with_data[mid:]
    )
    set_grid: str = (
        '<div class="vt-set-cols">'
        '<div class="vt-set-col">' + col1_html + "</div>"
        '<div class="vt-set-col">' + col2_html + "</div>"
        "</div>"
    )

    sidebar: str = (
        '<div id="vt-controls">'
        '<div class="vt-section">'
        '<div class="vt-label">Display</div>' + display_html + "</div>"
        '<div class="vt-section">'
        '<div class="vt-label">Type</div>' + type_html + "</div>"
        '<div class="vt-section">'
        '<div class="vt-label">Lexical sets</div>'
        '<button class="vt-btn active" id="vt-all-btn">All</button>'
        + set_grid
        + "</div>"
        "</div>"
    )

    css: str = (
        "<style>"
        "body{display:flex;align-items:flex-start;gap:20px;"
        "padding:12px;margin:0;box-sizing:border-box;}"
        "#vt-controls+div{flex:1;min-width:0;}"
        "#vt-controls{width:185px;flex-shrink:0;font-family:sans-serif;padding-top:8px;}"
        ".vt-section{margin-bottom:14px;}"
        ".vt-label{font-size:10px;font-weight:bold;color:#aaa;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px;}"
        ".vt-btn{display:block;width:100%;padding:6px 8px;margin-bottom:4px;"
        "border:none;border-radius:6px;font-size:13px;font-weight:bold;"
        "cursor:pointer;background:#2c7bb6;color:white;"
        "text-align:center;box-sizing:border-box;"
        "transition:background .15s,color .15s;}"
        ".vt-btn:not(.active){background:#c8d8e8;color:#5a7a99;}"
        ".vt-set-cols{display:flex;gap:4px;}"
        ".vt-set-col{flex:1;display:flex;flex-direction:column;gap:4px;}"
        ".vt-group-hdr{font-size:8px;color:#bbb;text-transform:uppercase;"
        "letter-spacing:.04em;margin-top:6px;margin-bottom:1px;}"
        ".vt-set-col>:first-child.vt-group-hdr{margin-top:0;}"
        ".vt-set-btn{padding:4px 2px;border:none;border-radius:4px;font-size:11px;"
        "font-weight:bold;cursor:pointer;text-align:center;transition:opacity .15s;}"
        ".vt-set-btn:not(.active){opacity:0.25;}"
        "</style>"
    )

    js: str = (
        "<script>"
        "function setupToggles(gd){"
        "var state={showWords:true,showMeans:true,showMono:true,showDiph:true};"
        "function isOn(t){"
        "var m=t.meta||{};"
        "if(!m.set)return true;"
        "if(state['hide_'+m.set])return false;"
        "if(m.vtype==='mono'){"
        "if(!state.showMono)return false;"
        "if(m.kind==='tokens'&&!state.showWords)return false;"
        "if(m.kind==='means'&&!state.showMeans)return false;"
        "}"
        "if(m.vtype==='diph'){"
        "if(!state.showDiph)return false;"
        "if(m.kind==='tokens'&&!state.showWords)return false;"
        "if(m.kind==='means'&&!state.showMeans)return false;"
        "}"
        "return true;"
        "}"
        "function reapply(){"
        "var vis=gd.data.map(function(t){return isOn(t);});"
        "var idx=gd.data.map(function(_,i){return i;});"
        "Plotly.restyle(gd,{visible:vis},idx);"
        "}"
        "document.querySelectorAll('.vt-btn').forEach(function(btn){"
        "btn.addEventListener('click',function(){"
        "var sig=this.dataset.signal;"
        "if(!sig)return;"
        "var active=this.classList.toggle('active');"
        "state[sig]=active;"
        "reapply();"
        "});"
        "});"
        "var allBtn=document.getElementById('vt-all-btn');"
        "function syncAllBtn(){"
        "var setBtns=document.querySelectorAll('.vt-set-btn');"
        "var allOn=Array.from(setBtns).every(function(b){return b.classList.contains('active');});"
        "allBtn.classList.toggle('active',allOn);"
        "}"
        "allBtn.addEventListener('click',function(){"
        "var setBtns=document.querySelectorAll('.vt-set-btn');"
        "var allOn=Array.from(setBtns).every(function(b){return b.classList.contains('active');});"
        "var next=!allOn;"
        "this.classList.toggle('active',next);"
        "setBtns.forEach(function(btn){"
        "btn.classList.toggle('active',next);"
        "state['hide_'+btn.dataset.set]=!next;"
        "});"
        "reapply();"
        "});"
        "document.querySelectorAll('.vt-set-btn').forEach(function(btn){"
        "btn.addEventListener('click',function(){"
        "var active=this.classList.toggle('active');"
        "state['hide_'+this.dataset.set]=!active;"
        "reapply();"
        "syncAllBtn();"
        "});"
        "});"
        "}"
        "</script>"
    )

    html = html.replace("</head>", css + js + "</head>", 1)
    html = html.replace("<body>", f"<body>{sidebar}", 1)
    html = html.replace(
        'class="plotly-graph-div"',
        'class="plotly-graph-div" style="width:100%;height:700px;"',
        1,
    )
    return html


def save_bark_chart(session: str) -> None:
    df: pl.DataFrame = _load_formants(session)
    has_diph: bool = df["label"].str.contains(":").any()
    all_sets: list[str] = sorted(df["set"].unique().to_list())
    set_colors: dict[str, str] = {s: Wells[s].value for s in all_sets}

    fig: go.Figure = build_bark_chart(df, session)
    html: str = fig.to_html(
        div_id=DIV_ID,
        include_plotlyjs=True,
        full_html=True,
        post_script=f"setupToggles(document.getElementById('{DIV_ID}'));",
    )
    html = _inject_bark_controls(html, has_diph=has_diph, set_colors=set_colors)

    out_path: Path = session_dir(session) / f"{session}_bark_space.html"
    out_path.write_text(html)
    print(f"Created {out_path}")


# ── 2-D projections ──────────────────────────────────────────────────────────

_AXIS_TITLES: Final[dict[str, str]] = {
    "Frontness": "Frontness (Z2 - Z1)",
    "Openness": "Openness (Z1 - Z0)",
    "Roundness": "Roundness (Z3 - Z2)",
}

# Mirror the IPA vowel chart convention: front vowels left, open vowels down.
_AXIS_REVERSED: Final[dict[str, bool]] = {
    "Frontness": True,
    "Openness": True,
    "Roundness": False,
}


def _projection_chart(
    mono_df: pl.DataFrame,
    diph_df: pl.DataFrame,
    std_df: pl.DataFrame,
    x_col: str,
    y_col: str,
    color_scale: alt.Scale,
    mono_vis: str,
    means_vis: str,
    ellipse_vis: str,
    diph_vis: str,
    diph_means_vis: str,
) -> alt.LayerChart | alt.FacetChart:
    x_enc = alt.X(
        f"{x_col}:Q",
        scale=alt.Scale(reverse=_AXIS_REVERSED[x_col]),
        title=_AXIS_TITLES[x_col],
    )
    y_enc = alt.Y(
        f"{y_col}:Q",
        scale=alt.Scale(reverse=_AXIS_REVERSED[y_col]),
        title=_AXIS_TITLES[y_col],
    )

    ref_layer: alt.Chart = (
        alt.Chart(std_df)
        .mark_text(color="#c0c0c0", fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X(f"{x_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[x_col])),
            y=alt.Y(f"{y_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[y_col])),
            text="label:N",
        )
    )

    # Confidence ellipses
    ellipse_records: list[dict[str, float | str]] = []
    for s in sorted(mono_df["set"].unique().to_list()):
        subset: pl.DataFrame = mono_df.filter(pl.col("set") == s)
        if len(subset) < 3:
            continue
        pts: list[dict[str, float]] | None = precompute_ellipse(
            subset.get_column(x_col).to_numpy(), subset.get_column(y_col).to_numpy()
        )
        if pts:
            for k, pt in enumerate(pts):
                ellipse_records.append(
                    {"set": s, x_col: pt["F2"], y_col: pt["F1"], "idx": k}
                )

    layers: list[alt.Chart] = [ref_layer]
    if ellipse_records:
        layers.append(
            alt.Chart(pl.DataFrame(ellipse_records))
            .mark_line(filled=True, fillOpacity=0.12, strokeWidth=1.5)
            .encode(
                x=alt.X(f"{x_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[x_col])),
                y=alt.Y(f"{y_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[y_col])),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                detail="set:N",
                order=alt.Order("idx:O"),
                opacity=alt.when(ellipse_vis)
                .then(alt.value(0.7))
                .otherwise(alt.value(0.0)),
            )
        )

    layers.append(
        alt.Chart(mono_df)
        .mark_circle(size=60, stroke="white", strokeWidth=0.5)
        .encode(
            x=x_enc,
            y=y_enc,
            color=alt.Color(
                "set:N", scale=color_scale, legend=alt.Legend(title="Lexical set")
            ),
            opacity=alt.when(mono_vis).then(alt.value(0.85)).otherwise(alt.value(0.0)),
            tooltip=[
                alt.Tooltip("word:N", title="Word"),
                alt.Tooltip("set:N", title="Set"),
                alt.Tooltip(f"{x_col}:Q", title=_AXIS_TITLES[x_col], format=".2f"),
                alt.Tooltip(f"{y_col}:Q", title=_AXIS_TITLES[y_col], format=".2f"),
            ],
        )
    )

    layers.append(
        alt.Chart(
            mono_df.group_by("set").agg(pl.col(x_col).mean(), pl.col(y_col).mean())
        )
        .mark_point(
            shape="diamond", size=200, filled=True, stroke="white", strokeWidth=1.5
        )
        .encode(
            x=alt.X(f"{x_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[x_col])),
            y=alt.Y(f"{y_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[y_col])),
            color=alt.Color("set:N", scale=color_scale, legend=None),
            opacity=alt.when(means_vis).then(alt.value(1.0)).otherwise(alt.value(0.0)),
            tooltip=[
                alt.Tooltip("set:N", title="Set"),
                alt.Tooltip(
                    f"{x_col}:Q", title=f"{_AXIS_TITLES[x_col]} mean", format=".2f"
                ),
                alt.Tooltip(
                    f"{y_col}:Q", title=f"{_AXIS_TITLES[y_col]} mean", format=".2f"
                ),
            ],
        )
    )

    if not diph_df.is_empty():
        diph_means_df: pl.DataFrame = (
            diph_df.group_by(["set", "point_num"])
            .agg(pl.col(x_col).mean(), pl.col(y_col).mean())
            .sort(["set", "point_num"])
        )

        x_rev: bool = _AXIS_REVERSED[x_col]
        y_rev: bool = _AXIS_REVERSED[y_col]
        x_rng: float = (
            float(mono_df.get_column(x_col).max())  # type: ignore
            - float(mono_df.get_column(x_col).min())  # type: ignore
            or 1.0
        )
        y_rng: float = (
            float(mono_df.get_column(y_col).max())  # type: ignore
            - float(mono_df.get_column(y_col).min())  # type: ignore
            or 1.0
        )

        pt1_t: pl.DataFrame = diph_df.filter(pl.col("point_num") == 1)[
            ["token", "set", x_col, y_col]
        ].rename({x_col: f"{x_col}_s", y_col: f"{y_col}_s"})
        pt2_t: pl.DataFrame = diph_df.filter(pl.col("point_num") == 2)[
            ["token", "set", x_col, y_col, "word"]
        ]
        tok_arr: pl.DataFrame = pt2_t.join(pt1_t, on=["token", "set"]).with_columns(
            _proj_angle_expr(x_col, y_col, x_rng, y_rng, x_rev, y_rev)
        )

        pt1_m: pl.DataFrame = diph_means_df.filter(pl.col("point_num") == 1)[
            ["set", x_col, y_col]
        ].rename({x_col: f"{x_col}_s", y_col: f"{y_col}_s"})
        pt2_m: pl.DataFrame = diph_means_df.filter(pl.col("point_num") == 2)[
            ["set", x_col, y_col]
        ]
        mean_arr: pl.DataFrame = pt2_m.join(pt1_m, on="set").with_columns(
            _proj_angle_expr(x_col, y_col, x_rng, y_rng, x_rev, y_rev)
        )

        ang_scale: alt.Scale = alt.Scale(domain=[-180, 180], range=[-180, 180])
        x_sc: alt.Scale = alt.Scale(reverse=x_rev)
        y_sc: alt.Scale = alt.Scale(reverse=y_rev)

        # Token lines
        layers.append(
            alt.Chart(diph_df)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X(f"{x_col}:Q", scale=x_sc),
                y=alt.Y(f"{y_col}:Q", scale=y_sc),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                detail="token:N",
                order=alt.Order("point_num:O"),
                opacity=alt.when(diph_vis)
                .then(alt.value(0.4))
                .otherwise(alt.value(0.0)),
            )
        )
        # Start dots at point 1
        layers.append(
            alt.Chart(diph_df.filter(pl.col("point_num") == 1))
            .mark_point(shape="circle", size=25, filled=True)
            .encode(
                x=alt.X(f"{x_col}:Q", scale=x_sc),
                y=alt.Y(f"{y_col}:Q", scale=y_sc),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                opacity=alt.when(diph_vis)
                .then(alt.value(0.5))
                .otherwise(alt.value(0.0)),
            )
        )
        # Arrowhead triangles at point 2
        layers.append(
            alt.Chart(tok_arr)
            .mark_point(shape="triangle", size=60, filled=True)
            .encode(
                x=alt.X(f"{x_col}:Q", scale=x_sc),
                y=alt.Y(f"{y_col}:Q", scale=y_sc),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                angle=alt.Angle("angle:Q", scale=ang_scale),
                opacity=alt.when(diph_vis)
                .then(alt.value(0.7))
                .otherwise(alt.value(0.0)),
                tooltip=[
                    alt.Tooltip("word:N", title="Word"),
                    alt.Tooltip("set:N", title="Set"),
                    alt.Tooltip(f"{x_col}:Q", title=_AXIS_TITLES[x_col], format=".2f"),
                    alt.Tooltip(f"{y_col}:Q", title=_AXIS_TITLES[y_col], format=".2f"),
                ],
            )
        )
        # Mean line
        layers.append(
            alt.Chart(diph_means_df)
            .mark_line(strokeWidth=5)
            .encode(
                x=alt.X(f"{x_col}:Q", scale=x_sc),
                y=alt.Y(f"{y_col}:Q", scale=y_sc),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                detail="set:N",
                order=alt.Order("point_num:O"),
                opacity=alt.when(diph_means_vis)
                .then(alt.value(0.9))
                .otherwise(alt.value(0.0)),
            )
        )
        # Mean arrowhead at point 2
        layers.append(
            alt.Chart(mean_arr)
            .mark_point(
                shape="triangle", size=250, filled=True, stroke="white", strokeWidth=1.5
            )
            .encode(
                x=alt.X(f"{x_col}:Q", scale=x_sc),
                y=alt.Y(f"{y_col}:Q", scale=y_sc),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                angle=alt.Angle("angle:Q", scale=ang_scale),
                opacity=alt.when(diph_means_vis)
                .then(alt.value(1.0))
                .otherwise(alt.value(0.0)),
                tooltip=[
                    alt.Tooltip("set:N", title="Set"),
                    alt.Tooltip(
                        f"{x_col}:Q", title=f"{_AXIS_TITLES[x_col]} mean", format=".2f"
                    ),
                    alt.Tooltip(
                        f"{y_col}:Q", title=f"{_AXIS_TITLES[y_col]} mean", format=".2f"
                    ),
                ],
            )
        )

    return (
        alt.layer(*layers)
        .resolve_scale(x="shared", y="shared")
        .properties(width=380, height=380, title=f"{x_col} × {y_col}")
    )


def build_bark_projections(df: pl.DataFrame, session: str) -> alt.HConcatChart:
    is_diph: pl.Expr = pl.col("label").str.contains(":")
    mono_df: pl.DataFrame = df.filter(~is_diph)
    diph_df: pl.DataFrame = df.filter(is_diph)

    if not diph_df.is_empty():
        diph_df = diph_df.with_columns(
            pl.col("label").str.split(":").list.first().alias("token"),
            pl.col("label")
            .str.split(":")
            .list.last()
            .cast(pl.Int32)
            .alias("point_num"),
        )

    all_sets: list[str] = sorted(df["set"].unique().to_list())
    color_scale: alt.Scale = alt.Scale(
        domain=all_sets, range=[Wells[s].value for s in all_sets]
    )

    words_param: alt.Parameter = alt.param(name="showWords", value=True)
    means_param: alt.Parameter = alt.param(name="showMeans", value=True)
    mono_param: alt.Parameter = alt.param(name="showMono", value=True)
    diph_param: alt.Parameter = alt.param(name="showDiph", value=True)
    set_params: dict[str, alt.Parameter] = {
        s: alt.param(name=f"show_{s}", value=True) for s in all_sets
    }

    _sets: str = " && ".join(
        f'(datum.set === "{s}" ? show_{s} : true)' for s in all_sets
    )
    mono_vis: str = f"showWords && showMono && ({_sets})"
    means_vis: str = f"showMeans && showMono && ({_sets})"
    ellipse_vis: str = f"showMono && ({_sets})"
    diph_vis: str = f"showWords && showDiph && ({_sets})"
    diph_means_vis: str = f"showMeans && showDiph && ({_sets})"

    std_df: pl.DataFrame = _load_std(["Frontness", "Roundness"])

    def projection(x_col: str, y_col: str) -> alt.LayerChart | alt.FacetChart:
        return _projection_chart(
            mono_df=mono_df,
            diph_df=diph_df,
            std_df=std_df,
            x_col=x_col,
            y_col=y_col,
            color_scale=color_scale,
            mono_vis=mono_vis,
            means_vis=means_vis,
            ellipse_vis=ellipse_vis,
            diph_vis=diph_vis,
            diph_means_vis=diph_means_vis,
        )

    return (
        alt.hconcat(
            projection("Frontness", "Openness"),
            projection("Frontness", "Roundness"),
            projection("Openness", "Roundness"),
        )
        .add_params(
            words_param, means_param, mono_param, diph_param, *set_params.values()
        )
        .properties(title=f"Bark Z Projections — {session}")
        .configure_view(strokeWidth=0)
    )


def save_bark_projections(session: str) -> None:
    df: pl.DataFrame = _load_formants(session)
    has_diph: bool = df["label"].str.contains(":").any()
    all_sets: list[str] = sorted(df["set"].unique().to_list())
    set_colors: dict[str, str] = {s: Wells[s].value for s in all_sets}

    out_path: Path = session_dir(session) / f"{session}_bark_projections.html"
    html: str = build_bark_projections(df, session).to_html()
    html = _inject_controls(html, has_diph=has_diph, set_colors=set_colors)
    out_path.write_text(html)
    print(f"Created {out_path}")
