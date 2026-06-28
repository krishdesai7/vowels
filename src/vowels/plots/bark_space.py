from pathlib import Path

import altair as alt
import plotly.graph_objects as go
import polars as pl

from ..paths import project_root, session_dir
from ..schema import Wells
from .ellipse import precompute_ellipse
from .vowel_space import _inject_controls, _text_color

DIV_ID = "bark-plot"


def _bark(i: int) -> pl.Expr:
    col = pl.col(f"F{i}")
    return ((26.81 * col) / (1960 + col) - 0.53).alias(f"Z{i}")


def _add_bark_dims(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(_bark(i) for i in range(0, 4))
    return df.with_columns(
        (pl.col("Z1") - pl.col("Z0")).alias("Openness"),
        (pl.col("Z2") - pl.col("Z1")).alias("Frontness"),
        (pl.col("Z3") - pl.col("Z2")).alias("Roundness"),
    )


def build_bark_chart(session: str) -> go.Figure:
    df = pl.read_csv(session_dir(session) / f"{session}_formants.csv")
    df = _add_bark_dims(df)
    df = df.filter(pl.col("F0").is_not_nan())

    is_diph = pl.col("label").str.contains(":")
    mono_df = df.filter(~is_diph)
    diph_df = df.filter(is_diph)
    has_diph = len(diph_df) > 0

    all_sets = sorted(df["set"].unique().to_list())
    color_map = {s: Wells[s].value for s in all_sets}

    traces: list[go.BaseTraceType] = []

    # IPA reference overlay
    std_path: Path = project_root() / "male_standard.parquet"
    std_df = pl.read_parquet(std_path)
    if "Closeness" in std_df.columns:
        std_df = std_df.rename({"Closeness": "Openness"})
    std_df = std_df.drop_nulls(subset=["Openness", "Frontness", "Roundness"])
    traces.append(
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
    )

    for s in all_sets:
        c = color_map[s]
        sub = mono_df.filter(pl.col("set") == s)
        if len(sub) == 0:
            continue

        # Monophthong tokens
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
                customdata=list(zip(
                    sub["word"].to_list(),
                    sub["F0"].to_list(),
                    sub["F1"].to_list(),
                    sub["F2"].to_list(),
                    sub["F3"].to_list(),
                )),
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

        # Monophthong means
        means = sub.select(
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
                marker=dict(size=10, color=c, opacity=1.0, symbol="diamond",
                            line=dict(color="white", width=1)),
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
            pl.col("label").str.split(":").list.last().cast(pl.Int32).alias("point_num"),
        )

        for s in all_sets:
            c = color_map[s]
            sub = diph_df.filter(pl.col("set") == s)
            if len(sub) == 0:
                continue

            tokens = sub["token"].unique().to_list()
            x_segs, y_segs, z_segs = [], [], []
            for tok in tokens:
                pts = sub.filter(pl.col("token") == tok).sort("point_num")
                if len(pts) < 2:
                    continue
                x_segs += pts["Frontness"].to_list() + [None]
                y_segs += pts["Openness"].to_list() + [None]
                z_segs += pts["Roundness"].to_list() + [None]

            if x_segs:
                traces.append(
                    go.Scatter3d(
                        x=x_segs, y=y_segs, z=z_segs,
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

            dm = (
                diph_df.filter(pl.col("set") == s)
                .group_by("point_num")
                .agg(pl.col("Frontness").mean(), pl.col("Openness").mean(), pl.col("Roundness").mean())
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
                        marker=dict(size=8, color=c, opacity=1.0, symbol="diamond",
                                    line=dict(color="white", width=1)),
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

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=f"Bark Z Vowel Space — {session}",
            showlegend=False,
            scene=dict(
                xaxis=dict(title="Frontness (Z2 − Z1)", autorange=True),
                yaxis=dict(title="Openness (Z1 − Z0)", autorange=True),
                zaxis=dict(title="Roundness (Z3 − Z2)", autorange=True),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        ),
    )
    return fig


def _inject_bark_controls(
    html: str, *, has_diph: bool, set_colors: dict[str, str]
) -> str:
    def btn(sig: str, label: str) -> str:
        return f'<button class="vt-btn active" data-signal="{sig}">{label}</button>'

    def set_btn(name: str, color: str) -> str:
        tc = _text_color(color)
        return (
            f'<button class="vt-set-btn active" data-set="{name}" '
            f'style="background:{color};color:{tc}">{name}</button>'
        )

    display_html = btn("showWords", "Words") + btn("showMeans", "Set averages")
    type_html = btn("showMono", "Monophthongs")
    if has_diph:
        type_html += btn("showDiph", "Diphthongs")
    set_html = "".join(set_btn(s, c) for s, c in set_colors.items())

    sidebar = (
        '<div id="vt-controls">'
        '<div class="vt-section">'
        '<div class="vt-label">Display</div>'
        + display_html
        + "</div>"
        '<div class="vt-section">'
        '<div class="vt-label">Type</div>'
        + type_html
        + "</div>"
        '<div class="vt-section">'
        '<div class="vt-label">Lexical sets</div>'
        '<button class="vt-btn active" id="vt-all-btn">All</button>'
        '<div class="vt-set-grid">'
        + set_html
        + "</div></div>"
        "</div>"
    )

    css = (
        "<style>"
        "body{display:flex;align-items:flex-start;gap:20px;"
        "padding:12px;margin:0;box-sizing:border-box;}"
        "#vt-controls+div{flex:1;min-width:0;}"
        "#vt-controls{width:160px;flex-shrink:0;font-family:sans-serif;padding-top:8px;}"
        ".vt-section{margin-bottom:14px;}"
        ".vt-label{font-size:10px;font-weight:bold;color:#aaa;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px;}"
        ".vt-btn{display:block;width:100%;padding:6px 8px;margin-bottom:4px;"
        "border:none;border-radius:6px;font-size:13px;font-weight:bold;"
        "cursor:pointer;background:#2c7bb6;color:white;"
        "text-align:center;box-sizing:border-box;"
        "transition:background .15s,color .15s;}"
        ".vt-btn:not(.active){background:#c8d8e8;color:#5a7a99;}"
        ".vt-set-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px;}"
        ".vt-set-btn{padding:4px 4px;border:none;border-radius:4px;font-size:11px;"
        "font-weight:bold;cursor:pointer;text-align:center;transition:opacity .15s;}"
        ".vt-set-btn:not(.active){opacity:0.25;}"
        "</style>"
    )

    # setupToggles receives the graph div element directly (not via gd in post_script scope)
    js = (
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
    df = pl.read_csv(session_dir(session) / f"{session}_formants.csv")
    has_diph = df["label"].str.contains(":").any()
    all_sets = sorted(df["set"].unique().to_list())
    set_colors = {s: Wells[s].value for s in all_sets}

    fig = build_bark_chart(session)
    # post_script runs inside .then(function(){...}) with no parameter — use getElementById
    html = fig.to_html(
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

_AXIS_TITLES = {
    "Frontness": "Frontness (Z2−Z1)",
    "Openness":  "Openness (Z1−Z0)",
    "Roundness": "Roundness (Z3−Z2)",
}

# Mirror the IPA vowel chart convention: front vowels left, open vowels down.
_AXIS_REVERSED = {"Frontness": True, "Openness": True, "Roundness": False}


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
    has_diph: bool,
) -> alt.LayerChart:
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

    ref_layer = (
        alt.Chart(std_df)
        .mark_text(color="#c0c0c0", fontSize=11, fontWeight="bold")
        .encode(x=alt.X(f"{x_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[x_col])),
                y=alt.Y(f"{y_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[y_col])),
                text="label:N")
    )

    # Confidence ellipses
    ellipse_records: list[dict] = []
    for s in sorted(mono_df["set"].unique().to_list()):
        subset = mono_df.filter(pl.col("set") == s)
        if len(subset) < 3:
            continue
        pts = precompute_ellipse(subset[x_col].to_numpy(), subset[y_col].to_numpy())
        if pts:
            for k, pt in enumerate(pts):
                ellipse_records.append({"set": s, x_col: pt["F2"], y_col: pt["F1"], "idx": k})

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
                opacity=alt.when(ellipse_vis).then(alt.value(0.7)).otherwise(alt.value(0.0)),
            )
        )

    layers.append(
        alt.Chart(mono_df)
        .mark_circle(size=60, stroke="white", strokeWidth=0.5)
        .encode(
            x=x_enc,
            y=y_enc,
            color=alt.Color("set:N", scale=color_scale, legend=alt.Legend(title="Lexical set")),
            opacity=alt.when(mono_vis).then(alt.value(0.85)).otherwise(alt.value(0.0)),
            tooltip=[
                alt.Tooltip("word:N", title="Word"),
                alt.Tooltip("set:N", title="Set"),
                alt.Tooltip(f"{x_col}:Q", title=_AXIS_TITLES[x_col], format=".2f"),
                alt.Tooltip(f"{y_col}:Q", title=_AXIS_TITLES[y_col], format=".2f"),
            ],
        )
    )

    means_df = mono_df.group_by("set").agg(
        pl.col(x_col).mean(), pl.col(y_col).mean()
    )
    layers.append(
        alt.Chart(means_df)
        .mark_point(shape="diamond", size=200, filled=True, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X(f"{x_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[x_col])),
            y=alt.Y(f"{y_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[y_col])),
            color=alt.Color("set:N", scale=color_scale, legend=None),
            opacity=alt.when(means_vis).then(alt.value(1.0)).otherwise(alt.value(0.0)),
            tooltip=[
                alt.Tooltip("set:N", title="Set"),
                alt.Tooltip(f"{x_col}:Q", title=f"{_AXIS_TITLES[x_col]} mean", format=".2f"),
                alt.Tooltip(f"{y_col}:Q", title=f"{_AXIS_TITLES[y_col]} mean", format=".2f"),
            ],
        )
    )

    if has_diph and len(diph_df) > 0:
        diph_means_df = (
            diph_df.group_by(["set", "point_num"])
            .agg(pl.col(x_col).mean(), pl.col(y_col).mean())
            .sort(["set", "point_num"])
        )
        layers.append(
            alt.Chart(diph_df)
            .mark_trail()
            .encode(
                x=alt.X(f"{x_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[x_col])),
                y=alt.Y(f"{y_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[y_col])),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                size=alt.Size("point_num:Q", scale=alt.Scale(domain=[1, 2], range=[1, 4]), legend=None),
                detail="token:N",
                order=alt.Order("point_num:O"),
                opacity=alt.when(diph_vis).then(alt.value(0.6)).otherwise(alt.value(0.0)),
            )
        )
        layers.append(
            alt.Chart(diph_means_df)
            .mark_trail()
            .encode(
                x=alt.X(f"{x_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[x_col])),
                y=alt.Y(f"{y_col}:Q", scale=alt.Scale(reverse=_AXIS_REVERSED[y_col])),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                size=alt.Size("point_num:Q", scale=alt.Scale(domain=[1, 2], range=[3, 9]), legend=None),
                detail="set:N",
                order=alt.Order("point_num:O"),
                opacity=alt.when(diph_means_vis).then(alt.value(0.9)).otherwise(alt.value(0.0)),
            )
        )

    return (
        alt.layer(*layers)
        .resolve_scale(x="shared", y="shared")
        .properties(width=380, height=380, title=f"{x_col} × {y_col}")
    )


def build_bark_projections(session: str) -> alt.HConcatChart:
    df = pl.read_csv(session_dir(session) / f"{session}_formants.csv")
    df = _add_bark_dims(df)
    df = df.filter(pl.col("F0").is_not_nan())

    is_diph = pl.col("label").str.contains(":")
    mono_df = df.filter(~is_diph)
    diph_df = df.filter(is_diph)
    has_diph = len(diph_df) > 0

    if has_diph:
        diph_df = diph_df.with_columns(
            pl.col("label").str.split(":").list.first().alias("token"),
            pl.col("label").str.split(":").list.last().cast(pl.Int32).alias("point_num"),
        )

    all_sets = sorted(df["set"].unique().to_list())
    color_scale = alt.Scale(domain=all_sets, range=[Wells[s].value for s in all_sets])

    words_param = alt.param(name="showWords", value=True)
    means_param = alt.param(name="showMeans", value=True)
    mono_param  = alt.param(name="showMono",  value=True)
    diph_param  = alt.param(name="showDiph",  value=True)
    set_params  = {s: alt.param(name=f"show_{s}", value=True) for s in all_sets}

    _sets = " && ".join(f'(datum.set === "{s}" ? show_{s} : true)' for s in all_sets)
    mono_vis       = f"showWords && showMono && ({_sets})"
    means_vis      = f"showMeans && showMono && ({_sets})"
    ellipse_vis    = f"showMono && ({_sets})"
    diph_vis       = f"showWords && showDiph && ({_sets})"
    diph_means_vis = f"showMeans && showDiph && ({_sets})"

    std_path: Path = project_root() / "male_standard.parquet"
    std_df = pl.read_parquet(std_path).drop_nulls(subset=["Frontness", "Roundness"])
    if "Closeness" in std_df.columns:
        std_df = std_df.rename({"Closeness": "Openness"})

    kwargs = dict(
        mono_df=mono_df, diph_df=diph_df, std_df=std_df,
        color_scale=color_scale,
        mono_vis=mono_vis, means_vis=means_vis, ellipse_vis=ellipse_vis,
        diph_vis=diph_vis, diph_means_vis=diph_means_vis, has_diph=has_diph,
    )
    proj_fo = _projection_chart(x_col="Frontness", y_col="Openness",  **kwargs)
    proj_fr = _projection_chart(x_col="Frontness", y_col="Roundness", **kwargs)
    proj_or = _projection_chart(x_col="Openness",  y_col="Roundness", **kwargs)

    return (
        alt.hconcat(proj_fo, proj_fr, proj_or)
        .add_params(words_param, means_param, mono_param, diph_param, *set_params.values())
        .properties(title=f"Bark Z Projections — {session}")
        .configure_view(strokeWidth=0)
    )


def save_bark_projections(session: str) -> None:
    df = pl.read_csv(session_dir(session) / f"{session}_formants.csv")
    has_diph = df["label"].str.contains(":").any()
    all_sets = sorted(df["set"].unique().to_list())
    set_colors = {s: Wells[s].value for s in all_sets}

    out_path: Path = session_dir(session) / f"{session}_bark_projections.html"
    html = build_bark_projections(session).to_html()
    html = _inject_controls(html, has_diph=has_diph, set_colors=set_colors)
    out_path.write_text(html)
    print(f"Created {out_path}")
