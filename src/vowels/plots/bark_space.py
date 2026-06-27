from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from ..paths import session_dir
from ..schema import Wells
from .vowel_space import _text_color

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
                    "<b>%{customdata[0]}</b> (%{meta[set]})<br>"
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

            # Group by token: one line per token connecting point 1 → 2
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
                            "Openness: %{y:.2f}<br>"
                            "Frontness: %{x:.2f}<br>"
                            "Roundness: %{z:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

            # Diphthong means trajectory
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
        "Plotly.restyle(gd,{visible:gd.data.map(function(t){return isOn(t);})},"
        "gd.data.map(function(_,i){return i;}));"
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
    # Sidebar is the first element in <body>; body itself is the flex container
    html = html.replace("<body>", f"<body>{sidebar}", 1)
    # Let the Plotly graph div fill the remaining flex space
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
    html = fig.to_html(
        div_id=DIV_ID,
        include_plotlyjs=True,
        full_html=True,
        post_script="setupToggles(gd);",
    )
    html = _inject_bark_controls(html, has_diph=has_diph, set_colors=set_colors)

    out_path: Path = session_dir(session) / f"{session}_bark_space.html"
    out_path.write_text(html)
    print(f"Created {out_path}")
