import re
from pathlib import Path

import altair as alt
import polars as pl

from ..paths import project_root, session_dir
from ..schema import Wells
from .ellipse import precompute_ellipse


def _text_color(hex_color: str) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return "#333" if (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.55 else "white"


def build_chart(session: str) -> alt.LayerChart:
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

    # Granularity / type toggles
    words_param: alt.Parameter = alt.param(name="showWords", value=True)
    means_param: alt.Parameter = alt.param(name="showMeans", value=True)
    mono_param: alt.Parameter = alt.param(name="showMono", value=True)
    diph_param: alt.Parameter = alt.param(name="showDiph", value=True)

    # One boolean param per lexical set, toggled by injected set buttons
    set_params: dict[str, alt.Parameter] = {
        s: alt.param(name=f"show_{s}", value=True) for s in all_sets
    }

    # Each clause passes (true) for non-matching rows and gates on the param for matches
    _sets: str = " && ".join(
        f'(datum.set === "{s}" ? show_{s} : true)' for s in all_sets
    )

    mono_vis:       str = f"showWords && showMono && ({_sets})"
    means_vis:      str = f"showMeans && showMono && ({_sets})"
    ellipse_vis:    str = f"showMono && ({_sets})"
    diph_vis:       str = f"showWords && showDiph && ({_sets})"
    diph_means_vis: str = f"showMeans && showDiph && ({_sets})"

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
                    alt.when(ellipse_vis).then(alt.value(0.7)).otherwise(alt.value(0.0))
                ),
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
                "set:N", scale=color_scale, legend=alt.Legend(title="Lexical set")
            ),
            opacity=(
                alt.when(mono_vis).then(alt.value(0.85)).otherwise(alt.value(0.0))
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
                alt.when(means_vis).then(alt.value(1.0)).otherwise(alt.value(0.0))
            ),
            tooltip=[
                alt.Tooltip("set:N", title="Set"),
                alt.Tooltip("F1:Q", title="F1 mean (Hz)", format=".0f"),
                alt.Tooltip("F2:Q", title="F2 mean (Hz)", format=".0f"),
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
        diph_means_df: pl.DataFrame = (
            diph_df.group_by(["set", "point_num"])
            .agg(pl.col("F1").mean(), pl.col("F2").mean())
            .sort(["set", "point_num"])
        )

        diph_trail_layer: alt.Chart = (
            alt.Chart(diph_df)
            .mark_trail()
            .encode(
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                size=alt.Size(
                    "point_num:Q",
                    scale=alt.Scale(domain=[1, 2], range=[1, 4]),
                    legend=None,
                ),
                detail="token:N",
                order=alt.Order("point_num:O"),
                opacity=(
                    alt.when(diph_vis).then(alt.value(0.6)).otherwise(alt.value(0.0))
                ),
                tooltip=[
                    alt.Tooltip("word:N", title="Word"),
                    alt.Tooltip("set:N", title="Set"),
                    alt.Tooltip("F1:Q", title="F1 (Hz)", format=".0f"),
                    alt.Tooltip("F2:Q", title="F2 (Hz)", format=".0f"),
                ],
            )
        )
        layers.append(diph_trail_layer)

        diph_mean_trail_layer: alt.Chart = (
            alt.Chart(diph_means_df)
            .mark_trail()
            .encode(
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                size=alt.Size(
                    "point_num:Q",
                    scale=alt.Scale(domain=[1, 2], range=[3, 9]),
                    legend=None,
                ),
                detail="set:N",
                order=alt.Order("point_num:O"),
                opacity=(
                    alt.when(diph_means_vis)
                    .then(alt.value(0.9))
                    .otherwise(alt.value(0.0))
                ),
            )
        )
        layers.append(diph_mean_trail_layer)

        diph_mean_pts_layer: alt.Chart = (
            alt.Chart(diph_means_df)
            .mark_point(shape="diamond", filled=True, stroke="white", strokeWidth=1.5)
            .encode(
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                size=alt.Size(
                    "point_num:Q",
                    scale=alt.Scale(domain=[1, 2], range=[80, 200]),
                    legend=None,
                ),
                opacity=(
                    alt.when(diph_means_vis)
                    .then(alt.value(1.0))
                    .otherwise(alt.value(0.0))
                ),
                tooltip=[
                    alt.Tooltip("set:N", title="Set"),
                    alt.Tooltip("point_num:O", title="Point"),
                    alt.Tooltip("F1:Q", title="F1 mean (Hz)", format=".0f"),
                    alt.Tooltip("F2:Q", title="F2 mean (Hz)", format=".0f"),
                ],
            )
        )
        layers.append(diph_mean_pts_layer)

    return (
        alt.layer(*layers)
        .add_params(words_param, means_param, mono_param, diph_param, *set_params.values())
        .resolve_scale(x="shared", y="shared")
        .properties(width=650, height=550, title=f"Vowel Space — {session}")
        .configure_view(strokeWidth=0)
    )


def _inject_controls(
    html: str, *, has_diph: bool, set_colors: dict[str, str]
) -> str:
    def btn(sig: str, label: str) -> str:
        return f'<button class="vt-btn active" data-signal="{sig}">{label}</button>'

    def set_btn(name: str, color: str) -> str:
        tc = _text_color(color)
        return (
            f'<button class="vt-set-btn active" data-signal="show_{name}" '
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
        "#vt-controls{"
        "width:160px;flex-shrink:0;font-family:sans-serif;"
        "}"
        ".vt-section{margin-bottom:14px;}"
        ".vt-label{"
        "font-size:10px;font-weight:bold;color:#aaa;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px;"
        "}"
        ".vt-btn{"
        "display:block;width:100%;padding:6px 8px;margin-bottom:4px;"
        "border:none;border-radius:6px;font-size:13px;font-weight:bold;"
        "cursor:pointer;background:#2c7bb6;color:white;"
        "text-align:center;box-sizing:border-box;"
        "transition:background .15s,color .15s;"
        "}"
        ".vt-btn:not(.active){background:#c8d8e8;color:#5a7a99;}"
        ".vt-set-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px;}"
        ".vt-set-btn{"
        "padding:4px 4px;border:none;border-radius:4px;font-size:11px;"
        "font-weight:bold;cursor:pointer;text-align:center;"
        "transition:opacity .15s;"
        "}"
        ".vt-set-btn:not(.active){opacity:0.25;}"
        "</style>"
    )

    js = (
        "<script>"
        "function setupToggles(view){"
        # Granularity / type buttons (each has data-signal; All button has none so guard skips it)
        "document.querySelectorAll('.vt-btn').forEach(function(btn){"
        "btn.addEventListener('click',function(){"
        "var sig=this.dataset.signal;"
        "if(!sig)return;"
        "var active=this.classList.toggle('active');"
        "view.signal(sig,active).run();"
        "});"
        "});"
        # All button — flip every set to the opposite of the current all-on state
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
        "view.signal(btn.dataset.signal,next).run();"
        "});"
        "});"
        # Individual set buttons — also sync the All button after each click
        "document.querySelectorAll('.vt-set-btn').forEach(function(btn){"
        "btn.addEventListener('click',function(){"
        "var active=this.classList.toggle('active');"
        "view.signal(this.dataset.signal,active).run();"
        "syncAllBtn();"
        "});"
        "});"
        "}"
        "</script>"
    )

    html = re.sub(
        r"(vegaEmbed\(\"#vis\"[^)]*\))\s*\n(\s*\.catch)",
        r"\1\n        .then(function(result){setupToggles(result.view);})\n\2",
        html,
        count=1,
    )
    html = html.replace("</head>", css + js + "</head>", 1)

    # Wrap sidebar + vis in a flex row
    html = re.sub(
        r'(<div id="vis"></div>)',
        '<div style="display:flex;align-items:flex-start;gap:20px;padding:12px">'
        + sidebar
        + r"\1</div>",
        html,
        count=1,
    )

    return html


def save_chart(session: str) -> None:
    df: pl.DataFrame = pl.read_csv(session_dir(session) / f"{session}_formants.csv")
    has_diph: bool = df["label"].str.contains(":").any()
    all_sets: list[str] = sorted(df["set"].unique().to_list())
    set_colors: dict[str, str] = {s: Wells[s].value for s in all_sets}

    out_path: Path = session_dir(session) / f"{session}_vowel_space.html"
    html: str = build_chart(session).to_html()
    html = _inject_controls(html, has_diph=has_diph, set_colors=set_colors)
    out_path.write_text(html)
    print(f"Created {out_path}")
