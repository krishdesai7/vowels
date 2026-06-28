import re
from pathlib import Path

import altair as alt
import polars as pl

from ..paths import data_dir, session_dir
from ..schema import GROUPS, Wells
from .ellipse import precompute_ellipse

_DIPH_NAMES: frozenset[str] = frozenset(s for s in GROUPS.get("Diphthongs", []))


def _text_color(hex_color: str) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return "#333" if (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.55 else "white"


def build_chart(df: pl.DataFrame, session: str) -> alt.LayerChart | alt.FacetChart:
    is_diph: pl.Expr = pl.col("label").str.contains(":")
    mono_df: pl.DataFrame = df.filter(~is_diph)
    diph_df: pl.DataFrame = df.filter(is_diph)
    has_diph: bool = not diph_df.is_empty()

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

    mono_vis: str = f"showWords && showMono && ({_sets})"
    means_vis: str = f"showMeans && showMono && ({_sets})"
    ellipse_vis: str = f"showMono && ({_sets})"
    diph_vis: str = f"showWords && showDiph && ({_sets})"
    diph_means_vis: str = f"showMeans && showDiph && ({_sets})"

    # Layer 1: IPA reference text
    std_df: pl.DataFrame = pl.read_parquet(
        data_dir / "standards" / "male_standard.parquet"
    ).drop_nulls(subset=["F1", "F2"])
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
        layers.append(
            alt.Chart(pl.DataFrame(ellipse_records))
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

    # Layer 3: Monophthong tokens
    layers.append(
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

    # Layer 4: Per-set means (monophthongs only)
    layers.append(
        alt.Chart(
            mono_df.group_by("set").agg(
                pl.col("F1").mean(), pl.col("F2").mean()
            )
        )
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

    # Layers 5+: Diphthong tokens and mean trajectories with directional arrows
    if has_diph:
        diph_df = diph_df.with_columns(
            pl.col("label").str.split(":").list.first().alias("token"),
            pl.col("label")
            .str.split(":")
            .list.last()
            .cast(pl.Int32)
            .alias("point_num"),
        )
        diph_means_df: pl.DataFrame = (
            diph_df.group_by(["set", "point_num"])
            .agg(pl.col("F1").mean(), pl.col("F2").mean())
            .sort(["set", "point_num"])
        )

        # Axis spans for normalising arrowhead angles into screen space.
        # x=F2 reversed → screen moves left as F2 rises; y=F1 reversed → same.
        # Chart pixel dims (650×550) are folded in so angles are visually correct.
        F2_rng: float = float((mono_df["F2"].max() - mono_df["F2"].min()) or 1.0)  # type: ignore
        F1_rng: float = float((mono_df["F1"].max() - mono_df["F1"].min()) or 1.0)  # type: ignore
        W, H = 650.0, 550.0

        def _angle_expr() -> pl.Expr:
            return pl.arctan2(
                -(pl.col("F2") - pl.col("F2_s")) / F2_rng * W,
                -(pl.col("F1") - pl.col("F1_s")) / F1_rng * H,
            ).degrees().alias("angle")

        pt1_t: pl.DataFrame = diph_df.filter(pl.col("point_num") == 1)[
            ["token", "set", "F1", "F2"]
        ].rename({"F1": "F1_s", "F2": "F2_s"})
        pt2_t: pl.DataFrame = diph_df.filter(pl.col("point_num") == 2)[
            ["token", "set", "F1", "F2", "word"]
        ]
        tok_arr: pl.DataFrame = pt2_t.join(pt1_t, on=["token", "set"]).with_columns(
            _angle_expr()
        )

        pt1_m: pl.DataFrame = diph_means_df.filter(pl.col("point_num") == 1)[
            ["set", "F1", "F2"]
        ].rename({"F1": "F1_s", "F2": "F2_s"})
        pt2_m: pl.DataFrame = diph_means_df.filter(pl.col("point_num") == 2)[
            ["set", "F1", "F2"]
        ]
        mean_arr: pl.DataFrame = pt2_m.join(pt1_m, on="set").with_columns(
            _angle_expr()
        )

        ang_scale: alt.Scale = alt.Scale(domain=[-180, 180], range=[-180, 180])

        # Token lines
        layers.append(
            alt.Chart(diph_df)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
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
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
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
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                angle=alt.Angle("angle:Q", scale=ang_scale),
                opacity=alt.when(diph_vis)
                .then(alt.value(0.7))
                .otherwise(alt.value(0.0)),
                tooltip=[
                    alt.Tooltip("word:N", title="Word"),
                    alt.Tooltip("set:N", title="Set"),
                    alt.Tooltip("F1:Q", title="F1 (Hz)", format=".0f"),
                    alt.Tooltip("F2:Q", title="F2 (Hz)", format=".0f"),
                ],
            )
        )
        # Mean line
        layers.append(
            alt.Chart(diph_means_df)
            .mark_line(strokeWidth=5)
            .encode(
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
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
                x=alt.X("F2:Q", scale=alt.Scale(reverse=True)),
                y=alt.Y("F1:Q", scale=alt.Scale(reverse=True)),
                color=alt.Color("set:N", scale=color_scale, legend=None),
                angle=alt.Angle("angle:Q", scale=ang_scale),
                opacity=alt.when(diph_means_vis)
                .then(alt.value(1.0))
                .otherwise(alt.value(0.0)),
                tooltip=[
                    alt.Tooltip("set:N", title="Set"),
                    alt.Tooltip("F1:Q", title="F1 mean (Hz)", format=".0f"),
                    alt.Tooltip("F2:Q", title="F2 mean (Hz)", format=".0f"),
                ],
            )
        )

    return (
        alt.layer(*layers)
        .add_params(
            words_param, means_param, mono_param, diph_param, *set_params.values()
        )
        .resolve_scale(x="shared", y="shared")
        .properties(width=650, height=550, title=f"Vowel Space — {session}")
        .configure_view(strokeWidth=0)
    )


def _inject_controls(html: str, *, has_diph: bool, set_colors: dict[str, str]) -> str:
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

    groups_with_data = [
        (
            grp_name,
            "".join(set_btn(s, set_colors[s]) for s in grp_sets if s in set_colors),
        )
        for grp_name, grp_sets in GROUPS.items()
    ]
    groups_with_data = [(g, b) for g, b in groups_with_data if b]
    mid = (len(groups_with_data) + 1) // 2
    col1_html = "".join(
        f'<div class="vt-group-hdr">{g}</div>{b}' for g, b in groups_with_data[:mid]
    )
    col2_html = "".join(
        f'<div class="vt-group-hdr">{g}</div>{b}' for g, b in groups_with_data[mid:]
    )
    set_grid = (
        '<div class="vt-set-cols">'
        '<div class="vt-set-col">' + col1_html + "</div>"
        '<div class="vt-set-col">' + col2_html + "</div>"
        "</div>"
    )

    sidebar = (
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

    css = (
        "<style>"
        "#vt-controls{"
        "width:185px;flex-shrink:0;font-family:sans-serif;"
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
        ".vt-set-cols{display:flex;gap:4px;}"
        ".vt-set-col{flex:1;display:flex;flex-direction:column;gap:4px;}"
        ".vt-group-hdr{font-size:8px;color:#bbb;text-transform:uppercase;"
        "letter-spacing:.04em;margin-top:6px;margin-bottom:1px;}"
        ".vt-set-col>:first-child.vt-group-hdr{margin-top:0;}"
        ".vt-set-btn{"
        "padding:4px 2px;border:none;border-radius:4px;font-size:11px;"
        "font-weight:bold;cursor:pointer;text-align:center;"
        "transition:opacity .15s;"
        "}"
        ".vt-set-btn:not(.active){opacity:0.25;}"
        "</style>"
    )

    js = (
        "<script>"
        "function setupToggles(view){"
        "document.querySelectorAll('.vt-btn').forEach(function(btn){"
        "btn.addEventListener('click',function(){"
        "var sig=this.dataset.signal;"
        "if(!sig)return;"
        "var active=this.classList.toggle('active');"
        "view.signal(sig,active).run();"
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
        "view.signal(btn.dataset.signal,next).run();"
        "});"
        "});"
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
    from ..aggregate import load_points

    df: pl.DataFrame = load_points(session)
    has_diph: bool = df["label"].str.contains(":").any()
    all_sets: list[str] = sorted(df["set"].unique().to_list())
    set_colors: dict[str, str] = {s: Wells[s].value for s in all_sets}

    out_path: Path = session_dir(session) / f"{session}_vowel_space.html"
    html: str = build_chart(df, session).to_html()
    html = _inject_controls(html, has_diph=has_diph, set_colors=set_colors)
    out_path.write_text(html)
    print(f"Created {out_path}")
