"""
04.generate_report.py

Generates a self-contained interactive HTML report combining:
  - Statistical methods (lay summary + formal, Stanford WitS style)
  - Summary tables with Excel and CSV download buttons
  - Four interactive Plotly boxplots with PDF download buttons (via kaleido)

Output: reports/thesis_results_report.html

Usage:
    .venv\\Scripts\\python.exe scripts/04.generate_report.py
"""

import base64
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data" / "processed" / "survey_clean.csv"
TABLE1_HTML = ROOT / "tables" / "table1_summary.html"
TABLE2_HTML = ROOT / "tables" / "table2_per_question.html"
TABLE1_CSV  = ROOT / "tables" / "table1_summary.csv"
TABLE2_CSV  = ROOT / "tables" / "table2_per_question.csv"
EXCEL_PATH  = ROOT / "tables" / "tables_summary.xlsx"
OUT_DIR     = ROOT / "reports"
OUT_HTML    = OUT_DIR / "thesis_results_report.html"

# ── Visual constants ───────────────────────────────────────────────────────────
HUE_ORDER   = ["No Resource", "PDF", "ChatGPT"]
HUE_PALETTE = {"No Resource": "#F8766D", "PDF": "#CD9600", "ChatGPT": "#00A9FF"}
FONT        = "Arial, Helvetica, sans-serif"

PLOT_CONFIG = {
    "displayModeBar": True,
    "toImageButtonOptions": {"format": "png", "scale": 3},
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "responsive": True,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 0.35) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def strip_style_tag(html: str) -> str:
    """Remove standalone <style>…</style> blocks (we supply our own CSS)."""
    return re.sub(r"<style>.*?</style>", "", html, flags=re.DOTALL).strip()


# ── Figure builders ────────────────────────────────────────────────────────────

def _box_trace(x_vals, y_vals, group: str, show_legend: bool) -> go.Box:
    color = HUE_PALETTE[group]
    return go.Box(
        x=x_vals,
        y=y_vals,
        name=group,
        marker_color=color,
        line=dict(color=color, width=1.5),
        fillcolor=hex_to_rgba(color, 0.35),
        boxpoints="all",
        jitter=0.35,
        pointpos=0,
        marker=dict(size=7, opacity=0.75, color=color),
        showlegend=show_legend,
    )


def _common_layout(fig: go.Figure, title: str, yaxis_title: str,
                   show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, family=FONT), x=0.5),
        xaxis=dict(
            showgrid=False, zeroline=False,
            showline=True, linecolor="#d0d0d0",
            tickfont=dict(size=12, family=FONT),
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=13, family=FONT)),
            showgrid=False, zeroline=False,
            showline=True, linecolor="#d0d0d0",
            tickfont=dict(size=12, family=FONT),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT, size=13),
        boxmode="group",
        showlegend=show_legend,
        legend=dict(
            title=dict(text="Group", font=dict(size=12, family=FONT)),
            font=dict(size=12, family=FONT),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#d0d0d0",
            borderwidth=1,
        ),
        margin=dict(l=65, r=30, t=65, b=55),
        height=460,
    )
    return fig


def build_fig1(df: pd.DataFrame) -> go.Figure:
    var_map = {
        "n_correct":   "Total Correct",
        "n_incorrect": "Total Incorrect",
        "n_not_sure":  "Total Not Sure",
    }
    df_long = (
        df[["group_label"] + list(var_map)]
        .melt(id_vars="group_label", var_name="_col", value_name="count")
        .assign(metric=lambda x: x["_col"].map(var_map))
    )
    df_long["metric"] = pd.Categorical(df_long["metric"], categories=list(var_map.values()))

    fig = go.Figure()
    for group in HUE_ORDER:
        sub = df_long[df_long["group_label"] == group]
        fig.add_trace(_box_trace(sub["metric"].tolist(), sub["count"].tolist(),
                                 group, show_legend=True))
    _common_layout(fig, "Knowledge Outcomes by Group", "Count (out of 12)", show_legend=True)
    return fig


def build_fig2(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for group in HUE_ORDER:
        sub = df[df["group_label"] == group]
        fig.add_trace(_box_trace(
            ["Percent Correct"] * len(sub),
            sub["pct_correct_of_attempted"].tolist(),
            group, show_legend=False,
        ))
    _common_layout(fig, "Percent Correct by Group", "Correct of Attempted (%)", show_legend=False)
    return fig


def build_fig3(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for group in HUE_ORDER:
        sub = df[df["group_label"] == group]
        fig.add_trace(_box_trace(
            ["Completion Time"] * len(sub),
            sub["duration_min"].tolist(),
            group, show_legend=False,
        ))
    _common_layout(fig, "Completion Time by Group", "Time (minutes)", show_legend=False)
    return fig


def build_fig4(df: pd.DataFrame) -> go.Figure:
    var_map = {
        "self_knowledge_tdi":       "TDI Knowledge",
        "self_confidence_avulsion": "Avulsion Confidence",
        "self_confidence_fracture": "Fracture Confidence",
        "self_confidence_mean":     "Average Self-Assessment",
    }
    df_long = (
        df[["group_label"] + list(var_map)]
        .melt(id_vars="group_label", var_name="_col", value_name="rating")
        .assign(metric=lambda x: x["_col"].map(var_map))
    )
    df_long["metric"] = pd.Categorical(df_long["metric"], categories=list(var_map.values()))

    fig = go.Figure()
    for group in HUE_ORDER:
        sub = df_long[df_long["group_label"] == group]
        fig.add_trace(_box_trace(sub["metric"].tolist(), sub["rating"].tolist(),
                                 group, show_legend=False))
    _common_layout(fig, "Self-Assessment by Group", "Rating (0\u201310)", show_legend=False)
    return fig


def fig_to_html_div(fig: go.Figure, div_id: str) -> str:
    config = {**PLOT_CONFIG, "toImageButtonOptions": {
        **PLOT_CONFIG["toImageButtonOptions"], "filename": div_id,
    }}
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config=config, div_id=div_id)


def fig_to_pdf_b64(fig: go.Figure) -> str:
    pdf_bytes = pio.to_image(fig, format="pdf", width=900, height=520)
    return base64.b64encode(pdf_bytes).decode()


# ── Methods text ───────────────────────────────────────────────────────────────

LAY_SUMMARY = """\
<p>
This study compared the effect of three different reference resources on emergency medicine
providers' knowledge of dental trauma management. Participants were randomly assigned to one of
three groups: <strong>No Resource</strong> (no reference material provided),
<strong>PDF</strong> (a printed reference guide), or <strong>ChatGPT</strong> (an AI assistant).
Each participant answered 12 case-based questions covering tooth avulsion, tooth fracture, and
antibiotic indications. We recorded how many questions each person answered correctly,
incorrectly, or marked "not sure," as well as how long they took to complete the survey and how
confident they felt in their knowledge before starting.
</p>
<p>
Because the study enrolled a small number of participants (18 total) and the outcome variables
are counts or ordinal ratings — not continuous, normally distributed measurements — we used
non-parametric statistical tests that do not require a bell-curve distribution. The tables show
summary statistics (median and interquartile range for numeric variables; counts and percentages
for categorical ones) alongside p-values comparing the three groups. The figures display the
same data as interactive boxplots with individual data points overlaid so you can see the full
spread of responses. None of the between-group comparisons reached statistical significance
(p&nbsp;&lt;&nbsp;0.05), which is expected given the small sample size — these results are
preliminary and intended to inform the design of a larger, adequately powered study.
</p>
"""

FORMAL_METHODS = """\
<h3>Study Design</h3>
<p>
We conducted a three-arm, randomized, survey-based study. Participants were randomly assigned
using block randomization embedded within the Qualtrics survey platform (Qualtrics, Provo, UT)
to one of three resource conditions: No Resource, PDF reference guide, or ChatGPT (OpenAI,
GPT-4). All participants completed an identical 12-item clinical vignette assessment comprising
three cases — primary tooth avulsion (Case 1), permanent tooth avulsion (Case 2), and
complicated crown fracture of a permanent incisor (Case 3) — with four questions per case
addressing injury identification, treatment selection, imaging, and antibiotic indication.
</p>

<h3>Outcome Variables</h3>
<p>
The primary outcomes were the number of correct, incorrect, and "not sure" responses (each out
of 12) and the percent of attempted questions answered correctly. Secondary outcomes were survey
completion time (minutes) and pre-assessment self-rated knowledge and confidence on 0–10 Likert
scales: self-rated TDI knowledge, avulsion management confidence, fracture management
confidence, and their mean (average self-assessment score).
</p>

<h3>Statistical Analysis</h3>
<p>
Continuous and ordinal outcomes were summarized as median (interquartile range, IQR).
Categorical variables were reported as n (%). Between-group comparisons for continuous and
ordinal outcomes were performed using the Kruskal-Wallis H-test. Categorical group differences
were assessed using Fisher's exact test for tables with small expected cell counts. Per-question
correctness rates were compared across the three groups using Fisher's exact test; p-values were
adjusted for multiple comparisons using the Bonferroni correction (12 tests; adjusted
α&nbsp;=&nbsp;0.0042). All analyses were performed in Python 3.12.0 using
<em>pandas</em>&nbsp;3.0, <em>scipy</em>&nbsp;1.17, and <em>statsmodels</em>&nbsp;0.14.
Statistical significance was set at α&nbsp;=&nbsp;0.05 for all primary comparisons.
</p>

<h3>Figures</h3>
<p>
Boxplots display the median (horizontal line), interquartile range (box boundaries), and
1.5&times; IQR (whiskers) for each group. Individual observations are overlaid as jittered
points to show the full data distribution given the small sample size. Groups are color-coded
consistently across all figures: No Resource
(<span style="color:#F8766D;font-weight:bold;">#F8766D</span>), PDF
(<span style="color:#CD9600;font-weight:bold;">#CD9600</span>), ChatGPT
(<span style="color:#00A9FF;font-weight:bold;">#00A9FF</span>). Figures were generated using
Plotly&nbsp;6.7 and are interactive — hover over any element for exact values, and use the
camera icon in the toolbar to export a PNG at the current display size.
</p>
"""

# ── Table legends ─────────────────────────────────────────────────────────────

TABLE_LEGENDS = {
    "table1": (
        "<strong>Table 1.</strong> Summary statistics for all participants and by experimental "
        "group. Continuous and ordinal variables are presented as median (interquartile range); "
        "categorical variables as n&nbsp;(%). P-values are from the Kruskal-Wallis H-test "
        "(continuous/ordinal outcomes) or Fisher's exact test (categorical outcomes). "
        "NT&nbsp;=&nbsp;not tested (cell counts too small for reliable inference). "
        "Groups: No Resource n&nbsp;=&nbsp;7, PDF n&nbsp;=&nbsp;5, ChatGPT n&nbsp;=&nbsp;6."
    ),
    "table2": (
        "<strong>Table 2.</strong> Per-question correctness rates by experimental group. "
        "Values represent the number of correct responses out of the group total, with the "
        "percentage in parentheses. Unadjusted p-values are from Fisher's exact test; "
        "corrected p-values were adjusted for multiple comparisons using the Bonferroni "
        "method (12 tests; adjusted significance threshold &alpha;&nbsp;=&nbsp;0.0042). "
        "No question reached statistical significance after correction."
    ),
}

# ── Figure legends ─────────────────────────────────────────────────────────────

FIGURE_LEGENDS = [
    (
        "<strong>Figure 1. Knowledge outcomes by experimental group.</strong> "
        "Number of correct, incorrect, and \u2018not sure\u2019 responses (out of 12 "
        "questions) for participants in each resource condition. Boxes show median and "
        "interquartile range; whiskers extend to 1.5&times; IQR; individual responses "
        "overlaid as jittered points. No statistically significant between-group "
        "differences were observed (Kruskal-Wallis: correct p&nbsp;=&nbsp;0.577, "
        "incorrect p&nbsp;=&nbsp;0.577, not sure p&nbsp;=&nbsp;0.405)."
    ),
    (
        "<strong>Figure 2. Percent correct of attempted questions by experimental group.</strong> "
        "Proportion of answered questions (excluding \u2018not sure\u2019 responses) that were "
        "correct, expressed as a percentage. Boxes show median and interquartile range; "
        "whiskers extend to 1.5&times; IQR. No statistically significant difference was "
        "observed across groups (Kruskal-Wallis p&nbsp;=&nbsp;0.577)."
    ),
    (
        "<strong>Figure 3. Survey completion time by experimental group.</strong> "
        "Time from survey start to submission, in minutes. One participant in the No Resource "
        "group took substantially longer than others (&gt;100&nbsp;min), visible as an outlier. "
        "No statistically significant difference was observed across groups "
        "(Kruskal-Wallis p&nbsp;=&nbsp;0.145)."
    ),
    (
        "<strong>Figure 4. Pre-assessment self-rated knowledge and confidence by experimental "
        "group.</strong> Scores on 0\u201310 Likert scales completed before viewing any cases. "
        "TDI&nbsp;=&nbsp;traumatic dental injury knowledge; Avulsion and Fracture = management "
        "confidence for each injury type; Average = mean of the three ratings. No statistically "
        "significant between-group differences were observed (Kruskal-Wallis: TDI knowledge "
        "p&nbsp;=&nbsp;0.639, avulsion confidence p&nbsp;=&nbsp;0.620, fracture confidence "
        "p&nbsp;=&nbsp;0.759, average p&nbsp;=&nbsp;0.761)."
    ),
]

# ── HTML template ──────────────────────────────────────────────────────────────

CSS = """\
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 15px;
    line-height: 1.65;
    color: #2c2c2c;
    background: #f5f6f8;
}

.layout {
    display: flex;
    min-height: 100vh;
}

/* ── Sidebar ── */
.sidebar {
    width: 230px;
    flex-shrink: 0;
    background: #ffffff;
    border-right: 1px solid #e0e0e0;
    padding: 28px 16px 28px 20px;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
}

.sidebar h2 {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin-bottom: 14px;
}

.sidebar nav ul {
    list-style: none;
    padding: 0;
}

.sidebar nav ul li { margin: 0; }

.sidebar nav ul li a {
    display: block;
    padding: 5px 8px;
    color: #444;
    text-decoration: none;
    font-size: 13px;
    border-radius: 4px;
    transition: background 0.15s;
}

.sidebar nav ul li a:hover { background: #f0f2f5; color: #1a1a1a; }

.sidebar nav ul li.section-head > a {
    font-weight: 700;
    color: #222;
    margin-top: 10px;
    font-size: 13px;
}

.sidebar nav ul li.sub > a {
    padding-left: 20px;
    color: #555;
}

/* ── Main content ── */
.content {
    flex: 1;
    max-width: 960px;
    padding: 40px 48px 60px 48px;
}

.report-title {
    font-size: 26px;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 6px;
}

.report-subtitle {
    font-size: 15px;
    color: #666;
    margin-bottom: 36px;
}

/* ── Sections ── */
.section {
    background: #ffffff;
    border-radius: 8px;
    border: 1px solid #e4e6ea;
    padding: 32px 36px;
    margin-bottom: 28px;
}

.section h2 {
    font-size: 19px;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid #f0f2f5;
}

.section h3 {
    font-size: 15px;
    font-weight: 700;
    color: #333;
    margin: 20px 0 8px;
}

.section p { margin-bottom: 12px; color: #3a3a3a; }

.tab-container { margin-bottom: 6px; }

.tabs {
    display: flex;
    gap: 4px;
    border-bottom: 2px solid #e0e0e0;
    margin-bottom: 20px;
}

.tab-btn {
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    color: #666;
    transition: color 0.15s, border-color 0.15s;
    margin-bottom: -2px;
}

.tab-btn.active { color: #2563eb; border-bottom-color: #2563eb; }
.tab-btn:hover:not(.active) { color: #333; }

.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ── Tables ── */
.table-wrap { overflow-x: auto; margin-bottom: 14px; }

.table-wrap table {
    border-collapse: collapse;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 13px;
    width: 100%;
}

.table-wrap th, .table-wrap td {
    border: 1px solid #d8dce3;
    padding: 7px 12px;
    text-align: left;
}

.table-wrap th {
    background: #f0f3f8;
    font-weight: 700;
    color: #1a1a1a;
}

.table-wrap tr:nth-child(even) { background: #fafbfc; }
.table-wrap tr:hover { background: #f3f5f9; }

/* ── Download buttons ── */
.download-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    border: 1.5px solid;
    border-radius: 5px;
    cursor: pointer;
    background: white;
    text-decoration: none;
    transition: background 0.15s, color 0.15s;
}

.btn-excel  { color: #1e7e34; border-color: #1e7e34; }
.btn-excel:hover  { background: #1e7e34; color: white; }
.btn-csv    { color: #0066cc; border-color: #0066cc; }
.btn-csv:hover    { background: #0066cc; color: white; }
.btn-pdf    { color: #c0392b; border-color: #c0392b; }
.btn-pdf:hover    { background: #c0392b; color: white; }
.btn-png    { color: #0891b2; border-color: #0891b2; }
.btn-png:hover    { background: #0891b2; color: white; }

/* ── Figure sections ── */
.figure-block { margin-bottom: 28px; overflow-x: auto; }
.figure-block h3 { font-size: 15px; font-weight: 700; color: #333; margin-bottom: 8px; }
.figure-plot { border-radius: 6px; overflow: visible; border: 1px solid #e4e6ea; display: inline-block; min-width: 100%; }

/* ── Figure/table legends ── */
.legend-text {
    font-size: 12.5px;
    color: #4a4a4a;
    line-height: 1.6;
    margin: 10px 0 10px;
    padding: 10px 14px;
    background: #f8f9fb;
    border-left: 3px solid #d0d5de;
    border-radius: 0 4px 4px 0;
}

/* ── Resize controls ── */
.resize-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px 18px;
    margin: 8px 0 6px;
    font-size: 12px;
    color: #555;
    padding: 8px 12px;
    background: #f5f6f8;
    border-radius: 5px;
    border: 1px solid #e4e6ea;
}
.resize-controls label { font-weight: 600; white-space: nowrap; }
.resize-controls input[type=range] { width: 140px; cursor: pointer; accent-color: #0891b2; }
.resize-controls input[type=checkbox] { cursor: pointer; accent-color: #0891b2; }
.resize-val { display: inline-block; width: 44px; color: #0891b2; font-weight: 700; }
.resize-hrow { display: none; align-items: center; gap: 8px; }

/* ── Color chips ── */
.chip {
    display: inline-block;
    width: 11px; height: 11px;
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 4px;
}

@media (max-width: 768px) {
    .layout { flex-direction: column; }
    .sidebar { width: 100%; height: auto; position: relative; border-right: none;
               border-bottom: 1px solid #e0e0e0; }
    .content { padding: 20px 16px 40px; }
}
"""

JS = """\
function dlFile(b64, filename, mime) {
    const a = document.createElement('a');
    a.href = 'data:' + mime + ';base64,' + b64;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function dlPng(divId, filename) {
    Plotly.downloadImage(document.getElementById(divId), {
        format: 'png', scale: 3, filename: filename
    });
}

function updateFigSize(divId) {
    const wSlider = document.getElementById(divId + '-w');
    const hSlider = document.getElementById(divId + '-h');
    const locked  = document.getElementById(divId + '-lock').checked;
    const defAR   = parseFloat(document.getElementById(divId + '-ar').dataset.ratio);
    const w = parseInt(wSlider.value);
    let h;
    if (locked) {
        h = Math.round(w * defAR);
        hSlider.value = h;
    } else {
        h = parseInt(hSlider.value);
    }
    document.getElementById(divId + '-wval').textContent = w + 'px';
    document.getElementById(divId + '-hval').textContent = h + 'px';
    document.getElementById(divId + '-hrow').style.display = locked ? 'none' : 'flex';
    Plotly.relayout(divId, {width: w, height: h});
    // Expand the container frame to match the new figure size
    const plotEl = document.getElementById(divId);
    if (plotEl) {
        const frame = plotEl.closest('.figure-plot');
        if (frame) { frame.style.width = w + 'px'; frame.style.maxWidth = 'none'; }
    }
}
"""


def download_btn(label: str, b64: str, filename: str, mime: str, css_class: str) -> str:
    return (
        f'<button class="btn {css_class}" '
        f'onclick="dlFile(\'{b64}\',\'{filename}\',\'{mime}\')">'
        f'{label}</button>'
    )


def figure_section(fig_num: int, title: str, fig_div: str, pdf_b64: str,
                   pdf_filename: str, png_filename: str, legend: str,
                   default_w: int = 800, default_h: int = 460) -> str:
    """Render a complete figure block with resize controls, download buttons, and legend."""
    div_id  = f"plotly-fig{fig_num}"
    ar      = round(default_h / default_w, 4)
    pdf_btn = download_btn("⬇ Download PDF", pdf_b64, pdf_filename, "application/pdf", "btn-pdf")
    png_btn = (
        f'<button class="btn btn-png" '
        f'onclick="dlPng(\'{div_id}\',\'{png_filename.replace(".png","")}\')">&#11015; Download PNG</button>'
    )
    resize = f"""\
<div class="resize-controls">
  <span>&#8596; Resize figure:</span>
  <label>W: <span id="{div_id}-wval" class="resize-val">{default_w}px</span></label>
  <input type="range" id="{div_id}-w" min="400" max="1200" value="{default_w}"
         oninput="updateFigSize('{div_id}')">
  <label>
    <input type="checkbox" id="{div_id}-lock" checked onchange="updateFigSize('{div_id}')">
    Lock aspect ratio
  </label>
  <span id="{div_id}-ar" data-ratio="{ar}" style="display:none"></span>
  <div id="{div_id}-hrow" class="resize-hrow">
    <label>H: <span id="{div_id}-hval" class="resize-val">{default_h}px</span></label>
    <input type="range" id="{div_id}-h" min="200" max="900" value="{default_h}"
           oninput="updateFigSize('{div_id}')">
  </div>
</div>"""
    return f"""\
<div class="figure-block" id="fig{fig_num}">
  <h3>Figure {fig_num} \u2014 {title}</h3>
  {resize}
  <div class="figure-plot">{fig_div}</div>
  <div class="download-bar" style="margin-top:10px;">{png_btn}{pdf_btn}</div>
  <div class="legend-text">{legend}</div>
</div>
"""


def table_section(section_id: str, title: str, table_html: str,
                  csv_b64: str, csv_name: str, legend: str,
                  excel_b64: str = None, excel_name: str = None) -> str:
    """Render a complete table block with download buttons and a legend."""
    btns = download_btn("⬇ Download CSV", csv_b64, csv_name, "text/csv", "btn-csv")
    if excel_b64:
        btns += "\n    " + download_btn(
            "⬇ Download Excel (both tables)", excel_b64, excel_name,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "btn-excel",
        )
    return f"""\
<div id="{section_id}" style="margin-bottom:28px;">
  <h3 style="font-size:16px;font-weight:700;margin-bottom:12px;">{title}</h3>
  <div class="table-wrap">{table_html}</div>
  <div class="download-bar">{btns}</div>
  <div class="legend-text">{legend}</div>
</div>
"""


def build_html(figures: list[dict], table1_html: str, table2_html: str,
               t1_csv_b64: str, t2_csv_b64: str, excel_b64: str) -> str:
    """Assemble the full HTML report string."""
    fig_sections = "\n".join(
        figure_section(
            i + 1, f["title"], f["div"], f["pdf_b64"],
            f["pdf_filename"], f["png_filename"], f["legend"],
        )
        for i, f in enumerate(figures)
    )

    t1_section = table_section(
        "table1", "Table 1 \u2014 Summary Statistics",
        table1_html, t1_csv_b64, "table1_summary.csv",
        TABLE_LEGENDS["table1"],
        excel_b64, "tables_summary.xlsx",
    )
    t2_section = table_section(
        "table2", "Table 2 \u2014 Per-question Correctness by Group",
        table2_html, t2_csv_b64, "table2_per_question.csv",
        TABLE_LEGENDS["table2"],
    )

    sidebar_figs = "\n".join(
        f'    <li class="sub"><a href="#fig{i+1}">Fig {i+1}: {f["short"]}</a></li>'
        for i, f in enumerate(figures)
    )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dental Trauma Study — Results Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>{CSS}</style>
</head>
<body>
<div class="layout">

  <!-- ── Sidebar ── -->
  <aside class="sidebar">
    <h2>Contents</h2>
    <nav>
      <ul>
        <li class="section-head"><a href="#methods">Methods</a></li>
        <li class="sub"><a href="#lay-summary">Lay Summary</a></li>
        <li class="sub"><a href="#formal-methods">Formal Methods</a></li>
        <li class="section-head"><a href="#results">Results</a></li>
        <li class="sub"><a href="#table1">Table 1</a></li>
        <li class="sub"><a href="#table2">Table 2</a></li>
{sidebar_figs}
      </ul>
    </nav>
  </aside>

  <!-- ── Main content ── -->
  <main class="content">
    <div class="report-title">Dental Trauma Study — Results Report</div>
    <div class="report-subtitle">
      Three-arm randomized survey: No Resource &nbsp;·&nbsp; PDF &nbsp;·&nbsp; ChatGPT
      &nbsp;|&nbsp; N&nbsp;=&nbsp;18 participants
    </div>

    <!-- Methods -->
    <section class="section" id="methods">
      <h2>Methods</h2>

      <div id="lay-summary">
        <h3>Lay Summary</h3>
        {LAY_SUMMARY}
      </div>

      <div id="formal-methods" style="margin-top:24px;">
        <h3>Formal Methods</h3>
        {FORMAL_METHODS}
      </div>
    </section>

    <!-- Results -->
    <section class="section" id="results">
      <h2>Results</h2>

      <!-- Tables -->
      <div style="margin-bottom:32px;">
        {t1_section}
      </div>
      <div style="margin-bottom:32px;">
        {t2_section}
      </div>

      <!-- Figures -->
      <div style="margin-top:8px;">
        <p style="font-size:13px;color:#666;margin-bottom:20px;">
          <strong>Color key:</strong>&nbsp;
          <span class="chip" style="background:#F8766D;"></span>No Resource&ensp;
          <span class="chip" style="background:#CD9600;"></span>PDF&ensp;
          <span class="chip" style="background:#00A9FF;"></span>ChatGPT
          &nbsp;&mdash;&nbsp;Legend shown on Figure 1 only; color coding is consistent across
          all figures.
        </p>
        {fig_sections}
      </div>
    </section>

  </main>
</div>

<script>{JS}</script>
</body>
</html>
"""


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["group_label"] = pd.Categorical(df["group_label"], categories=HUE_ORDER, ordered=True)

    print("Building Plotly figures...")
    figs_raw = [
        (build_fig1(df), "Knowledge Outcomes by Group", "Knowledge Outcomes",
         "fig1_knowledge_outcomes.pdf", "fig1_knowledge_outcomes.png"),
        (build_fig2(df), "Percent Correct by Group",    "Percent Correct",
         "fig2_pct_correct.pdf",        "fig2_pct_correct.png"),
        (build_fig3(df), "Completion Time by Group",    "Completion Time",
         "fig3_completion_time.pdf",    "fig3_completion_time.png"),
        (build_fig4(df), "Self-Assessment by Group",    "Self-Assessment",
         "fig4_self_assessment.pdf",    "fig4_self_assessment.png"),
    ]

    print("Rendering PDFs via kaleido (this may take a moment)...")
    figures = []
    for i, (fig, title, short, pdf_name, png_name) in enumerate(figs_raw, start=1):
        print(f"  Figure {i}: {title}")
        figures.append({
            "title":        title,
            "short":        short,
            "div":          fig_to_html_div(fig, f"plotly-fig{i}"),
            "pdf_b64":      fig_to_pdf_b64(fig),
            "pdf_filename": pdf_name,
            "png_filename": png_name,
            "legend":       FIGURE_LEGENDS[i - 1],
        })

    print("Encoding table assets...")
    table1_html = strip_style_tag(TABLE1_HTML.read_text(encoding="utf-8"))
    table2_html = strip_style_tag(TABLE2_HTML.read_text(encoding="utf-8"))
    t1_csv_b64  = b64_file(TABLE1_CSV)
    t2_csv_b64  = b64_file(TABLE2_CSV)
    excel_b64   = b64_file(EXCEL_PATH)

    print("Assembling HTML...")
    html = build_html(figures, table1_html, table2_html, t1_csv_b64, t2_csv_b64, excel_b64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")

    size_mb = OUT_HTML.stat().st_size / 1_000_000
    print(f"\nReport saved: {OUT_HTML.relative_to(ROOT)}")
    print(f"File size: {size_mb:.1f} MB")
    print("Open the HTML file in any browser to view the report.")
