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
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data" / "processed" / "survey_clean.csv"
TABLE1_HTML = ROOT / "tables" / "table1_summary.html"
TABLE2_HTML = ROOT / "tables" / "table2_per_question.html"
TABLE1_CSV   = ROOT / "tables" / "table1_summary.csv"
TABLE2_CSV   = ROOT / "tables" / "table2_per_question.csv"
TABLE3_CSV   = ROOT / "tables" / "table3_correlation.csv"
TABLE3_EXCEL = ROOT / "tables" / "table3_correlation.xlsx"
EXCEL_PATH   = ROOT / "tables" / "tables_summary.xlsx"
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
            group, show_legend=True,
        ))
    _common_layout(fig, "Percent Correct by Group", "Correct of Attempted (%)", show_legend=True)
    return fig


def build_fig3(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for group in HUE_ORDER:
        sub = df[df["group_label"] == group]
        fig.add_trace(_box_trace(
            ["Completion Time"] * len(sub),
            sub["duration_min"].tolist(),
            group, show_legend=True,
        ))
    _common_layout(fig, "Completion Time by Group", "Time (minutes)", show_legend=True)
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
                                 group, show_legend=True))
    _common_layout(fig, "Self-Assessment by Group", "Rating (0\u201310)", show_legend=True)
    return fig


def fmt_pval(p: float) -> str:
    """Format a p-value; returns 'p < 0.001' when below threshold, else 'p = 0.xxx'."""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def build_scatter(df_sub: pd.DataFrame, x_col: str, y_col: str,
                  x_label: str, y_label: str, title: str, color: str) -> go.Figure:
    """
    Build a single Plotly scatter plot with a Spearman rho annotation in the top-right.
    No trendline; single color; no legend.
    """
    import numpy as np

    valid = df_sub[[x_col, y_col]].dropna()
    x = valid[x_col].reset_index(drop=True)
    y = valid[y_col].reset_index(drop=True)
    rho, pval = spearmanr(x, y)

    # Add a small random jitter to integer-valued axes to reduce overplotting
    rng = np.random.default_rng(42)
    x_jit = (x + rng.uniform(-0.15, 0.15, size=len(x))).tolist()
    y_jit = (y + rng.uniform(-0.15, 0.15, size=len(y))).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_jit,
        y=y_jit,
        mode="markers",
        marker=dict(size=10, color=color, opacity=0.70,
                    line=dict(color="white", width=0.5)),
        showlegend=False,
        hovertemplate=(
            f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.0f}}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, family=FONT), x=0.5),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=13, family=FONT)),
            range=[-0.4, 10.7],
            tick0=0, dtick=2,
            showgrid=False, zeroline=False,
            showline=True, linecolor="#d0d0d0",
            tickfont=dict(size=12, family=FONT),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=13, family=FONT)),
            range=[-0.4, 12.7],
            tick0=0, dtick=2,
            showgrid=False, zeroline=False,
            showline=True, linecolor="#d0d0d0",
            tickfont=dict(size=12, family=FONT),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT, size=13),
        showlegend=False,
        margin=dict(l=65, r=30, t=65, b=55),
        height=460,
    )
    return fig


def build_correlation_table(df: pd.DataFrame) -> str:
    """
    Compute Spearman correlations between self_confidence_mean and n_correct
    for the full sample and each experimental group.
    Includes Bonferroni-adjusted p-values (4 tests; adjusted α = 0.0125).
    Returns an HTML table string.
    """
    subsets = [
        ("All Participants", df),
        ("No Resource",      df[df["group_label"] == "No Resource"]),
        ("PDF",              df[df["group_label"] == "PDF"]),
        ("ChatGPT",          df[df["group_label"] == "ChatGPT"]),
    ]
    n_tests = len(subsets)

    th = 'style="padding:7px 14px;text-align:center;background:#f0f3f8;font-weight:700;color:#1a1a1a;border:1px solid #d8dce3;"'
    th_left = 'style="padding:7px 14px;text-align:left;background:#f0f3f8;font-weight:700;color:#1a1a1a;border:1px solid #d8dce3;"'
    td_c = 'style="padding:7px 14px;text-align:center;border:1px solid #d8dce3;"'
    td_l = 'style="padding:7px 14px;text-align:left;border:1px solid #d8dce3;"'

    header = (
        f"<thead><tr>"
        f"<th {th_left}>Subset</th>"
        f"<th {th}>n</th>"
        f"<th {th}>\u03c1 (rho)</th>"
        f"<th {th}>p-value</th>"
        f"<th {th}>p (Bonferroni)</th>"
        f"</tr></thead>"
    )

    rows = []
    for label, sub in subsets:
        valid = sub[["self_confidence_mean", "n_correct"]].dropna()
        n = len(valid)
        if n >= 3:
            rho, pval = spearmanr(valid["self_confidence_mean"], valid["n_correct"])
            rho_str  = f"{rho:.2f}"
            pval_str = "< 0.001" if pval < 0.001 else f"{pval:.3f}"
            p_adj    = min(pval * n_tests, 1.0)
            padj_str = "< 0.001" if p_adj < 0.001 else f"{p_adj:.3f}"
        else:
            rho_str  = "—"
            pval_str = "—"
            padj_str = "—"
        rows.append(
            f"<tr>"
            f"<td {td_l}>{label}</td>"
            f"<td {td_c}>{n}</td>"
            f"<td {td_c}>{rho_str}</td>"
            f"<td {td_c}>{pval_str}</td>"
            f"<td {td_c}>{padj_str}</td>"
            f"</tr>"
        )

    return (
        f'<table style="border-collapse:collapse;font-family:{FONT};'
        f'font-size:13px;width:auto;">'
        f"{header}"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
    )


def build_correlation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Table 3 correlation results as a DataFrame.
    Mirrors the subsets and logic used in build_correlation_table().
    """
    subsets = [
        ("All Participants", df),
        ("No Resource",      df[df["group_label"] == "No Resource"]),
        ("PDF",              df[df["group_label"] == "PDF"]),
        ("ChatGPT",          df[df["group_label"] == "ChatGPT"]),
    ]
    n_tests = len(subsets)
    rows = []
    for label, sub in subsets:
        valid = sub[["self_confidence_mean", "n_correct"]].dropna()
        n = len(valid)
        if n >= 3:
            rho, pval = spearmanr(valid["self_confidence_mean"], valid["n_correct"])
            p_adj = min(pval * n_tests, 1.0)
            rows.append({
                "Subset":          label,
                "n":               n,
                "rho":             round(rho, 2),
                "p_value":         round(pval, 3) if pval >= 0.001 else "<0.001",
                "p_bonferroni":    round(p_adj, 3) if p_adj >= 0.001 else "<0.001",
            })
        else:
            rows.append({
                "Subset":       label,
                "n":            n,
                "rho":          None,
                "p_value":      None,
                "p_bonferroni": None,
            })
    return pd.DataFrame(rows)


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
<p>
We also examined whether participants who rated themselves as more confident before the
assessment tended to answer more questions correctly. This question was explored both across all
participants combined and within each experimental group separately, using Spearman rank
correlation — a method suited to ordinal ratings and count data. The scatter plots and
correlation table at the end of this report display those results.
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
<p>
To examine the association between pre-assessment self-rated confidence and knowledge
performance, we computed Spearman rank-order correlations (&rho;) between the average
self-assessment score and the total number of correct responses (out of 12). The average
self-assessment score was the mean of three 0&ndash;10 Likert ratings completed before the
assessment: self-rated TDI knowledge, avulsion management confidence, and fracture management
confidence. Correlations were computed for the full sample (n&nbsp;=&nbsp;18) and separately
within each experimental group (No Resource n&nbsp;=&nbsp;7, PDF n&nbsp;=&nbsp;5,
ChatGPT n&nbsp;=&nbsp;6); p-values were adjusted for multiple comparisons using the
Bonferroni correction (4 tests; adjusted &alpha;&nbsp;=&nbsp;0.0125). A small uniform jitter (&plusmn;0.15 units) was applied to both
axes to reduce overplotting of identical values. Axis ranges were fixed across all four
scatter plots (x: 0&ndash;10; y: 0&ndash;12) to allow direct visual comparison across
subsets.
</p>

<h3>Figures</h3>
<p>
Boxplots (Figures 1&ndash;4) display the median (horizontal line), interquartile range (box
boundaries), and 1.5&times; IQR (whiskers) for each group. Individual observations are
overlaid as jittered points to show the full data distribution given the small sample size.
Groups are color-coded consistently across all figures: No Resource
(<span style="color:#F8766D;font-weight:bold;">#F8766D</span>), PDF
(<span style="color:#CD9600;font-weight:bold;">#CD9600</span>), ChatGPT
(<span style="color:#00A9FF;font-weight:bold;">#00A9FF</span>). Scatter plots
(Figures 5&ndash;8) display the relationship between average self-rated confidence (x-axis,
0&ndash;10) and total correct responses (y-axis, 0&ndash;12); each point represents one
participant. Axis ranges are fixed across all four plots to allow direct visual comparison.
Spearman &rho; and the associated p-value are annotated in the upper-right corner of each
scatter plot.
All figures were generated using Plotly&nbsp;6.7 and are interactive — hover over any element
for exact values, and use the camera icon in the toolbar to export a PNG at the current
display size.
</p>
<p>
<strong>Code availability.</strong> All analysis scripts are publicly available at
<a href="https://github.com/bernardo-heberle/lauren_master_thesis" target="_blank">
github.com/bernardo-heberle/lauren_master_thesis</a>.
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
    "table3": (
        "<strong>Table 3.</strong> Spearman rank-order correlations between average "
        "pre-assessment self-rated confidence (0&ndash;10 scale; mean of TDI knowledge, "
        "avulsion confidence, and fracture confidence ratings) and the number of correct "
        "responses (out of 12), computed for the full sample and separately within each "
        "experimental group. &rho;&nbsp;=&nbsp;Spearman rho. P-values were adjusted for "
        "multiple comparisons using the Bonferroni correction (4 tests; adjusted "
        "&alpha;&nbsp;=&nbsp;0.0125). "
        "Groups: No Resource n&nbsp;=&nbsp;7, PDF n&nbsp;=&nbsp;5, ChatGPT n&nbsp;=&nbsp;6."
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


def build_scatter_legend(fig_num: int, subset_label: str, rho: float,
                         pval: float, pval_adj: float) -> str:
    """
    Build the HTML legend string for a correlation scatter plot.
    Includes subset description, Spearman rho, raw p-value, and
    Bonferroni-adjusted p-value.
    """
    pval_str = "< 0.001" if pval < 0.001 else f"= {pval:.3f}"
    padj_str = "< 0.001" if pval_adj < 0.001 else f"= {pval_adj:.3f}"
    return (
        f"<strong>Figure {fig_num}. Average self-rated confidence vs. total correct "
        f"({subset_label}).</strong> "
        f"Each point represents one participant. "
        f"x-axis: average pre-assessment self-assessment score (mean of TDI knowledge, "
        f"avulsion confidence, and fracture confidence ratings, each 0&ndash;10). "
        f"y-axis: number of correct responses (out of 12). "
        f"Spearman &rho;&nbsp;=&nbsp;{rho:.2f}, p&nbsp;{pval_str}, "
        f"p<sub>Bonferroni</sub>&nbsp;{padj_str}."
    )


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
    min-width: 0;
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
.btn-legend { color: #7c3aed; border-color: #7c3aed; }
.btn-legend:hover { background: #7c3aed; color: white; }
.btn-title  { color: #b45309; border-color: #b45309; }
.btn-title:hover  { background: #b45309; color: white; }

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

function toggleLegend(divId, btn) {
    const fig = document.getElementById(divId);
    const current = fig._fullLayout.showlegend;
    Plotly.relayout(divId, {showlegend: !current});
    btn.textContent = current ? '\u25a1 Show Legend' : '\u25a0 Hide Legend';
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

const _titleStore = {};

function toggleTitle(divId, btn) {
    const fig = document.getElementById(divId);
    const current = fig._fullLayout.title.text;
    if (current) {
        _titleStore[divId] = current;
        Plotly.relayout(divId, {'title.text': ''});
        btn.textContent = '\u25a1 Show Title';
    } else {
        Plotly.relayout(divId, {'title.text': _titleStore[divId] || ''});
        btn.textContent = '\u25a0 Hide Title';
    }
}

function updateFontSizes(divId) {
    const title   = parseInt(document.getElementById(divId + '-fs-title').value);
    const axlabel = parseInt(document.getElementById(divId + '-fs-axlabel').value);
    const axtick  = parseInt(document.getElementById(divId + '-fs-axtick').value);
    const legend  = parseInt(document.getElementById(divId + '-fs-legend').value);
    document.getElementById(divId + '-fs-title-val').textContent   = title   + 'pt';
    document.getElementById(divId + '-fs-axlabel-val').textContent = axlabel + 'pt';
    document.getElementById(divId + '-fs-axtick-val').textContent  = axtick  + 'pt';
    document.getElementById(divId + '-fs-legend-val').textContent  = legend  + 'pt';
    Plotly.relayout(divId, {
        'title.font.size':          title,
        'xaxis.title.font.size':    axlabel,
        'yaxis.title.font.size':    axlabel,
        'xaxis.tickfont.size':      axtick,
        'yaxis.tickfont.size':      axtick,
        'legend.font.size':         legend,
        'legend.title.font.size':   legend,
    });
}
"""


def download_btn(label: str, b64: str, filename: str, mime: str, css_class: str) -> str:
    return (
        f'<button class="btn {css_class}" '
        f'onclick="dlFile(\'{b64}\',\'{filename}\',\'{mime}\')">'
        f'{label}</button>'
    )


def font_size_controls(div_id: str,
                       def_title: int = 16, def_axlabel: int = 13,
                       def_axtick: int = 12, def_legend: int = 12) -> str:
    """Render a row of font-size sliders (title, axis labels, axis ticks, legend)."""
    return (
        f'<div class="resize-controls">'
        f'<span>&#65; Font sizes:</span>'
        f'<label>Title:&nbsp;<span id="{div_id}-fs-title-val" class="resize-val">{def_title}pt</span></label>'
        f'<input type="range" id="{div_id}-fs-title" min="8" max="28" value="{def_title}"'
        f' oninput="updateFontSizes(\'{div_id}\')">'
        f'<label>Axis labels:&nbsp;<span id="{div_id}-fs-axlabel-val" class="resize-val">{def_axlabel}pt</span></label>'
        f'<input type="range" id="{div_id}-fs-axlabel" min="6" max="24" value="{def_axlabel}"'
        f' oninput="updateFontSizes(\'{div_id}\')">'
        f'<label>Ticks:&nbsp;<span id="{div_id}-fs-axtick-val" class="resize-val">{def_axtick}pt</span></label>'
        f'<input type="range" id="{div_id}-fs-axtick" min="6" max="20" value="{def_axtick}"'
        f' oninput="updateFontSizes(\'{div_id}\')">'
        f'<label>Legend:&nbsp;<span id="{div_id}-fs-legend-val" class="resize-val">{def_legend}pt</span></label>'
        f'<input type="range" id="{div_id}-fs-legend" min="6" max="20" value="{def_legend}"'
        f' oninput="updateFontSizes(\'{div_id}\')">'
        f'</div>'
    )


def figure_section(fig_num: int, title: str, fig_div: str, pdf_b64: str,
                   pdf_filename: str, png_filename: str, legend: str,
                   default_w: int = 800, default_h: int = 460,
                   show_legend_btn: bool = True) -> str:
    """Render a complete figure block with resize controls, download buttons, and legend."""
    div_id  = f"plotly-fig{fig_num}"
    ar      = round(default_h / default_w, 4)
    pdf_btn    = download_btn("⬇ Download PDF", pdf_b64, pdf_filename, "application/pdf", "btn-pdf")
    png_btn    = (
        f'<button class="btn btn-png" '
        f'onclick="dlPng(\'{div_id}\',\'{png_filename.replace(".png","")}\')">&#11015; Download PNG</button>'
    )
    legend_btn = (
        f'<button class="btn btn-legend" id="{div_id}-legend-btn" '
        f'onclick="toggleLegend(\'{div_id}\', this)">&#9632; Hide Legend</button>'
    ) if show_legend_btn else ""
    title_btn = (
        f'<button class="btn btn-title" id="{div_id}-title-btn" '
        f'onclick="toggleTitle(\'{div_id}\', this)">&#9632; Hide Title</button>'
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
    font_controls = font_size_controls(div_id)
    return f"""\
<div class="figure-block" id="fig{fig_num}">
  <h3>Figure {fig_num} \u2014 {title}</h3>
  {resize}
  {font_controls}
  <div class="download-bar" style="margin-bottom:6px;">{png_btn}{pdf_btn}{legend_btn}{title_btn}</div>
  <p style="font-size:11.5px;color:#888;margin-bottom:10px;">
    <strong>PNG</strong> reflects the current figure size and legend visibility.
    &ensp;<strong>PDF</strong> is pre-rendered at a fixed size (900&times;520&nbsp;px) with the
    legend always shown &mdash; resize and legend-toggle have no effect on it.
  </p>
  <div class="legend-text" style="margin-bottom:10px;">{legend}</div>
  <div class="figure-plot">{fig_div}</div>
</div>
"""


def table_section(section_id: str, title: str, table_html: str, legend: str,
                  download_bar_html: str = "") -> str:
    """Render a complete table block with an optional download bar and legend."""
    bar = f'<div class="download-bar" style="margin-top:10px;">{download_bar_html}</div>' \
          if download_bar_html else ""
    return f"""\
<div id="{section_id}" style="margin-bottom:28px;">
  <h3 style="font-size:16px;font-weight:700;margin-bottom:12px;">{title}</h3>
  <div class="table-wrap">{table_html}</div>
  {bar}
  <div class="legend-text">{legend}</div>
</div>
"""


def build_html(figures: list[dict], table1_html: str, table2_html: str,
               t1_csv_b64: str, t2_csv_b64: str, excel_b64: str,
               corr_figures: list[dict] | None = None,
               corr_table_html: str = "",
               t3_csv_b64: str = "", t3_excel_b64: str = "") -> str:
    """Assemble the full HTML report string."""
    n_box = len(figures)
    fig_sections = "\n".join(
        figure_section(
            i + 1, f["title"], f["div"], f["pdf_b64"],
            f["pdf_filename"], f["png_filename"], f["legend"],
        )
        for i, f in enumerate(figures)
    )

    t1_section = table_section(
        "table1", "Table 1 \u2014 Summary Statistics",
        table1_html, TABLE_LEGENDS["table1"],
    )
    xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    t2_btns = (
        download_btn("\u2b07 Table 1 \u2014 CSV",  t1_csv_b64, "table1_summary.csv",       "text/csv",  "btn-csv")
        + download_btn("\u2b07 Table 2 \u2014 CSV",  t2_csv_b64, "table2_per_question.csv",   "text/csv",  "btn-csv")
        + download_btn("\u2b07 Both Tables \u2014 Excel", excel_b64, "tables_summary.xlsx", xlsx_mime, "btn-excel")
    )
    t2_section = table_section(
        "table2", "Table 2 \u2014 Per-question Correctness by Group",
        table2_html, TABLE_LEGENDS["table2"],
        download_bar_html=t2_btns,
    )

    # ── Correlation section ──────────────────────────────────────────────────
    corr_section = ""
    sidebar_corr = ""
    if corr_figures:
        corr_fig_sections = "\n".join(
            figure_section(
                n_box + i + 1, f["title"], f["div"], f["pdf_b64"],
                f["pdf_filename"], f["png_filename"], f["legend"],
                show_legend_btn=False,
            )
            for i, f in enumerate(corr_figures)
        )
        t3_section = table_section(
            "table3", "Table 3 \u2014 Spearman Correlation Summary",
            corr_table_html, TABLE_LEGENDS["table3"],
            download_bar_html=(
                download_btn("\u2b07 Table 3 \u2014 CSV",   t3_csv_b64,   "table3_correlation.csv",   "text/csv",  "btn-csv")
                + download_btn("\u2b07 Table 3 \u2014 Excel", t3_excel_b64, "table3_correlation.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "btn-excel")
            ) if t3_csv_b64 else "",
        )
        corr_section = f"""\
    <!-- Correlation Analysis -->
    <section class="section" id="correlations">
      <h2>Correlation Analysis</h2>
      <p style="font-size:13px;color:#666;margin-bottom:20px;">
        Scatter plots show the relationship between average pre-assessment self-rated
        confidence and total correct responses. Spearman &rho; and p-value are annotated
        in the upper-right corner of each plot. Each point represents one participant.
        <strong>Color key (per-group plots):</strong>&nbsp;
        <span class="chip" style="background:#F8766D;"></span>No Resource&ensp;
        <span class="chip" style="background:#CD9600;"></span>PDF&ensp;
        <span class="chip" style="background:#00A9FF;"></span>ChatGPT
      </p>
      {corr_fig_sections}
      <div style="margin-top:32px;">
        {t3_section}
      </div>
    </section>"""
        sidebar_corr_figs = "\n".join(
            f'    <li class="sub"><a href="#fig{n_box + i + 1}">Fig {n_box + i + 1}: {f["short"]}</a></li>'
            for i, f in enumerate(corr_figures)
        )
        sidebar_corr = f"""\
        <li class="section-head"><a href="#correlations">Correlations</a></li>
        <li class="sub"><a href="#table3">Table 3</a></li>
{sidebar_corr_figs}"""

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
        {sidebar_corr}
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

    {corr_section}

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

    print("Building correlation scatter plots...")
    X_COL    = "self_confidence_mean"
    Y_COL    = "n_correct"
    X_LABEL  = "Average Self-Assessment (0\u201310)"
    Y_LABEL  = "Total Correct (out of 12)"

    scatter_specs = [
        ("All Participants",  df,                                      "#7CAE00",
         "Confidence vs. Correct \u2014 All Participants",
         "fig5_conf_vs_correct_all",       "Confidence vs. Correct (all)"),
        ("No Resource",       df[df["group_label"] == "No Resource"],  "#F8766D",
         "Confidence vs. Correct \u2014 No Resource",
         "fig6_conf_vs_correct_noresource", "Confidence vs. Correct (No Resource)"),
        ("PDF",               df[df["group_label"] == "PDF"],          "#CD9600",
         "Confidence vs. Correct \u2014 PDF",
         "fig7_conf_vs_correct_pdf",        "Confidence vs. Correct (PDF)"),
        ("ChatGPT",           df[df["group_label"] == "ChatGPT"],      "#00A9FF",
         "Confidence vs. Correct \u2014 ChatGPT",
         "fig8_conf_vs_correct_chatgpt",    "Confidence vs. Correct (ChatGPT)"),
    ]

    n_box = len(figures)
    n_corr_tests = len(scatter_specs)
    corr_figures_raw = []
    for subset_label, df_sub, color, title, file_stem, short in scatter_specs:
        valid = df_sub[[X_COL, Y_COL]].dropna()
        rho, pval = spearmanr(valid[X_COL], valid[Y_COL])
        pval_adj = min(pval * n_corr_tests, 1.0)
        fig = build_scatter(df_sub, X_COL, Y_COL, X_LABEL, Y_LABEL, title, color)
        corr_figures_raw.append((fig, title, short, file_stem, subset_label, rho, pval, pval_adj))

    print("Rendering correlation PDFs via kaleido...")
    corr_figures = []
    for i, (fig, title, short, file_stem, subset_label, rho, pval, pval_adj) in \
            enumerate(corr_figures_raw, start=1):
        fig_num = n_box + i
        print(f"  Figure {fig_num}: {title}")
        div_id = f"plotly-fig{fig_num}"
        corr_figures.append({
            "title":        title,
            "short":        short,
            "div":          fig_to_html_div(fig, div_id),
            "pdf_b64":      fig_to_pdf_b64(fig),
            "pdf_filename": f"{file_stem}.pdf",
            "png_filename": f"{file_stem}.png",
            "legend":       build_scatter_legend(fig_num, subset_label, rho, pval, pval_adj),
        })

    print("Building correlation table...")
    corr_table_html = build_correlation_table(df)
    corr_df         = build_correlation_dataframe(df)
    corr_df.to_csv(TABLE3_CSV, index=False)
    corr_df.to_excel(TABLE3_EXCEL, index=False, sheet_name="Table3_Correlation")
    print(f"  Saved {TABLE3_CSV.name} and {TABLE3_EXCEL.name}")

    print("Encoding table assets...")
    table1_html  = strip_style_tag(TABLE1_HTML.read_text(encoding="utf-8"))
    table2_html  = strip_style_tag(TABLE2_HTML.read_text(encoding="utf-8"))
    t1_csv_b64   = b64_file(TABLE1_CSV)
    t2_csv_b64   = b64_file(TABLE2_CSV)
    t3_csv_b64   = b64_file(TABLE3_CSV)
    t3_excel_b64 = b64_file(TABLE3_EXCEL)
    excel_b64    = b64_file(EXCEL_PATH)

    print("Assembling HTML...")
    html = build_html(
        figures, table1_html, table2_html, t1_csv_b64, t2_csv_b64, excel_b64,
        corr_figures=corr_figures,
        corr_table_html=corr_table_html,
        t3_csv_b64=t3_csv_b64,
        t3_excel_b64=t3_excel_b64,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")

    size_mb = OUT_HTML.stat().st_size / 1_000_000
    print(f"\nReport saved: {OUT_HTML.relative_to(ROOT)}")
    print(f"File size: {size_mb:.1f} MB")
    print("Open the HTML file in any browser to view the report.")
