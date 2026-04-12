"""
06.power_analysis.py

Generates a self-contained interactive HTML power analysis report with:
  - Sensitivity analysis: minimum detectable effect at N=18, 80% power
  - Prospective sample size estimation: N required for 80% power at observed effects
  - Interactive Plotly power curve figures (one per test family)
  - Summary table with CSV download

Output: reports/power_analysis_report.html
        tables/power_analysis_summary.csv

Usage:
    .venv\\Scripts\\python.exe scripts/06.power_analysis.py
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.optimize import brentq
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chi2_contingency, f as f_dist
from scipy.stats import ncf, ncx2, norm, spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "survey_clean.csv"
OUT_DIR  = ROOT / "reports"
OUT_HTML = OUT_DIR / "power_analysis_report.html"
CSV_OUT  = ROOT / "tables" / "power_analysis_summary.csv"

# ── Study constants ────────────────────────────────────────────────────────────
GROUP_NS     = [7, 5, 6]     # No Resource, PDF, ChatGPT
N_TOTAL      = 18
K_GROUPS     = 3
N_ITEMS      = 12            # knowledge items (Bonferroni denominator)
ALPHA        = 0.05
ALPHA_BONF   = ALPHA / N_ITEMS   # ≈ 0.00417
POWER_TARGET = 0.80
MAX_N        = 500           # cap for displaying "N > 500" in table

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

COLOR_KW     = "#7CAE00"    # green (all main curves share one colour)
COLOR_CHISQ  = "#7CAE00"    # green – solid  (unadjusted alpha line)
COLOR_CHISQB = "#7CAE00"    # green – dashed (Bonferroni alpha line)
COLOR_SPEAR  = "#7CAE00"    # green – solid  (all-participants curve)

CORRECT_COLS = [
    "c1_injury_type_correct", "c1_treatment_correct", "c1_antibiotics_correct",
    "c2_injury_type_correct", "c2_treatment_correct", "c2_tf_60min_correct",
    "c2_storage_rank_correct", "c2_antibiotics_correct",
    "c3_injury_type_correct", "c3_treatment_correct",
    "c3_imaging_correct",     "c3_antibiotics_correct",
]


# ── CSS (identical to 04.generate_report.py) ──────────────────────────────────
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


# ── JS (identical to 04.generate_report.py) ───────────────────────────────────
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


# ── HTML helper functions (identical pattern to 04.generate_report.py) ─────────

def b64_file(path: Path) -> str:
    """Base64-encode a file's bytes."""
    return base64.b64encode(path.read_bytes()).decode()


def b64_str(s: str) -> str:
    """Base64-encode a UTF-8 string."""
    return base64.b64encode(s.encode()).decode()


def fig_to_html_div(fig: go.Figure, div_id: str) -> str:
    """Serialize a Plotly figure to an HTML div string (no full page, no JS)."""
    config = {**PLOT_CONFIG, "toImageButtonOptions": {
        **PLOT_CONFIG["toImageButtonOptions"], "filename": div_id,
    }}
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config=config, div_id=div_id)


def fig_to_pdf_b64(fig: go.Figure, width: int = 900, height: int = 520) -> str:
    """Render a Plotly figure to PDF bytes and base64-encode."""
    pdf_bytes = pio.to_image(fig, format="pdf", width=width, height=height)
    return base64.b64encode(pdf_bytes).decode()


def download_btn(label: str, b64: str, filename: str, mime: str, css_class: str) -> str:
    """Render an HTML download button that triggers a base64 file download."""
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
                   default_w: int = 800, default_h: int = 460) -> str:
    """Render a complete figure block with resize controls, download buttons, and legend."""
    div_id = f"plotly-fig{fig_num}"
    ar     = round(default_h / default_w, 4)
    pdf_btn = download_btn("⬇ Download PDF", pdf_b64, pdf_filename,
                           "application/pdf", "btn-pdf")
    png_btn = (
        f'<button class="btn btn-png" '
        f'onclick="dlPng(\'{div_id}\',\'{png_filename.replace(".png","")}\')">&#11015; Download PNG</button>'
    )
    legend_btn = (
        f'<button class="btn btn-legend" id="{div_id}-legend-btn" '
        f'onclick="toggleLegend(\'{div_id}\', this)">&#9632; Hide Legend</button>'
    )
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


# ── Power analysis computation functions ──────────────────────────────────────

def _harmonic_mean(ns: list) -> float:
    """Harmonic mean of a list of positive integers."""
    return len(ns) / sum(1.0 / n for n in ns)


def compute_effect_sizes(df: pd.DataFrame) -> dict:
    """
    Compute observed effect sizes for all three test families from survey_clean data.

    Returns a dict with keys:
        cohens_f         – Cohen's f for KW/ANOVA on n_correct
        eta_sq           – eta-squared (for reference)
        cohens_w_median  – median Cohen's w across 12 per-question chi-square tests
        cohens_w_values  – list of per-question w values
        spearman_rho     – Spearman rho (self_confidence_mean vs n_correct, all N)
        spearman_pval    – associated p-value
    """
    # ── A. Cohen's f (KW via ANOVA approximation) ─────────────────────────────
    groups_data = [
        df[df["group_label"] == g]["n_correct"].dropna().values
        for g in HUE_ORDER
    ]
    group_ns    = np.array([len(g) for g in groups_data])
    group_means = np.array([g.mean() for g in groups_data])
    grand_mean  = df["n_correct"].dropna().mean()

    ss_between = float(np.sum(group_ns * (group_means - grand_mean) ** 2))
    ss_within  = float(sum(np.sum((g - m) ** 2)
                           for g, m in zip(groups_data, group_means)))
    ss_total   = ss_between + ss_within

    eta_sq   = ss_between / ss_total if ss_total > 0 else 0.0
    cohens_f = float(np.sqrt(eta_sq / (1.0 - eta_sq))) if 0 < eta_sq < 1 else 0.0

    # ── B. Cohen's w (per-question, 2×3 chi-square tables) ───────────────────
    w_values = []
    for col in CORRECT_COLS:
        ct = pd.crosstab(df[col], df["group_label"])
        ct = ct.reindex(columns=HUE_ORDER, fill_value=0)
        chi2_stat, _, _, _ = chi2_contingency(ct, correction=False)
        w = float(np.sqrt(max(chi2_stat, 0.0) / N_TOTAL))
        w_values.append(w)

    # ── C. Spearman rho (self-confidence vs n_correct) ────────────────────────
    X_COL, Y_COL = "self_confidence_mean", "n_correct"
    valid = df[[X_COL, Y_COL]].dropna()
    rho_all, pval_all = spearmanr(valid[X_COL], valid[Y_COL])

    # Per-group Spearman rhos
    group_rhos: dict[str, tuple[float, float, int]] = {}
    for g in HUE_ORDER:
        sub = df[df["group_label"] == g][[X_COL, Y_COL]].dropna()
        if len(sub) >= 4:
            r, p = spearmanr(sub[X_COL], sub[Y_COL])
        else:
            r, p = float("nan"), float("nan")
        group_rhos[g] = (float(r), float(p), len(sub))

    return {
        "cohens_f":        cohens_f,
        "eta_sq":          eta_sq,
        "cohens_w_median": float(np.median(w_values)),
        "cohens_w_values": w_values,
        "spearman_rho":    float(rho_all),
        "spearman_pval":   float(pval_all),
        "spearman_groups": group_rhos,
    }


# ── KW (one-way ANOVA F-test, non-central F distribution) ────────────────────
# Using scipy's ncf directly avoids statsmodels' internal solve_power limitations.
# Non-centrality: lambda = N * f^2; df_num = k-1; df_denom = N-k.

def _kw_power(f: float, n_total: float, k: int = K_GROUPS,
              alpha: float = ALPHA) -> float:
    """
    Statistical power for Kruskal-Wallis (ANOVA F-test approximation).
    Assumes equal group sizes (n_total / k per group). Uses a non-central F.
    """
    if f <= 0 or n_total < k:
        return alpha
    df_num   = k - 1
    df_denom = n_total - k
    nc       = n_total * f ** 2
    f_crit   = f_dist.ppf(1.0 - alpha, df_num, df_denom)
    return float(1.0 - ncf.cdf(f_crit, df_num, df_denom, nc))


def _kw_required_n(f: float, k: int = K_GROUPS, alpha: float = ALPHA,
                   power: float = POWER_TARGET) -> int | None:
    """
    Total N (equal groups) required for the target power. Returns None if > MAX_N.
    """
    if f <= 0.01:
        return None
    if _kw_power(f, float(MAX_N), k, alpha) < power:
        return None

    def target(n: float) -> float:
        return _kw_power(f, n, k, alpha) - power

    # Lower bound must ensure df_denom = n - k > 0; use k*3 to be safe
    n_req = brentq(target, float(k * 3), float(MAX_N))
    total = int(np.ceil(n_req / k)) * k   # round up to nearest multiple of k
    return total if total <= MAX_N else None


def _kw_sensitivity(alpha: float = ALPHA, power: float = POWER_TARGET) -> float:
    """
    Minimum detectable Cohen's f at N=18 with 80% power.
    Uses actual total N=18 (df_denom=15) to correctly represent the current study.
    """
    def target(f: float) -> float:
        return _kw_power(f, float(N_TOTAL), K_GROUPS, alpha) - power

    if target(0.001) >= 0:
        return 0.001
    return float(brentq(target, 0.001, 5.0))


# ── Chi-square (non-central chi-square, df=2 for 2×3 table) ──────────────────
# Non-centrality: lambda = N * w^2; df = (r-1)(c-1) = 2 for 2×3 table.

def _chisq_power(w: float, n_total: float, alpha: float = ALPHA,
                 df: int = 2) -> float:
    """Power for chi-square test (2×3 table, df=2) using a non-central chi-square."""
    if w <= 0:
        return alpha
    nc       = n_total * w ** 2
    chi_crit = chi2_dist.ppf(1.0 - alpha, df)
    return float(1.0 - ncx2.cdf(chi_crit, df, nc))


def _chisq_required_n(w: float, alpha: float = ALPHA,
                      power: float = POWER_TARGET, df: int = 2) -> int | None:
    """Total N required for given power, chi-square test. Returns None if > MAX_N."""
    if w <= 0.01:
        return None
    if _chisq_power(w, float(MAX_N), alpha, df) < power:
        return None

    def target(n: float) -> float:
        return _chisq_power(w, n, alpha, df) - power

    n_req = brentq(target, 9.0, float(MAX_N))
    return int(np.ceil(n_req)) if int(np.ceil(n_req)) <= MAX_N else None


def _chisq_sensitivity(alpha: float = ALPHA, power: float = POWER_TARGET,
                       df: int = 2) -> float:
    """Minimum detectable Cohen's w at N=18 with 80% power."""
    def target(w: float) -> float:
        return _chisq_power(w, float(N_TOTAL), alpha, df) - power

    if target(0.001) >= 0:
        return 0.001
    return float(brentq(target, 0.001, 5.0))


# ── Spearman correlation (Fisher z-transform) ─────────────────────────────────

def _spearman_power(rho: float, n: int, alpha: float = ALPHA) -> float:
    """
    Power for two-sided test of H0: rho=0, using the Fisher z-transform
    approximation (standard for Pearson/Spearman correlation tests).
    """
    if abs(rho) < 1e-6 or n <= 3:
        return alpha
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    z_rho  = np.arctanh(abs(rho))
    lam    = np.sqrt(n - 3) * z_rho
    return float(norm.cdf(lam - z_crit) + norm.cdf(-lam - z_crit))


def _spearman_required_n(rho: float, alpha: float = ALPHA,
                         power: float = POWER_TARGET) -> int | None:
    """N required for the given power (two-sided). Returns None if > MAX_N."""
    if abs(rho) < 0.01:
        return None
    if _spearman_power(rho, MAX_N, alpha) < power:
        return None
    def f(n: float) -> float:
        return _spearman_power(rho, n, alpha) - power
    n_req = brentq(f, 4.0, float(MAX_N))
    return int(np.ceil(n_req))


def _spearman_sensitivity(alpha: float = ALPHA,
                          power: float = POWER_TARGET) -> float | None:
    """Minimum |rho| detectable at N=18 with 80% power."""
    def f(rho: float) -> float:
        return _spearman_power(rho, N_TOTAL, alpha) - power
    if _spearman_power(0.999, N_TOTAL, alpha) < power:
        return None
    return float(brentq(f, 0.001, 0.999))


# ── Power curve figure builders ────────────────────────────────────────────────

def _fmt_n(n: int | None) -> str:
    """Format a required-N result for display in the summary table."""
    if n is None:
        return f"&gt;&nbsp;{MAX_N}"
    return str(n)


def _build_power_curve_layout(title: str, x_max: int) -> dict:
    """Return a shared Plotly layout dict for power curve figures."""
    return dict(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=16, family=FONT), x=0.5),
        xaxis=dict(
            title=dict(text="Total Sample Size (N)",
                       font=dict(size=13, family=FONT)),
            range=[0, x_max],
            showgrid=False, zeroline=False,
            showline=True, linecolor="#d0d0d0",
            tickfont=dict(size=12, family=FONT),
        ),
        yaxis=dict(
            title=dict(text="Statistical Power (1 \u2212 \u03b2)",
                       font=dict(size=13, family=FONT)),
            range=[0, 1.05],
            showgrid=False, zeroline=False,
            showline=True, linecolor="#d0d0d0",
            tickformat=".0%",
            tickfont=dict(size=12, family=FONT),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT, size=13),
        showlegend=True,
        legend=dict(
            font=dict(size=12, family=FONT),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#d0d0d0",
            borderwidth=1,
        ),
        margin=dict(l=70, r=30, t=65, b=60),
        height=460,
    )


def _reference_shapes(x_max: int) -> list:
    """Horizontal power=0.80 and vertical N=18 dashed reference lines."""
    return [
        dict(type="line", x0=0, x1=x_max, y0=0.80, y1=0.80,
             line=dict(color="#888", width=1.5, dash="dot")),
        dict(type="line", x0=N_TOTAL, x1=N_TOTAL, y0=0, y1=1.05,
             line=dict(color="#aaa", width=1.5, dash="dash")),
    ]


def _annotation_current_n() -> dict:
    """Arrow annotation labeling the current N=18 vertical line.
    Arrow tip sits on the vertical line at (N_TOTAL, 0.05);
    the text box is placed at x=60 in data coordinates.
    """
    return dict(
        x=N_TOTAL, y=0.05,
        xref="x", yref="y",
        text=f"<b>Current study<br>N = {N_TOTAL}</b>",
        showarrow=True,
        arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor="#888",
        ax=60, ay=0.05,
        axref="x", ayref="y",
        font=dict(size=11, family=FONT, color="#666"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d0d0d0",
        borderwidth=1,
    )


def _annotation_required_n(n_req: int, power_at_req: float, color: str,
                            ax: int = -55, ay: int = -40) -> dict:
    """Arrow annotation marking the required N on the curve."""
    return dict(
        x=n_req, y=power_at_req,
        xref="x", yref="y",
        text=f"<b>N = {n_req}</b><br>for 80% power",
        showarrow=True,
        arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor=color,
        ax=ax, ay=ay,
        font=dict(size=11, family=FONT, color=color),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=color,
        borderwidth=1,
    )


def build_fig_kw(cohens_f: float, x_max: int | None = None) -> go.Figure:
    """
    Power curve figure for Kruskal-Wallis (ANOVA approximation).
    x-axis: total N (equal groups); y-axis: power at observed Cohen's f.
    Pass x_max to share a common axis range across figures.
    """
    n_req = _kw_required_n(cohens_f)
    if x_max is None:
        x_max = min(MAX_N, max(150, (n_req or 150) + 80))
    n_range = np.linspace(K_GROUPS, x_max, 300)
    powers  = [_kw_power(cohens_f, n) for n in n_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_range.tolist(), y=powers,
        mode="lines",
        line=dict(color=COLOR_KW, width=2.5),
        name=f"Cohen\u2019s f = {cohens_f:.3f}",
    ))

    annotations = [_annotation_current_n()]
    if n_req is not None:
        annotations.append(
            _annotation_required_n(n_req, 0.80, COLOR_KW, ax=-60, ay=-45)
        )

    layout = _build_power_curve_layout(
        "Power Curve \u2014 Kruskal-Wallis (ANOVA Approximation)", x_max
    )
    layout["shapes"]      = _reference_shapes(x_max)
    layout["annotations"] = annotations
    fig.update_layout(**layout)
    return fig


def build_fig_chisq(cohens_w: float, x_max: int | None = None) -> go.Figure:
    """
    Power curve figure for chi-square (2×3 table, df=2).
    Both curves are green: solid for unadjusted α=0.05,
    dashed for Bonferroni-adjusted α=0.0042.
    Pass x_max to share a common axis range across figures.
    """
    n_req_unadj = _chisq_required_n(cohens_w, alpha=ALPHA)
    n_req_bonf  = _chisq_required_n(cohens_w, alpha=ALPHA_BONF)
    if x_max is None:
        x_max = min(MAX_N, max(200, (n_req_bonf or n_req_unadj or 200) + 80))
    n_range = np.linspace(3, x_max, 300)

    powers_unadj = [_chisq_power(cohens_w, n, alpha=ALPHA)      for n in n_range]
    powers_bonf  = [_chisq_power(cohens_w, n, alpha=ALPHA_BONF)  for n in n_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_range.tolist(), y=powers_unadj,
        mode="lines",
        line=dict(color=COLOR_CHISQ, width=2.5),
        name=f"\u03b1 = 0.05 (unadjusted)",
    ))
    fig.add_trace(go.Scatter(
        x=n_range.tolist(), y=powers_bonf,
        mode="lines",
        line=dict(color=COLOR_CHISQB, width=2.5, dash="dash"),
        name=f"\u03b1 = {ALPHA_BONF:.4f} (Bonferroni)",
    ))

    annotations = [_annotation_current_n()]
    if n_req_unadj is not None:
        annotations.append(
            _annotation_required_n(n_req_unadj, 0.80, COLOR_CHISQ, ax=-55, ay=-45)
        )
    if n_req_bonf is not None:
        annotations.append(
            _annotation_required_n(n_req_bonf, 0.80, COLOR_CHISQB, ax=55, ay=-45)
        )

    layout = _build_power_curve_layout(
        "Power Curve \u2014 Chi-Square (Per-Question, df = 2)", x_max
    )
    layout["shapes"]      = _reference_shapes(x_max)
    layout["annotations"] = annotations
    fig.update_layout(**layout)
    return fig


def build_fig_spearman(
    rho_all: float,
    group_rhos: dict[str, tuple[float, float, int]],
    x_max: int | None = None,
) -> go.Figure:
    """
    Power curve figure for Spearman correlation (Fisher z-transform approximation).
    Solid green curve for the full sample; dotted group-colour curves per group.
    Pass x_max to share a common axis range across figures.
    """
    req_ns = {
        "all": _spearman_required_n(rho_all),
        "all_bonf": _spearman_required_n(rho_all, alpha=ALPHA_BONF),
        **{g: _spearman_required_n(v[0]) for g, v in group_rhos.items()},
    }
    finite_reqs = [n for n in req_ns.values() if n is not None]
    if x_max is None:
        x_max = min(MAX_N, max(200, (max(finite_reqs) if finite_reqs else 200) + 60))
    n_range = np.linspace(4, x_max, 400)

    fig = go.Figure()

    # ── Overall curve (all 18 participants) ──────────────────────────────────
    powers_all = [_spearman_power(rho_all, float(n)) for n in n_range]
    fig.add_trace(go.Scatter(
        x=n_range.tolist(), y=powers_all,
        mode="lines",
        line=dict(color=COLOR_SPEAR, width=2.5),
        name=f"All participants (\u03c1\u202f=\u202f{rho_all:.2f}, N\u202f=\u202f18)",
    ))

    # ── Overall curve Bonferroni ─────────────────────────────────────────────
    powers_all_bonf = [_spearman_power(rho_all, float(n), alpha=ALPHA_BONF)
                       for n in n_range]
    fig.add_trace(go.Scatter(
        x=n_range.tolist(), y=powers_all_bonf,
        mode="lines",
        line=dict(color=COLOR_SPEAR, width=2.5, dash="dot"),
        name=f"\u03b1 = {ALPHA_BONF:.4f} (Bonferroni)",
    ))

    # ── Per-group curves ──────────────────────────────────────────────────────
    # Annotation vertical offsets staggered to avoid overlap
    annot_offsets = {
        "No Resource": dict(ax=50,  ay=-40),   # large rho, small N → push right
        "PDF":         dict(ax=-60, ay=30),    # moderate rho, above 80% line
        "ChatGPT":     dict(ax=-60, ay=-50),   # negligible rho, N > MAX_N
    }
    for g, (rho_g, pval_g, n_g) in group_rhos.items():
        if np.isnan(rho_g):
            continue
        color = HUE_PALETTE[g]
        powers_g = [_spearman_power(rho_g, float(n)) for n in n_range]
        fig.add_trace(go.Scatter(
            x=n_range.tolist(), y=powers_g,
            mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            name=f"{g} (\u03c1\u202f=\u202f{rho_g:.2f}, n\u202f=\u202f{n_g})",
        ))

    # ── Reference lines ───────────────────────────────────────────────────────
    annotations = [_annotation_current_n()]

    # Annotate required N for overall
    if req_ns["all"] is not None:
        annotations.append(
            _annotation_required_n(req_ns["all"], 0.80, COLOR_SPEAR,
                                   ax=-60, ay=-45)
        )
    # Annotate required N for overall Bonferroni
    if req_ns["all_bonf"] is not None:
        annotations.append(
            _annotation_required_n(req_ns["all_bonf"], 0.80, COLOR_SPEAR,
                                   ax=55, ay=-45)
        )
    # Annotate per-group required Ns where achievable
    for g, n_req_g in {k: v for k, v in req_ns.items()
                        if k not in ("all", "all_bonf")}.items():
        if n_req_g is not None:
            off = annot_offsets.get(g, dict(ax=-55, ay=-40))
            annotations.append(
                _annotation_required_n(n_req_g, 0.80, HUE_PALETTE[g], **off)
            )

    layout = _build_power_curve_layout(
        "Power Curve \u2014 Spearman Correlation (Fisher z-Transform)", x_max
    )
    # Override x-axis label: per-group N is subgroup size, not study total
    layout["xaxis"]["title"]["text"] = "Sample Size (N)"
    layout["shapes"]      = _reference_shapes(x_max)
    layout["annotations"] = annotations
    fig.update_layout(**layout)
    return fig


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary_table(effects: dict) -> tuple[str, str]:
    """
    Build the HTML and CSV for the power analysis summary table.

    Returns (html_string, csv_string).
    """
    import re
    from scipy.optimize import brentq as _brentq

    f   = effects["cohens_f"]
    w   = effects["cohens_w_median"]
    rho = effects["spearman_rho"]
    group_rhos = effects["spearman_groups"]

    sens_f   = _kw_sensitivity()
    req_n_kw = _kw_required_n(f)

    sens_w_unadj  = _chisq_sensitivity(alpha=ALPHA)
    sens_w_bonf   = _chisq_sensitivity(alpha=ALPHA_BONF)
    req_n_chisq_u = _chisq_required_n(w, alpha=ALPHA)
    req_n_chisq_b = _chisq_required_n(w, alpha=ALPHA_BONF)

    sens_rho       = _spearman_sensitivity()
    sens_rho_bonf  = _spearman_sensitivity(alpha=ALPHA_BONF)
    req_n_rho      = _spearman_required_n(rho)
    req_n_rho_bonf = _spearman_required_n(rho, alpha=ALPHA_BONF)

    def _n_per(total: int | None, k: int = K_GROUPS) -> str:
        if total is None:
            return f"&gt;&nbsp;{MAX_N // k}"
        return str(int(np.ceil(total / k)))

    def _sens_at_n(n_g: int) -> str:
        """Min detectable |rho| at a specific per-group n."""
        def _f(r):
            return _spearman_power(r, n_g, ALPHA) - POWER_TARGET
        if _spearman_power(0.999, n_g, ALPHA) < POWER_TARGET:
            return "not achievable"
        mde = _brentq(_f, 0.001, 0.999)
        return f"|\u03c1| \u2265 {mde:.3f}"

    rows_data = [
        {
            "Analysis": "Kruskal-Wallis<br>(total correct)",
            "Test Approx.": "One-way ANOVA F-test<br>(k = 3 groups)",
            "Obs. Effect": f"f = {f:.3f}",
            "Effect Label": _cohen_f_label(f),
            "Min Detectable (at study n)": f"f \u2265 {sens_f:.3f}<br><small>(N = 18)</small>",
            "Req. Total N": _fmt_n(req_n_kw),
            "Req. n/group": _n_per(req_n_kw),
        },
        {
            "Analysis": "Chi-square per question<br>(\u03b1 = 0.05, unadjusted)",
            "Test Approx.": "Chi-square<br>(df = 2, 2\u00d73 table)",
            "Obs. Effect": f"w = {w:.3f}<br><small>(median of 12)</small>",
            "Effect Label": _cohen_w_label(w),
            "Min Detectable (at study n)": f"w \u2265 {sens_w_unadj:.3f}<br><small>(N = 18)</small>",
            "Req. Total N": _fmt_n(req_n_chisq_u),
            "Req. n/group": _n_per(req_n_chisq_u),
        },
        {
            "Analysis": "Chi-square per question<br>(\u03b1 = 0.0042, Bonferroni)",
            "Test Approx.": "Chi-square<br>(df = 2, 2\u00d73 table)",
            "Obs. Effect": f"w = {w:.3f}<br><small>(median of 12)</small>",
            "Effect Label": _cohen_w_label(w),
            "Min Detectable (at study n)": f"w \u2265 {sens_w_bonf:.3f}<br><small>(N = 18)</small>",
            "Req. Total N": _fmt_n(req_n_chisq_b),
            "Req. n/group": _n_per(req_n_chisq_b),
        },
        {
            "Analysis": "Spearman correlation<br>(self-confidence vs correct,<br><em>all participants</em>)",
            "Test Approx.": "Fisher z-transform<br>(two-sided)",
            "Obs. Effect": f"\u03c1 = {rho:.3f}",
            "Effect Label": _cohen_rho_label(rho),
            "Min Detectable (at study n)": f"|\u03c1| \u2265 {sens_rho:.3f}<br><small>(N = 18)</small>"
                                           if sens_rho else "not achievable",
            "Req. Total N": _fmt_n(req_n_rho),
            "Req. n/group": "N/A<br><small>(1 sample)</small>",
        },
        {
            "Analysis": "Spearman correlation<br>(self-confidence vs correct,<br><em>all participants</em>,<br>\u03b1 = 0.0042 Bonferroni)",
            "Test Approx.": "Fisher z-transform<br>(two-sided)",
            "Obs. Effect": f"\u03c1 = {rho:.3f}",
            "Effect Label": _cohen_rho_label(rho),
            "Min Detectable (at study n)": f"|\u03c1| \u2265 {sens_rho_bonf:.3f}<br><small>(N = 18)</small>"
                                           if sens_rho_bonf else "not achievable",
            "Req. Total N": _fmt_n(req_n_rho_bonf),
            "Req. n/group": "N/A<br><small>(1 sample)</small>",
        },
    ]

    # ── Per-group Spearman rows ───────────────────────────────────────────────
    for g, (rho_g, pval_g, n_g) in group_rhos.items():
        if np.isnan(rho_g):
            continue
        p_str = "p &lt; 0.001" if pval_g < 0.001 else f"p = {pval_g:.3f}"
        chip  = (f'<span style="display:inline-block;width:9px;height:9px;'
                 f'border-radius:2px;background:{HUE_PALETTE[g]};'
                 f'vertical-align:middle;margin-right:4px;"></span>')
        req_n_g = _spearman_required_n(rho_g)
        rows_data.append({
            "Analysis": (f"{chip}Spearman ({g})<br>"
                         f"<small>(self-confidence vs correct)</small>"),
            "Test Approx.": "Fisher z-transform<br>(two-sided)",
            "Obs. Effect": f"\u03c1 = {rho_g:.3f}<br><small>({p_str})</small>",
            "Effect Label": _cohen_rho_label(rho_g),
            "Min Detectable (at study n)": f"{_sens_at_n(n_g)}<br><small>(n\u202f=\u202f{n_g} in group)</small>",
            "Req. Total N": (f"{_fmt_n(req_n_g)}<br>"
                             f"<small>(n needed in {g} group)</small>"),
            "Req. n/group": "&mdash;",
        })

    cols = ["Analysis", "Test Approx.", "Obs. Effect", "Effect Label",
            "Min Detectable (at study n)", "Req. Total N", "Req. n/group"]

    header = "".join(f"<th>{c}</th>" for c in cols)
    body_rows = []
    for row in rows_data:
        cells = "".join(f"<td>{row[c]}</td>" for c in cols)
        body_rows.append(f"<tr>{cells}</tr>")

    html = (
        f'<table style="border-collapse:collapse;font-family:{FONT};'
        f'font-size:13px;width:100%;">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table>"
    )

    def _strip(s: str) -> str:
        return re.sub(r"<[^>]+>", "", s).strip()

    csv_rows = [",".join(f'"{c}"' for c in cols)]
    for row in rows_data:
        csv_rows.append(",".join(f'"{_strip(str(row[c]))}"' for c in cols))
    csv_str = "\n".join(csv_rows)

    return html, csv_str


def _cohen_f_label(f: float) -> str:
    if f < 0.10:
        return "negligible"
    if f < 0.25:
        return "small"
    if f < 0.40:
        return "medium"
    return "large"


def _cohen_w_label(w: float) -> str:
    if w < 0.10:
        return "negligible"
    if w < 0.30:
        return "small"
    if w < 0.50:
        return "medium"
    return "large"


def _cohen_rho_label(rho: float) -> str:
    r = abs(rho)
    if r < 0.10:
        return "negligible"
    if r < 0.30:
        return "small"
    if r < 0.50:
        return "medium"
    return "large"


# ── Narrative text ────────────────────────────────────────────────────────────

LAY_SUMMARY = """\
<p>
Statistical power is the probability that a study will correctly detect a real
effect when one truly exists. When a study enrolls too few participants, it may
fail to find statistically significant differences not because those differences
don't exist, but simply because the sample is too small to detect them reliably.
This situation &mdash; a type II error, or "false negative" &mdash; is a well-recognized
limitation of small pilot and feasibility studies.
</p>
<p>
To quantify <em>how big</em> an effect is, researchers use standardised
<strong>effect size measures</strong> that do not depend on sample size:
</p>
<ul style="margin: 6px 0 12px 24px;">
  <li style="margin-bottom:5px;">
    <strong>Cohen&rsquo;s <em>f</em></strong> &mdash; used for the
    Kruskal-Wallis group comparison. It captures how spread out the three
    group means are relative to the overall variability in scores. A value
    of 0 means no difference between groups; larger values indicate bigger
    group differences. Conventional benchmarks: small&nbsp;&asymp;&nbsp;0.10,
    medium&nbsp;&asymp;&nbsp;0.25, large&nbsp;&asymp;&nbsp;0.40.
  </li>
  <li style="margin-bottom:5px;">
    <strong>Cohen&rsquo;s <em>w</em></strong> &mdash; used for the
    per-question chi-square tests. It measures how far the observed pattern
    of correct/incorrect answers across groups deviates from what would be
    expected if group membership had no effect at all. Conventional benchmarks:
    small&nbsp;&asymp;&nbsp;0.10, medium&nbsp;&asymp;&nbsp;0.30,
    large&nbsp;&asymp;&nbsp;0.50.
  </li>
  <li>
    <strong>Spearman&rsquo;s <em>&rho;</em> (rho)</strong> &mdash; used for
    the correlation between self-rated confidence and number of correct
    answers. It ranges from &minus;1 (perfect negative association) through 0
    (no association) to +1 (perfect positive association). Conventional
    benchmarks: small&nbsp;&asymp;&nbsp;0.10, medium&nbsp;&asymp;&nbsp;0.30,
    large&nbsp;&asymp;&nbsp;0.50.
  </li>
</ul>
<p>
This report asks two complementary questions about the dental trauma survey study
(N&nbsp;=&nbsp;18 participants across three groups):
</p>
<ol style="margin: 10px 0 12px 24px;">
  <li style="margin-bottom:6px;">
    <strong>Sensitivity analysis:</strong> Given the 18 participants enrolled,
    how large would a true difference need to be before this study could
    reliably detect it? This tells us the study's "minimum detection threshold."
  </li>
  <li>
    <strong>Prospective sample size estimation:</strong> Using the effect sizes
    actually observed in this pilot data as estimates, how many participants
    would a future, adequately powered study need to enroll to have an 80%
    chance of detecting those effects? This translates directly into concrete
    planning guidance for follow-up research.
  </li>
</ol>
<p>
A common but methodologically unsound alternative &mdash; computing
<em>post-hoc observed power</em> from the same data used to test the hypothesis
&mdash; was deliberately avoided. As Hoenig and Heisey (2001) demonstrated,
observed power is mathematically equivalent to a transformation of the p-value
and conveys no independent information: a non-significant result will always
produce low observed power, adding nothing to the interpretation.
</p>
<p>
The figures below show power curves for each type of statistical test used in
this study. Each curve shows how statistical power increases as sample size
grows, for an effect of the size actually observed. The dashed horizontal line
marks the conventional 80% power threshold; the dashed vertical line marks the
current study's N&nbsp;=&nbsp;18. The gap between the vertical line and the
annotated required N makes visually explicit the sample size shortfall.
</p>
"""

FORMAL_METHODS = """\
<h3>Effect Size Measures</h3>
<p>
Three standardised effect size indices are reported, one for each statistical
test used in the main study. Standardised effect sizes express the magnitude of
an observed difference or association on a scale that is independent of sample
size, allowing meaningful comparison across studies.
</p>
<ul style="margin: 6px 0 12px 24px;">
  <li style="margin-bottom:6px;">
    <strong>Cohen&rsquo;s <em>f</em></strong> (Kruskal-Wallis group comparison):
    reflects how much the average knowledge scores differ across the three
    groups relative to the variability within groups. A value of 0 indicates
    no group differences; larger values indicate greater separation between
    groups. Conventional benchmarks (Cohen, 1988):
    small&nbsp;=&nbsp;0.10, medium&nbsp;=&nbsp;0.25, large&nbsp;=&nbsp;0.40.
  </li>
  <li style="margin-bottom:6px;">
    <strong>Cohen&rsquo;s <em>w</em></strong> (chi-square per-question test):
    reflects how much the pattern of correct and incorrect answers across
    groups deviates from what would be expected by chance alone. Conventional
    benchmarks: small&nbsp;=&nbsp;0.10, medium&nbsp;=&nbsp;0.30,
    large&nbsp;=&nbsp;0.50.
  </li>
  <li>
    <strong>Spearman&rsquo;s <em>&rho;</em></strong> (rank-order correlation):
    the strength and direction of the association between self-rated confidence
    and number of correct answers, ranging from &minus;1 (perfect negative
    association) to +1 (perfect positive association). Conventional benchmarks:
    small&nbsp;=&nbsp;0.10, medium&nbsp;=&nbsp;0.30, large&nbsp;=&nbsp;0.50.
  </li>
</ul>

<h3>Power Analysis Approach</h3>
<p>
Post-hoc observed power was not reported. Computing power from the same data
used to obtain the p-value is circular: it simply re-expresses the p-value in a
different form and adds no new information about the study
(Hoenig &amp; Heisey, 2001, <em>The American Statistician</em>, 55:19&ndash;24).
Instead, two prospective analyses were carried out for each statistical test:
</p>
<ol style="margin: 6px 0 12px 24px;">
  <li style="margin-bottom:6px;">
    <strong>Sensitivity analysis:</strong> The smallest effect size that could
    have been reliably detected (80% power) given the 18 participants enrolled
    and a significance threshold of &alpha;&nbsp;=&nbsp;0.05. For the
    Bonferroni-corrected per-question comparisons the adjusted threshold
    &alpha;&nbsp;=&nbsp;0.0042 (0.05&nbsp;&divide;&nbsp;12 tests) was applied.
  </li>
  <li>
    <strong>Prospective sample size estimation:</strong> The number of
    participants a future, adequately powered study would need to detect the
    same effect sizes observed here, at 80% power. These figures are
    <em>planning estimates</em> derived from a small pilot; they carry
    substantial uncertainty and should not be interpreted as precise targets.
  </li>
</ol>

<h3>Statistical Methods</h3>
<p>
Power for the <strong>Kruskal-Wallis H-test</strong> was estimated using the
one-way ANOVA approximation, which is standard practice for this non-parametric
test. The effect size (Cohen&rsquo;s f) was derived from the observed group
means on total correct responses.
</p>
<p>
Power for the <strong>chi-square independence tests</strong> (one per knowledge
question, 2&times;3 contingency table) was computed using the non-central
chi-square distribution. The median Cohen&rsquo;s w across the 12 questions
was used as the representative effect size. Two significance thresholds are
shown: the unadjusted &alpha;&nbsp;=&nbsp;0.05 (solid curve) and the
Bonferroni-adjusted &alpha;&nbsp;=&nbsp;0.0042 (dashed curve).
</p>
<p>
Power for the <strong>Spearman rank-order correlation</strong> was estimated
using the Fisher z-transformation, a standard approximation for correlation
tests. Two levels of analysis are reported: the full-sample correlation
(all 18 participants, solid curve in Figure&nbsp;3) and per-group correlations
for No&nbsp;Resource (n&nbsp;=&nbsp;7), PDF (n&nbsp;=&nbsp;5), and
ChatGPT (n&nbsp;=&nbsp;6), shown as dotted curves. Per-group estimates are
based on very small subsamples and should be interpreted with particular caution.
</p>
<p>
All analyses were conducted in Python&nbsp;3.12.
</p>
<p>
<strong>Code availability.</strong> All analysis scripts are publicly available at
<a href="https://github.com/bernardo-heberle/lauren_master_thesis" target="_blank">
github.com/bernardo-heberle/lauren_master_thesis</a>.
</p>
""".format(max_n=MAX_N)


# ── Figure legends ─────────────────────────────────────────────────────────────

def build_fig_legends(effects: dict) -> list[str]:
    """Return a list of three legend strings for the three power curve figures."""
    f           = effects["cohens_f"]
    w           = effects["cohens_w_median"]
    rho         = effects["spearman_rho"]
    group_rhos  = effects["spearman_groups"]

    req_n_kw    = _kw_required_n(f)
    req_n_chi_u = _chisq_required_n(w, alpha=ALPHA)
    req_n_chi_b = _chisq_required_n(w, alpha=ALPHA_BONF)
    req_n_rho     = _spearman_required_n(rho)
    req_n_rho_b   = _spearman_required_n(rho, alpha=ALPHA_BONF)

    def _nstr(n: int | None) -> str:
        return f"N&nbsp;=&nbsp;{n}" if n else f"N&nbsp;&gt;&nbsp;{MAX_N}"

    leg1 = (
        "<strong>Figure 1. Power curve for Kruskal-Wallis H-test "
        "(one-way ANOVA approximation).</strong> "
        f"Observed Cohen\u2019s f&nbsp;=&nbsp;{f:.3f} "
        f"({_cohen_f_label(f)} effect), computed from the observed group means "
        "on total correct responses (out of 12). "
        "The dashed horizontal line marks 80% power; the dashed vertical line "
        f"marks the current study\u2019s N&nbsp;=&nbsp;{N_TOTAL}. "
        f"Achieving 80% power at this effect size requires {_nstr(req_n_kw)} "
        f"(equal groups of {int(np.ceil(req_n_kw / K_GROUPS)) if req_n_kw else '&gt;&thinsp;' + str(MAX_N // K_GROUPS)} per group). "
        "Assumes equal group sizes in the future study."
    )

    leg2 = (
        "<strong>Figure 2. Power curve for chi-square independence test "
        "(per-question, 2\u00d73 table, df&nbsp;=&nbsp;2).</strong> "
        f"Observed median Cohen\u2019s w&nbsp;=&nbsp;{w:.3f} "
        f"({_cohen_w_label(w)} effect) across 12 knowledge questions. "
        f"Solid line: unadjusted \u03b1&nbsp;=&nbsp;0.05 (requires {_nstr(req_n_chi_u)}). "
        f"Dashed line: Bonferroni-adjusted \u03b1&nbsp;=&nbsp;{ALPHA_BONF:.4f} "
        f"(requires {_nstr(req_n_chi_b)}). "
        "The dashed horizontal line marks 80% power; the dashed vertical line "
        f"marks the current study\u2019s N&nbsp;=&nbsp;{N_TOTAL}."
    )

    # Build per-group summary for leg3
    grp_parts = []
    for g, (rho_g, pval_g, n_g) in group_rhos.items():
        if np.isnan(rho_g):
            continue
        req_g = _spearman_required_n(rho_g)
        p_str = "p&nbsp;&lt;&nbsp;0.001" if pval_g < 0.001 else f"p&nbsp;=&nbsp;{pval_g:.3f}"
        chip  = f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;' \
                f'background:{HUE_PALETTE[g]};vertical-align:middle;margin-right:3px;"></span>'
        grp_parts.append(
            f"{chip}<strong>{g}</strong>: \u03c1&nbsp;=&nbsp;{rho_g:.2f}, "
            f"{p_str}, n&nbsp;=&nbsp;{n_g} &rarr; requires {_nstr(req_g)}"
        )

    leg3 = (
        "<strong>Figure 3. Power curve for Spearman rank-order correlation "
        "(Fisher z-transform approximation).</strong> "
        "Each curve shows how power grows with sample size at the observed "
        "\u03c1 (used as a pilot estimate). "
        f"<em>Solid green line</em> (all participants, \u03b1&nbsp;=&nbsp;0.05): "
        f"\u03c1&nbsp;=&nbsp;{rho:.2f} ({_cohen_rho_label(rho)} effect, "
        f"N&nbsp;=&nbsp;{N_TOTAL}); requires {_nstr(req_n_rho)}. "
        f"<em>Dotted green line</em> (all participants, Bonferroni "
        f"\u03b1&nbsp;=&nbsp;{ALPHA_BONF:.4f}): requires {_nstr(req_n_rho_b)}. "
        "<em>Dotted coloured lines</em> (per group): "
        + "; &ensp;".join(grp_parts) + ". "
        "The dashed horizontal line marks 80% power; the dashed vertical line "
        f"marks N&nbsp;=&nbsp;{N_TOTAL} (current total; per-group sizes are "
        "7, 5, and 6). "
        "Note: the No&nbsp;Resource group\u2019s very large \u03c1 is based on "
        "only 7 observations and is subject to substantial sampling variability."
    )

    return [leg1, leg2, leg3]


# ── HTML assembly ──────────────────────────────────────────────────────────────

def build_html(figures: list[dict], table_html: str, table_legend: str,
               csv_b64: str) -> str:
    """Assemble the full standalone HTML power analysis report."""
    fig_sections = "\n".join(
        figure_section(
            i + 1, f["title"], f["div"], f["pdf_b64"],
            f["pdf_filename"], f["png_filename"], f["legend"],
        )
        for i, f in enumerate(figures)
    )

    csv_btn = download_btn(
        "\u2b07 Table 1 \u2014 CSV", csv_b64,
        "power_analysis_summary.csv", "text/csv", "btn-csv"
    )
    tbl_section = table_section(
        "table1",
        "Table 1 \u2014 Power Analysis Summary",
        table_html,
        table_legend,
        download_bar_html=csv_btn,
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
  <title>Dental Trauma Study \u2014 Power Analysis</title>
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
{sidebar_figs}
      </ul>
    </nav>
  </aside>

  <!-- ── Main content ── -->
  <main class="content">
    <div class="report-title">Dental Trauma Study \u2014 Power Analysis</div>
    <div class="report-subtitle">
      Sensitivity analysis &amp; prospective sample size estimation
      &nbsp;|&nbsp; N&nbsp;=&nbsp;{N_TOTAL} participants (pilot study)
      &nbsp;|&nbsp; Target power: {int(POWER_TARGET * 100)}%
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

      <div style="margin-bottom:32px;">
        {tbl_section}
      </div>

      <div style="margin-top:8px;">
        <p style="font-size:13px;color:#666;margin-bottom:20px;">
          Each figure below shows how statistical power increases with total sample
          size for the observed effect size (used as a pilot estimate). The
          <strong>dashed horizontal line</strong> marks the 80% power threshold;
          the <strong>dashed vertical line</strong> marks the current
          N&nbsp;=&nbsp;{N_TOTAL}. Annotated N values indicate the minimum total
          sample size required to achieve 80% power in a future equal-groups study.
          All main power curves use a single green colour; within Figure&nbsp;2
          a dashed line distinguishes the Bonferroni threshold from the solid
          unadjusted line. In Figure&nbsp;3 the per-group curves use each
          group&rsquo;s assigned colour (dotted lines):
          &nbsp;&mdash;&nbsp;
          <span class="chip" style="background:{COLOR_SPEAR};"></span>KW / Chi-sq / Spearman (all)
          &ensp;
          <span class="chip" style="background:{HUE_PALETTE['No Resource']};"></span>No Resource
          &ensp;
          <span class="chip" style="background:{HUE_PALETTE['PDF']};"></span>PDF
          &ensp;
          <span class="chip" style="background:{HUE_PALETTE['ChatGPT']};"></span>ChatGPT
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


# ── Table legend ───────────────────────────────────────────────────────────────

TABLE_LEGEND = (
    "<strong>Table 1.</strong> Power analysis summary for each statistical test "
    "used in this study. "
    "<em>Obs. Effect</em>: observed effect size from the pilot data, used as a "
    "point estimate (interpret with caution given small sample sizes). "
    "<strong>Cohen&rsquo;s <em>f</em></strong> (Kruskal-Wallis) is the ratio of "
    "between-group to within-group standard deviation of scores "
    "(small&nbsp;&asymp;&nbsp;0.10, medium&nbsp;&asymp;&nbsp;0.25, "
    "large&nbsp;&asymp;&nbsp;0.40). "
    "<strong>Cohen&rsquo;s <em>w</em></strong> (chi-square) measures the "
    "deviation of observed from expected cell proportions, "
    "w&nbsp;=&nbsp;&radic;(&chi;&sup2;&nbsp;/&nbsp;N) "
    "(small&nbsp;&asymp;&nbsp;0.10, medium&nbsp;&asymp;&nbsp;0.30, "
    "large&nbsp;&asymp;&nbsp;0.50). "
    "<strong>Spearman&rsquo;s &rho;</strong> is the rank-order correlation "
    "between self-confidence and number of correct answers "
    "(small&nbsp;&asymp;&nbsp;0.10, medium&nbsp;&asymp;&nbsp;0.30, "
    "large&nbsp;&asymp;&nbsp;0.50). "
    "<em>Min Detectable (at study n)</em>: the smallest effect size detectable "
    "with 80% power at the relevant sample size (\u03b1&nbsp;=&nbsp;0.05 or "
    "Bonferroni threshold); the n used is shown in parentheses. "
    "<em>Req. Total N</em>: for KW and chi-square, the total participants "
    "required across equal-sized groups for 80% power at the observed effect size; "
    "for per-group Spearman rows, the required n <em>within that subgroup alone</em> "
    "(shown with a note in the cell). "
    f"Values&nbsp;&gt;&nbsp;{MAX_N} are displayed as &gt;&nbsp;{MAX_N}. "
    "KW harmonic-mean approximation uses <em>n<sub>h</sub></em>&nbsp;\u2248&nbsp;5.89 "
    "(harmonic mean of groups 7, 5, 6). "
    "Chi-square df&nbsp;=&nbsp;2 (2\u00d73 contingency table). "
    "Spearman power via Fisher z-transform: the full-sample row uses all 18 "
    "participants; the three per-group rows use each group\u2019s own n (7, 5, 6) "
    "and should be interpreted with particular caution given the very small "
    "subsample sizes."
)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["group_label"] = pd.Categorical(df["group_label"],
                                       categories=HUE_ORDER, ordered=True)

    print("Computing observed effect sizes...")
    effects = compute_effect_sizes(df)
    print(f"  Cohen's f  (KW / n_correct):      {effects['cohens_f']:.4f}"
          f"  [{_cohen_f_label(effects['cohens_f'])}]")
    print(f"  Cohen's w  (chi-sq, median):       {effects['cohens_w_median']:.4f}"
          f"  [{_cohen_w_label(effects['cohens_w_median'])}]")
    print(f"  Spearman rho (all N=18):          {effects['spearman_rho']:.4f}"
          f"  [p = {effects['spearman_pval']:.3f}]")
    for g, (r, p, n) in effects["spearman_groups"].items():
        p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"
        print(f"  Spearman rho ({g:12s}): {r:.4f}  [p {p_str}, n={n}]")

    print("\nComputing sensitivity (min detectable effect at N=18, 80% power)...")
    sens_f   = _kw_sensitivity()
    sens_w_u = _chisq_sensitivity(alpha=ALPHA)
    sens_w_b = _chisq_sensitivity(alpha=ALPHA_BONF)
    sens_rho   = _spearman_sensitivity()
    sens_rho_b = _spearman_sensitivity(alpha=ALPHA_BONF)
    print(f"  KW  min f:              {sens_f:.4f}")
    print(f"  Chi min w (a=0.05):     {sens_w_u:.4f}")
    print(f"  Chi min w (Bonf):       {sens_w_b:.4f}")
    if sens_rho:
        print(f"  Spr min |rho|:          {sens_rho:.4f}")
    else:
        print("  Spr min |rho|:          not achievable at N=18")
    if sens_rho_b:
        print(f"  Spr min |rho| (Bonf):   {sens_rho_b:.4f}")
    else:
        print("  Spr min |rho| (Bonf):   not achievable at N=18")

    print("\nComputing required N for 80% power at observed effects...")
    req_kw  = _kw_required_n(effects["cohens_f"])
    req_cu  = _chisq_required_n(effects["cohens_w_median"], alpha=ALPHA)
    req_cb  = _chisq_required_n(effects["cohens_w_median"], alpha=ALPHA_BONF)
    req_rho  = _spearman_required_n(effects["spearman_rho"])
    req_rhob = _spearman_required_n(effects["spearman_rho"], alpha=ALPHA_BONF)
    print(f"  KW  total N:               {req_kw or f'> {MAX_N}'}")
    print(f"  Chi total N (a=0.05):      {req_cu or f'> {MAX_N}'}")
    print(f"  Chi total N (Bonf):        {req_cb or f'> {MAX_N}'}")
    print(f"  Spearman total N:          {req_rho or f'> {MAX_N}'}")
    print(f"  Spearman total N (Bonf):   {req_rhob or f'> {MAX_N}'}")

    print("\nBuilding summary table...")
    table_html, csv_str = build_summary_table(effects)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    CSV_OUT.write_text(csv_str, encoding="utf-8")
    print(f"  CSV saved: {CSV_OUT.relative_to(ROOT)}")

    csv_b64 = b64_str(csv_str)

    print("\nBuilding Plotly power curve figures...")
    legends = build_fig_legends(effects)

    # ── Shared x-axis range for all three figures ─────────────────────────────
    _all_reqs = [req_kw, req_cu, req_cb, req_rho, req_rhob]
    for rho_g, _, _ in effects["spearman_groups"].values():
        if not np.isnan(rho_g):
            _all_reqs.append(_spearman_required_n(rho_g))
    _finite = [n for n in _all_reqs if n is not None]
    common_x_max = min(MAX_N, max(200, (max(_finite) if _finite else 200) + 80))
    print(f"  Common x-axis range: 0 – {common_x_max}")

    figs_raw = [
        (build_fig_kw(effects["cohens_f"], x_max=common_x_max),
         "Kruskal-Wallis Power Curve", "Kruskal-Wallis",
         "fig1_power_kruskal_wallis.pdf", "fig1_power_kruskal_wallis.png"),
        (build_fig_chisq(effects["cohens_w_median"], x_max=common_x_max),
         "Chi-Square Power Curve", "Chi-Square",
         "fig2_power_chi_square.pdf",    "fig2_power_chi_square.png"),
        (build_fig_spearman(effects["spearman_rho"], effects["spearman_groups"],
                            x_max=common_x_max),
         "Spearman Correlation Power Curve", "Spearman Correlation",
         "fig3_power_spearman.pdf",      "fig3_power_spearman.png"),
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
            "legend":       legends[i - 1],
        })

    print("\nAssembling HTML report...")
    html = build_html(figures, table_html, TABLE_LEGEND, csv_b64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")

    size_kb = OUT_HTML.stat().st_size / 1_000
    print(f"\nReport saved: {OUT_HTML.relative_to(ROOT)}")
    print(f"File size: {size_kb:.0f} KB")
    print("Open the HTML file in any browser to view the report.")
