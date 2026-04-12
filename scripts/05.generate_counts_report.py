"""
05.generate_counts_report.py

Generates a self-contained interactive HTML report showing per-question
response distributions (correct / incorrect / not sure) by experimental group,
as horizontal grouped bar charts.

Input:  data/processed/survey_clean.csv
        data/processed/question_labels.csv
Output: reports/counts_report.html

Usage:
    .venv\\Scripts\\python.exe scripts/05.generate_counts_report.py
"""

import base64
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATA_PATH    = ROOT / "data" / "processed" / "survey_clean.csv"
LABELS_PATH  = ROOT / "data" / "processed" / "question_labels.csv"
RAW_PATH     = ROOT / "data" / "raw" / "anonymised" / "dental_trauma_survey_responses.csv"
ANSWERS_PATH = ROOT / "data" / "raw" / "anonymised" / "answers_to_the_questions_form_questions.csv"
OUT_DIR      = ROOT / "reports"
OUT_HTML     = OUT_DIR / "counts_report.html"

# ── Visual constants ───────────────────────────────────────────────────────────
HUE_ORDER     = ["No Resource", "PDF", "ChatGPT"]
HUE_PALETTE   = {"No Resource": "#F8766D", "PDF": "#CD9600", "ChatGPT": "#00A9FF"}
FONT          = "Arial, Helvetica, sans-serif"
OUTCOME_ORDER = ["Correct", "Incorrect", "Not Sure"]

QUESTION_COLS = [
    "c1_injury_type", "c1_treatment", "c1_antibiotics",
    "c2_injury_type", "c2_treatment", "c2_tf_60min",
    "c2_storage_rank", "c2_antibiotics",
    "c3_injury_type", "c3_treatment", "c3_imaging", "c3_antibiotics",
]

# Maps Qualtrics Q-codes (row 0 of raw CSV) to internal column names — same as
# QCODE_TO_COL in 01.data_pre_processing.ipynb.
QCODE_TO_COL = {
    "Q1":  "c1_injury_type",
    "Q2":  "c1_treatment",
    "Q3":  "c1_antibiotics",
    "Q4":  "c2_injury_type",
    "Q39": "c2_treatment",
    "Q41": "c2_tf_60min",
    "Q40": "c2_storage_rank",
    "Q42": "c2_antibiotics",
    "Q44": "c3_injury_type",
    "Q45": "c3_treatment",
    "Q46": "c3_imaging",
    "Q47": "c3_antibiotics",
}

PLOT_CONFIG = {
    "displayModeBar": True,
    "toImageButtonOptions": {"format": "png", "scale": 3},
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "responsive": True,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def b64_str(text: str) -> str:
    """Base64-encode a UTF-8 string for use in data URIs."""
    return base64.b64encode(text.encode("utf-8")).decode()


def build_figure_legend(fig_num: int, label: str, question_text: str,
                        answer_counts: pd.Series, correct_answer: str,
                        group_counts: pd.DataFrame) -> str:
    """
    Build the HTML legend block for one figure, including:
      - Full survey question text
      - Per-answer response counts (all groups combined)
      - Per-answer response counts broken down by group
    group_counts: DataFrame with answer options as index and HUE_ORDER as columns.
    """
    td  = 'style="padding:2px 10px 2px 0;vertical-align:top;"'
    tdr = 'style="padding:2px 6px;text-align:center;white-space:nowrap;color:#555;"'

    # Answers in sorted order (most common first); used as row order in both tables
    sorted_answers = answer_counts.sort_values(ascending=False).index.tolist()
    n_total = int(answer_counts.sum())

    # ── Combined table ────────────────────────────────────────────────────────
    combined_rows = []
    for answer in sorted_answers:
        count = int(answer_counts[answer])
        is_correct = str(answer).strip().lower() == correct_answer.strip().lower()
        check = (
            ' <span style="color:#1a7a3a;font-weight:700;">&#10003;&nbsp;correct</span>'
            if is_correct else ""
        )
        ans_html = (
            f'<span style="font-weight:700;">{answer}</span>' if is_correct
            else str(answer)
        )
        n_word = "respondent" if count == 1 else "respondents"
        combined_rows.append(
            f'<tr>'
            f'<td {td}>{ans_html}{check}</td>'
            f'<td {tdr}>{count}&nbsp;{n_word}</td>'
            f'</tr>'
        )

    # ── By-group table ────────────────────────────────────────────────────────
    group_ns = {g: int(group_counts[g].sum()) for g in HUE_ORDER}
    header_ths = "".join(
        f'<th style="padding:2px 6px;text-align:center;font-weight:700;'
        f'border-bottom:1px solid #d8dce3;">'
        f'{g}<br><span style="font-weight:400;color:#666;">n&nbsp;=&nbsp;{group_ns[g]}'
        f'</span></th>'
        for g in HUE_ORDER
    )

    group_rows = []
    for answer in sorted_answers:
        is_correct = str(answer).strip().lower() == correct_answer.strip().lower()
        ans_html = (
            f'<span style="font-weight:700;">{answer}</span>' if is_correct
            else str(answer)
        )
        group_cells = "".join(
            f'<td {tdr}>'
            f'{int(group_counts.at[answer, g]) if answer in group_counts.index else 0}'
            f'</td>'
            for g in HUE_ORDER
        )
        group_rows.append(
            f'<tr><td {td}>{ans_html}</td>{group_cells}</tr>'
        )

    table_style = (
        'style="margin-top:5px;font-size:12px;border-collapse:collapse;width:auto;"'
    )
    return (
        f'<strong>Figure {fig_num}. {label}.</strong><br>'
        f'<span style="font-style:italic;color:#555;">{question_text}</span>'
        f'<br><br>'
        f'<span style="font-weight:600;">Response counts (all groups combined,'
        f' N&nbsp;=&nbsp;{n_total}):</span>'
        f'<table {table_style}><tbody>{"".join(combined_rows)}</tbody></table>'
        f'<br>'
        f'<span style="font-weight:600;">Breakdown by group:</span>'
        f'<table {table_style}>'
        f'<thead><tr>'
        f'<th style="padding:2px 10px 2px 0;text-align:left;border-bottom:1px solid #d8dce3;">'
        f'Answer</th>'
        f'{header_ths}'
        f'</tr></thead>'
        f'<tbody>{"".join(group_rows)}</tbody>'
        f'</table>'
    )


# ── Data classification ────────────────────────────────────────────────────────

def classify_responses(df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each respondent's answer per question into correct / incorrect / not sure,
    matching the logic in 01.data_pre_processing.ipynb:
      - Correct:   {col}_correct == 1
      - Not Sure:  raw answer stripped + lowercased == "i'm not sure"
      - Incorrect: {col}_correct == 0 AND not "i'm not sure"
      - Excluded:  {col}_correct is NaN (missing / incomplete response)

    Returns a long-form DataFrame with columns:
        [column, label, group, outcome, count]
    """
    label_map = dict(zip(labels_df["column"], labels_df["label"]))
    records = []

    for col in QUESTION_COLS:
        correct_col = f"{col}_correct"
        label = label_map[col]

        for group in HUE_ORDER:
            sub = df[df["group_label"] == group].copy()
            valid = sub[sub[correct_col].notna()]

            n_correct   = int((valid[correct_col] == 1).sum())
            n_not_sure  = int(
                valid[col].str.strip().str.lower().eq("i'm not sure").sum()
            )
            # n_incorrect = all zeros minus the "not sure" zeros
            n_incorrect = int((valid[correct_col] == 0).sum()) - n_not_sure

            records.extend([
                {"column": col, "label": label, "group": group,
                 "outcome": "Correct",   "count": n_correct},
                {"column": col, "label": label, "group": group,
                 "outcome": "Incorrect", "count": n_incorrect},
                {"column": col, "label": label, "group": group,
                 "outcome": "Not Sure",  "count": n_not_sure},
            ])

    counts_df = pd.DataFrame(records)
    counts_df["outcome"] = pd.Categorical(
        counts_df["outcome"], categories=OUTCOME_ORDER, ordered=True
    )
    return counts_df


# ── Table builder ─────────────────────────────────────────────────────────────

def build_summary_table(counts_df: pd.DataFrame) -> tuple[str, str]:
    """
    Build a wide HTML table and CSV string from the long-form counts DataFrame.
    Rows: 12 questions. Columns: group × outcome sub-columns.
    Returns (html_table_string, csv_string).
    """
    pivot = counts_df.pivot_table(
        index=["column", "label"],
        columns=["group", "outcome"],
        values="count",
        aggfunc="first",
    )
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_product([HUE_ORDER, OUTCOME_ORDER])
    )
    # Preserve original question order
    ordered_index = [(col, counts_df.loc[counts_df["column"] == col, "label"].iloc[0])
                     for col in QUESTION_COLS]
    pivot = pivot.loc[ordered_index]

    # Two-row header: group (colspan=3) then outcome
    group_ths = "".join(
        f'<th colspan="3" style="text-align:center;border-bottom:1px solid #d8dce3;">'
        f'{g}</th>'
        for g in HUE_ORDER
    )
    outcome_ths = "".join(
        f'<th style="text-align:center;">{o}</th>'
        for _ in HUE_ORDER for o in OUTCOME_ORDER
    )

    rows_html = []
    for (col, label), row in pivot.iterrows():
        cells = f"<td>{label}</td>"
        for group in HUE_ORDER:
            for outcome in OUTCOME_ORDER:
                val = int(row.get((group, outcome), 0))
                cells += f'<td style="text-align:center;">{val}</td>'
        rows_html.append(f"<tr>{cells}</tr>")

    table_html = (
        "<table>"
        "<thead>"
        f"<tr><th rowspan=\"2\">Question</th>{group_ths}</tr>"
        f"<tr>{outcome_ths}</tr>"
        "</thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )

    # CSV
    col_headers = ",".join(
        f"{g} \u2014 {o}" for g in HUE_ORDER for o in OUTCOME_ORDER
    )
    csv_lines = [f"Question,{col_headers}"]
    for (col, label), row in pivot.iterrows():
        vals = ",".join(
            str(int(row.get((group, outcome), 0)))
            for group in HUE_ORDER for outcome in OUTCOME_ORDER
        )
        csv_lines.append(f'"{label}",{vals}')
    csv_str = "\n".join(csv_lines)

    return table_html, csv_str


# ── Figure builder ─────────────────────────────────────────────────────────────

def build_count_figure(label: str, question_df: pd.DataFrame) -> go.Figure:
    """
    Build a horizontal grouped bar chart for one survey question.
    Y-axis: outcome categories (Correct, Incorrect, Not Sure) displayed top-to-bottom.
    X-axis: count of respondents.
    One bar per group (No Resource, PDF, ChatGPT), dodged within each outcome.
    """
    fig = go.Figure()
    for group in HUE_ORDER:
        color = HUE_PALETTE[group]
        sub = question_df[question_df["group"] == group].sort_values("outcome")
        fig.add_trace(go.Bar(
            y=sub["outcome"].tolist(),
            x=sub["count"].tolist(),
            name=group,
            orientation="h",
            marker_color=color,
            marker_line=dict(color=color, width=0.5),
        ))

    fig.update_layout(
        title=dict(text=f"<b>{label}</b>", font=dict(size=16, family=FONT), x=0.5),
        xaxis=dict(
            title=dict(text="Count", font=dict(size=13, family=FONT)),
            showgrid=False, zeroline=False,
            showline=True, linecolor="#d0d0d0",
            tickfont=dict(size=12, family=FONT),
            dtick=1,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            showline=True, linecolor="#d0d0d0",
            tickfont=dict(size=12, family=FONT),
            categoryorder="array",
            categoryarray=list(reversed(OUTCOME_ORDER)),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT, size=13),
        barmode="group",
        showlegend=True,
        legend=dict(
            title=dict(text="Group", font=dict(size=12, family=FONT)),
            font=dict(size=12, family=FONT),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#d0d0d0",
            borderwidth=1,
        ),
        margin=dict(l=65, r=30, t=65, b=55),
        height=340,
    )
    return fig


def build_answer_figure(label: str, group_counts: pd.DataFrame,
                        answer_counts: pd.Series, correct_answer: str) -> go.Figure:
    """
    Build a horizontal grouped bar chart showing raw answer-option counts by group.
    Y-axis: actual answer text (most common at top; correct answer marked with ✓).
    X-axis: count per group (No Resource, PDF, ChatGPT), dodged.
    Height is computed dynamically from the number of distinct answer options.
    """
    import textwrap

    sorted_answers = answer_counts.sort_values(ascending=False).index.tolist()

    # Wrap long answer text and mark the correct answer with ✓
    y_labels = []
    for a in sorted_answers:
        wrapped = textwrap.fill(str(a), width=42).replace("\n", "<br>")
        is_correct = str(a).strip().lower() == correct_answer.strip().lower()
        y_labels.append(wrapped + (" ✓" if is_correct else ""))

    answer_to_label = dict(zip(sorted_answers, y_labels))

    fig = go.Figure()
    for group in HUE_ORDER:
        color = HUE_PALETTE[group]
        x_vals = [
            int(group_counts.at[a, group]) if a in group_counts.index else 0
            for a in sorted_answers
        ]
        y_vals = [answer_to_label[a] for a in sorted_answers]
        fig.add_trace(go.Bar(
            y=y_vals,
            x=x_vals,
            name=group,
            orientation="h",
            marker_color=color,
            marker_line=dict(color=color, width=0.5),
        ))

    n_answers = len(sorted_answers)
    height = max(320, 110 + n_answers * 65)

    fig.update_layout(
        title=dict(
            text=f"<b>{label} \u2014 Answers by group</b>",
            font=dict(size=16, family=FONT), x=0.5,
        ),
        xaxis=dict(
            title=dict(text="Count", font=dict(size=13, family=FONT)),
            showgrid=False, zeroline=False,
            showline=True, linecolor="#d0d0d0",
            tickfont=dict(size=12, family=FONT),
            dtick=1,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            showline=True, linecolor="#d0d0d0",
            tickfont=dict(size=11, family=FONT),
            categoryorder="array",
            categoryarray=list(reversed(y_labels)),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT, size=13),
        barmode="group",
        showlegend=True,
        legend=dict(
            title=dict(text="Group", font=dict(size=12, family=FONT)),
            font=dict(size=12, family=FONT),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#d0d0d0",
            borderwidth=1,
        ),
        margin=dict(l=320, r=30, t=65, b=55),
        height=height,
    )
    return fig


def fig_to_html_div(fig: go.Figure, div_id: str) -> str:
    """Render a Plotly figure as an embeddable HTML div (no full page, no plotly.js)."""
    config = {
        **PLOT_CONFIG,
        "toImageButtonOptions": {
            **PLOT_CONFIG["toImageButtonOptions"],
            "filename": div_id,
        },
    }
    return pio.to_html(
        fig, full_html=False, include_plotlyjs=False,
        config=config, div_id=div_id,
    )


def fig_to_pdf_b64(fig: go.Figure, height: int = 420) -> str:
    """Rasterise a Plotly figure to PDF bytes via kaleido and return as base64."""
    pdf_bytes = pio.to_image(fig, format="pdf", width=900, height=height)
    return base64.b64encode(pdf_bytes).decode()


# ── Report text ───────────────────────────────────────────────────────────────

PREAMBLE = """\
<p>
This report presents the per-question response breakdown for a three-arm, randomized
survey-based study examining emergency medicine providers' knowledge of dental trauma
management. Participants (N&nbsp;=&nbsp;18) were randomly assigned to one of three
experimental groups — <strong>No Resource</strong> (n&nbsp;=&nbsp;7),
<strong>PDF</strong> reference guide (n&nbsp;=&nbsp;5), or
<strong>ChatGPT</strong> (n&nbsp;=&nbsp;6) — and completed a 12-item clinical vignette
assessment covering three cases: primary tooth avulsion (Case 1, 3 questions), permanent
tooth avulsion (Case 2, 5 questions), and complicated crown fracture of a permanent incisor
(Case 3, 4 questions).
</p>
<p>
For each question, two charts are shown side by side: an outcome chart (correct / incorrect /
not sure counts by group) and an answer-breakdown chart (raw response counts by group, with
the correct answer marked &#10003;). The table above the figures summarises all 12 questions
in a single view. Each response was classified as <em>correct</em> (matches the answer key),
<em>not sure</em> (participant selected &#8220;I&#8217;m not sure&#8221;), or
<em>incorrect</em> (any other response). Participants who did not complete a question are
excluded from all counts.
</p>
"""

TABLE_LEGEND = (
    "<strong>Table 1.</strong> Response counts (correct / incorrect / not sure) for each of "
    "the 12 survey questions, broken down by experimental group. "
    "Groups: No Resource n&nbsp;=&nbsp;7, PDF n&nbsp;=&nbsp;5, ChatGPT n&nbsp;=&nbsp;6. "
    "Respondents who did not complete a question (missing response) are excluded."
)


# ── CSS ───────────────────────────────────────────────────────────────────────

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

# ── JS ────────────────────────────────────────────────────────────────────────

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


# ── HTML sections ─────────────────────────────────────────────────────────────

def download_btn(label: str, b64: str, filename: str, mime: str, css_class: str) -> str:
    """Render an inline download button that triggers a base64 data URI download."""
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


def _plot_block(div_id: str, fig_div: str, pdf_b64: str, pdf_filename: str,
               png_filename: str, default_w: int, default_h: int,
               subtitle: str | None = None) -> str:
    """Render one plot sub-block: optional subtitle, resize controls, buttons, plot div."""
    ar         = round(default_h / default_w, 4)
    pdf_btn    = download_btn(
        "\u2b07 Download PDF", pdf_b64, pdf_filename, "application/pdf", "btn-pdf"
    )
    png_btn    = (
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
    resize = (
        f'<div class="resize-controls">'
        f'<span>&#8596; Resize figure:</span>'
        f'<label>W: <span id="{div_id}-wval" class="resize-val">{default_w}px</span></label>'
        f'<input type="range" id="{div_id}-w" min="400" max="1400" value="{default_w}"'
        f' oninput="updateFigSize(\'{div_id}\')">'
        f'<label>'
        f'<input type="checkbox" id="{div_id}-lock" checked onchange="updateFigSize(\'{div_id}\')">'
        f' Lock aspect ratio'
        f'</label>'
        f'<span id="{div_id}-ar" data-ratio="{ar}" style="display:none"></span>'
        f'<div id="{div_id}-hrow" class="resize-hrow">'
        f'<label>H: <span id="{div_id}-hval" class="resize-val">{default_h}px</span></label>'
        f'<input type="range" id="{div_id}-h" min="200" max="1000" value="{default_h}"'
        f' oninput="updateFigSize(\'{div_id}\')">'
        f'</div>'
        f'</div>'
    )
    font_ctrl = font_size_controls(div_id)
    sub_html = (
        f'<p style="font-size:13px;font-weight:600;color:#444;margin:16px 0 6px;">'
        f'{subtitle}</p>'
        if subtitle else ""
    )
    return (
        f'{sub_html}'
        f'{resize}'
        f'{font_ctrl}'
        f'<div class="download-bar" style="margin-bottom:6px;">'
        f'{png_btn}{pdf_btn}{legend_btn}{title_btn}</div>'
        f'<p style="font-size:11.5px;color:#888;margin-bottom:10px;">'
        f'<strong>PNG</strong> reflects the current figure size and legend visibility.'
        f'&ensp;<strong>PDF</strong> is pre-rendered at a fixed size (900&nbsp;px wide) with the'
        f' legend always shown &mdash; resize and legend-toggle have no effect on it.'
        f'</p>'
        f'<div class="figure-plot">{fig_div}</div>'
    )


def figure_section(fig_num: int, title: str,
                   fig_div: str, pdf_b64: str, pdf_filename: str, png_filename: str,
                   legend: str,
                   default_w: int = 800, default_h: int = 340,
                   fig2_div: str | None = None, fig2_pdf_b64: str | None = None,
                   fig2_pdf_filename: str | None = None,
                   fig2_png_filename: str | None = None,
                   default_w2: int = 800, default_h2: int = 450) -> str:
    """
    Render a complete figure block.
    If fig2_* parameters are supplied, a second plot (answer breakdown) is rendered
    below the first, each with its own resize controls and download buttons.
    The shared legend appears at the bottom.
    """
    div_id  = f"plotly-fig{fig_num}"
    div_id2 = f"plotly-fig{fig_num}b"

    block1 = _plot_block(
        div_id, fig_div, pdf_b64, pdf_filename, png_filename,
        default_w, default_h,
        subtitle="Outcomes by group" if fig2_div else None,
    )

    block2 = ""
    if fig2_div is not None:
        block2 = (
            f'<div style="margin-top:20px;padding-top:16px;border-top:1px solid #f0f2f5;">'
            + _plot_block(
                div_id2, fig2_div, fig2_pdf_b64, fig2_pdf_filename, fig2_png_filename,
                default_w2, default_h2,
                subtitle="Answers by group",
            )
            + f'</div>'
        )

    return (
        f'<div class="figure-block" id="fig{fig_num}">'
        f'<h3>Figure {fig_num} \u2014 {title}</h3>'
        f'{block1}'
        f'{block2}'
        f'<div class="legend-text" style="margin-top:14px;">{legend}</div>'
        f'</div>'
    )


def table_section(section_id: str, title: str, table_html: str, legend: str,
                  download_bar_html: str = "") -> str:
    """Render a complete table block with an optional download bar and legend."""
    bar = (
        f'<div class="download-bar" style="margin-top:10px;">{download_bar_html}</div>'
        if download_bar_html else ""
    )
    return (
        f'<div id="{section_id}" style="margin-bottom:28px;">'
        f'<h3 style="font-size:16px;font-weight:700;margin-bottom:12px;">{title}</h3>'
        f'<div class="table-wrap">{table_html}</div>'
        f'{bar}'
        f'<div class="legend-text">{legend}</div>'
        f'</div>'
    )


def build_html(figures: list[dict], tbl_html: str, tbl_csv_b64: str) -> str:
    """Assemble the full HTML counts report string."""
    fig_sections = "\n".join(
        figure_section(
            i + 1, f["title"],
            f["div"],   f["pdf_b64"],   f["pdf_filename"],   f["png_filename"],
            f["legend"],
            default_h=f.get("default_h", 340),
            fig2_div=f.get("fig2_div"),
            fig2_pdf_b64=f.get("fig2_pdf_b64"),
            fig2_pdf_filename=f.get("fig2_pdf_filename"),
            fig2_png_filename=f.get("fig2_png_filename"),
            default_h2=f.get("default_h2", 450),
        )
        for i, f in enumerate(figures)
    )

    tbl_section = table_section(
        "counts-table",
        "Table 1 \u2014 Response Counts by Question and Group",
        tbl_html,
        TABLE_LEGEND,
        download_bar_html=download_btn(
            "\u2b07 Download CSV", tbl_csv_b64, "counts_by_question.csv",
            "text/csv", "btn-csv"
        ),
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
  <title>Dental Trauma Study \u2014 Counts Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>{CSS}</style>
</head>
<body>
<div class="layout">

  <!-- \u2500\u2500 Sidebar \u2500\u2500 -->
  <aside class="sidebar">
    <h2>Contents</h2>
    <nav>
      <ul>
        <li class="section-head"><a href="#counts-table">Table 1</a></li>
        <li class="section-head"><a href="#figures">Figures</a></li>
{sidebar_figs}
      </ul>
    </nav>
  </aside>

  <!-- \u2500\u2500 Main content \u2500\u2500 -->
  <main class="content">
    <div class="report-title">Dental Trauma Study \u2014 Counts Report</div>
    <div class="report-subtitle">
      Per-question response distributions by group &nbsp;\u00b7&nbsp;
      No Resource &nbsp;\u00b7&nbsp; PDF &nbsp;\u00b7&nbsp; ChatGPT
      &nbsp;|&nbsp; N&nbsp;=&nbsp;18 participants
    </div>

    <!-- Preamble -->
    <section class="section">
      <h2>Overview</h2>
      {PREAMBLE}
    </section>

    <!-- Table -->
    <section class="section">
      <h2>Summary Table</h2>
      {tbl_section}
    </section>

    <!-- Figures -->
    <section class="section" id="figures">
      <h2>Figures</h2>
      <p style="font-size:13px;color:#666;margin-bottom:20px;">
        <strong>Color key:</strong>&nbsp;
        <span class="chip" style="background:#F8766D;"></span>No Resource&ensp;
        <span class="chip" style="background:#CD9600;"></span>PDF&ensp;
        <span class="chip" style="background:#00A9FF;"></span>ChatGPT
        &nbsp;&mdash;&nbsp;Color coding is consistent across all figures.
      </p>
      {fig_sections}
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
    df["group_label"] = pd.Categorical(
        df["group_label"], categories=HUE_ORDER, ordered=True
    )
    labels_df = pd.read_csv(LABELS_PATH)

    # Load full question text from row 1 of the raw Qualtrics CSV
    df_raw_header = pd.read_csv(RAW_PATH, header=0, nrows=2)
    question_text_row = df_raw_header.iloc[0]  # row 1 = full question text
    col_to_question = {
        col: str(question_text_row[qcode])
        for qcode, col in QCODE_TO_COL.items()
    }

    # Load answer key
    answers_df = pd.read_csv(ANSWERS_PATH)
    answer_key = {
        QCODE_TO_COL[row["question_col"]]: row["correct_answer"]
        for _, row in answers_df.iterrows()
    }

    print("Classifying responses...")
    counts_df = classify_responses(df, labels_df)

    print("Building summary table...")
    tbl_html, tbl_csv = build_summary_table(counts_df)
    tbl_csv_b64 = b64_str(tbl_csv)

    # Build one figure per question
    label_map = dict(zip(labels_df["column"], labels_df["label"]))
    figs_raw = [
        (col, label_map[col])
        for col in QUESTION_COLS
    ]

    print("Building Plotly figures...")
    figures = []
    for i, (col, label) in enumerate(figs_raw, start=1):
        q_df           = counts_df[counts_df["column"] == col]
        fig            = build_count_figure(label, q_df)
        answer_counts  = df[col].value_counts(dropna=True)
        question_text  = col_to_question[col]
        correct_answer = answer_key[col]
        # Per-group counts: index = answer option, columns = group labels
        group_counts = (
            df.groupby([col, "group_label"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=HUE_ORDER, fill_value=0)
        )

        fig2       = build_answer_figure(label, group_counts, answer_counts, correct_answer)
        n_answers  = len(answer_counts)
        default_h2 = max(320, 110 + n_answers * 65)

        safe_label  = col.replace("_", "-")
        pdf_name    = f"fig{i:02d}_{safe_label}.pdf"
        png_name    = f"fig{i:02d}_{safe_label}.png"
        pdf2_name   = f"fig{i:02d}_{safe_label}_answers.pdf"
        png2_name   = f"fig{i:02d}_{safe_label}_answers.png"
        div_id      = f"plotly-fig{i}"
        div_id2     = f"plotly-fig{i}b"

        figures.append({
            "title":           label,
            "short":           label,
            "div":             fig_to_html_div(fig, div_id),
            "pdf_b64":         None,   # filled in next loop
            "pdf_filename":    pdf_name,
            "png_filename":    png_name,
            "default_h":       340,
            "fig2_div":        fig_to_html_div(fig2, div_id2),
            "fig2_pdf_b64":    None,   # filled in next loop
            "fig2_pdf_filename": pdf2_name,
            "fig2_png_filename": png2_name,
            "default_h2":      default_h2,
            "legend":          build_figure_legend(
                                    i, label, question_text,
                                    answer_counts, correct_answer, group_counts,
                                ),
            "_fig":            fig,    # kept temporarily for PDF rendering
            "_fig2":           fig2,
            "_pdf_h2":         default_h2,
        })

    print("Rendering PDFs via kaleido (this may take a moment)...")
    for i, entry in enumerate(figures, start=1):
        print(f"  Figure {i}: {entry['title']}")
        entry["pdf_b64"]  = fig_to_pdf_b64(entry.pop("_fig"))
        entry["fig2_pdf_b64"] = fig_to_pdf_b64(entry.pop("_fig2"),
                                               height=entry.pop("_pdf_h2"))

    print("Assembling HTML report...")
    html = build_html(figures, tbl_html, tbl_csv_b64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")

    size_kb = OUT_HTML.stat().st_size / 1_000
    print(f"\nReport saved: {OUT_HTML.relative_to(ROOT)}")
    print(f"File size: {size_kb:.0f} KB")
    print("Open the HTML file in any browser to view the report.")
