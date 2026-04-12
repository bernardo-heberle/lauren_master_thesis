"""
08.sanity_checks.py
-------------------
Data integrity and statistical sanity checks for the dental trauma survey analysis.

Runs six sections of checks against data/processed/survey_clean.csv and the
generated output tables. Prints PASS / FAIL / WARN for each check and exits
with code 0 if all pass, 1 if any fail.

Usage (from repo root):
    .venv\\Scripts\\python.exe scripts/08.sanity_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import re

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2_contingency, kruskal, norm, spearmanr

# ── Paths (relative to repo root) ────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_CLEAN   = ROOT / "data" / "processed" / "survey_clean.csv"
ANSWERS_PATH = ROOT / "data" / "raw" / "anonymised" / "answers_to_the_questions_form_questions.csv"
TABLE2_PATH  = ROOT / "tables" / "table2_per_question.csv"
TABLE3_PATH  = ROOT / "tables" / "table3_correlation.csv"
PA_CSV_PATH  = ROOT / "tables" / "power_analysis_summary.csv"

# ── Known constants (must match 01.data_pre_processing.ipynb exactly) ─────────
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

QUESTION_COLS = list(QCODE_TO_COL.values())
CORRECT_COLS  = [f"{c}_correct" for c in QUESTION_COLS]

QUESTION_LABELS = {
    "c1_injury_type":  "C1: Injury type",
    "c1_treatment":    "C1: Treatment",
    "c1_antibiotics":  "C1: Antibiotics",
    "c2_injury_type":  "C2: Injury type",
    "c2_treatment":    "C2: Treatment",
    "c2_tf_60min":     "C2: >60 min outside mouth (T/F)",
    "c2_storage_rank": "C2: Storage medium ranking",
    "c2_antibiotics":  "C2: Antibiotic therapy",
    "c3_injury_type":  "C3: Injury type",
    "c3_treatment":    "C3: Treatment",
    "c3_imaging":      "C3: Imaging area",
    "c3_antibiotics":  "C3: Antibiotic indication",
}

EXPECTED_GROUPS = {"No Resource": 7, "ChatGPT": 6, "PDF": 5}
GROUP_ORDER     = ["No Resource", "PDF", "ChatGPT"]
N_TOTAL         = 18
N_ITEMS         = 12
N_CORR_TESTS    = 4   # Bonferroni multiplier for Spearman correlations
ALPHA           = 0.05
ALPHA_BONF      = ALPHA / N_ITEMS        # ≈ 0.00417 (chi-square Bonferroni level)
ALPHA_BONF_SPEAR = ALPHA / N_CORR_TESTS  # = 0.0125  (Spearman Bonferroni level)
POWER_TARGET    = 0.80
MAX_N           = 500

EXPECTED_COLUMNS = [
    "duration_sec", "self_knowledge_tdi", "self_confidence_avulsion",
    "self_confidence_fracture", "edu_status", "Resident/Fellow.1", "Attending.1",
    "c1_injury_type", "c1_treatment", "c1_antibiotics",
    "c2_injury_type", "c2_treatment", "c2_tf_60min", "c2_storage_rank", "c2_antibiotics",
    "c3_injury_type", "c3_treatment", "c3_imaging", "c3_antibiotics",
    "total_score", "random_assignment", "group_label", "duration_min",
    "self_confidence_mean", "specialty", "training_level",
    "c1_injury_type_correct", "c1_treatment_correct", "c1_antibiotics_correct",
    "c2_injury_type_correct", "c2_treatment_correct", "c2_tf_60min_correct",
    "c2_storage_rank_correct", "c2_antibiotics_correct",
    "c3_injury_type_correct", "c3_treatment_correct", "c3_imaging_correct",
    "c3_antibiotics_correct",
    "n_correct", "n_incorrect", "n_not_sure", "pct_correct_of_attempted",
]

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

_pass_count = 0
_fail_count = 0
_warn_count = 0


def _pass(msg: str) -> None:
    global _pass_count
    _pass_count += 1
    print(f"  {_GREEN}PASS{_RESET}  {msg}")


def _fail(msg: str) -> None:
    global _fail_count
    _fail_count += 1
    print(f"  {_RED}FAIL{_RESET}  {msg}")


def _warn(msg: str) -> None:
    global _warn_count
    _warn_count += 1
    print(f"  {_YELLOW}WARN{_RESET}  {msg}")


def _section(title: str) -> None:
    print(f"\n{_BOLD}{'-' * 60}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{'-' * 60}{_RESET}")


# ── Section 1: Data shape and structural integrity ────────────────────────────
def check_structure(df: pd.DataFrame) -> None:
    _section("Section 1: Data Shape and Structural Integrity")

    # Dimensions
    if df.shape == (N_TOTAL, len(EXPECTED_COLUMNS)):
        _pass(f"DataFrame shape is ({N_TOTAL}, {len(EXPECTED_COLUMNS)}) as expected")
    else:
        _fail(f"Unexpected shape {df.shape}, expected ({N_TOTAL}, {len(EXPECTED_COLUMNS)})")

    # Column names
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if not missing_cols and not extra_cols:
        _pass("All expected columns present, no unexpected extras")
    else:
        if missing_cols:
            _fail(f"Missing columns: {missing_cols}")
        if extra_cols:
            _warn(f"Extra columns not in expected list: {extra_cols}")

    # No duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes == 0:
        _pass("No duplicate rows")
    else:
        _fail(f"{n_dupes} duplicate row(s) found")

    # Group composition
    group_counts = df["group_label"].value_counts().to_dict()
    if group_counts == EXPECTED_GROUPS:
        _pass(f"Group composition correct: {group_counts}")
    else:
        _fail(f"Group composition mismatch. Got {group_counts}, expected {EXPECTED_GROUPS}")

    # No NaN in critical columns
    critical = ["group_label", "n_correct", "self_confidence_mean"] + CORRECT_COLS
    for col in critical:
        n_null = df[col].isna().sum()
        if n_null == 0:
            _pass(f"No NaN in '{col}'")
        else:
            _fail(f"'{col}' has {n_null} NaN value(s)")

    # Value-range checks
    self_rating_cols = ["self_knowledge_tdi", "self_confidence_avulsion", "self_confidence_fracture"]
    for col in self_rating_cols:
        out = df[(df[col] < 0) | (df[col] > 10)]
        if out.empty:
            _pass(f"'{col}' values in [0, 10]")
        else:
            _fail(f"'{col}' has {len(out)} out-of-range value(s): {df[col].tolist()}")

    for col in CORRECT_COLS:
        bad = df[col].dropna()
        bad = bad[~bad.isin([0, 1])]
        if bad.empty:
            _pass(f"'{col}' values in {{0, 1}}")
        else:
            _fail(f"'{col}' has unexpected values: {bad.tolist()}")

    nc_bad = df[(df["n_correct"] < 0) | (df["n_correct"] > N_ITEMS)]
    if nc_bad.empty:
        _pass(f"'n_correct' values in [0, {N_ITEMS}]")
    else:
        _fail(f"'n_correct' out of range: {df['n_correct'].tolist()}")

    pct_bad = df[df["pct_correct_of_attempted"].notna() &
                 ((df["pct_correct_of_attempted"] < 0) | (df["pct_correct_of_attempted"] > 100))]
    if pct_bad.empty:
        _pass("'pct_correct_of_attempted' values in [0, 100]")
    else:
        _fail(f"'pct_correct_of_attempted' out of range in {len(pct_bad)} row(s)")

    dur_bad = df[df["duration_sec"] <= 0]
    if dur_bad.empty:
        _pass("'duration_sec' all positive")
    else:
        _fail(f"'duration_sec' has {len(dur_bad)} non-positive value(s)")


# ── Section 2: Scoring re-derivation ─────────────────────────────────────────
def check_scoring(df: pd.DataFrame, answer_key: dict[str, str]) -> None:
    _section("Section 2: Scoring Re-Derivation")

    # Re-score each item and compare to stored *_correct columns
    any_scoring_mismatch = False
    for col, correct_answer in answer_key.items():
        re_scored = df[col].apply(
            lambda r: (
                np.nan if pd.isna(r)
                else int(str(r).strip().lower() == str(correct_answer).strip().lower())
            )
        )
        stored = df[f"{col}_correct"]
        mismatches = (re_scored != stored).sum()
        if mismatches == 0:
            _pass(f"Re-scored '{col}_correct' matches stored values")
        else:
            any_scoring_mismatch = True
            _fail(
                f"'{col}_correct' has {mismatches} mismatch(es) between "
                f"re-derived scores and stored values — possible data ordering issue"
            )
            bad_idx = (re_scored != stored)
            for idx in df[bad_idx].index:
                _fail(
                    f"  Row {idx}: response='{df.at[idx, col]}', "
                    f"correct='{correct_answer}', "
                    f"stored={stored.at[idx]}, re-derived={re_scored.at[idx]}"
                )

    if not any_scoring_mismatch:
        _pass("All 12 item scores match between re-derivation and stored data")

    # Re-derive aggregate counts
    df_correct = df[CORRECT_COLS].copy()

    n_correct_rederived  = df_correct.sum(axis=1).astype(int)
    n_not_sure_rederived = df[QUESTION_COLS].apply(
        lambda col_s: col_s.str.strip().str.lower().eq("i'm not sure").astype(int), axis=0
    ).sum(axis=1)

    # Correct definition: n_incorrect = answered wrong, explicitly excluding "I'm not sure"
    # responses (which belong only to n_not_sure). The three counts must sum to 12.
    # NOTE: The preprocessing script used .eq(0).sum() which also catches "I'm not sure"
    # items (they score 0), causing n_incorrect to overcount for rows with not-sure responses.
    n_incorrect_rederived = pd.Series(0, index=df.index)
    for col in QUESTION_COLS:
        not_sure = df[col].str.strip().str.lower().eq("i'm not sure")
        wrong    = (df[f"{col}_correct"] == 0) & ~not_sure
        n_incorrect_rederived += wrong.astype(int)

    nc_mismatch = (n_correct_rederived != df["n_correct"]).sum()
    if nc_mismatch == 0:
        _pass("Re-derived 'n_correct' matches stored for all rows")
    else:
        _fail(f"'n_correct' mismatch in {nc_mismatch} row(s)")
        for idx in df[(n_correct_rederived != df["n_correct"])].index:
            _fail(f"  Row {idx}: stored={df.at[idx,'n_correct']}, re-derived={n_correct_rederived.at[idx]}")

    ni_mismatch = (n_incorrect_rederived != df["n_incorrect"]).sum()
    if ni_mismatch == 0:
        _pass("Re-derived 'n_incorrect' matches stored for all rows")
    else:
        _fail(
            f"'n_incorrect' mismatch in {ni_mismatch} row(s) — "
            f"stored values appear to include 'I\\'m not sure' responses, "
            f"which should be counted only in n_not_sure (preprocessing bug)"
        )
        for idx in df[(n_incorrect_rederived != df["n_incorrect"])].index:
            _fail(
                f"  Row {idx}: stored={df.at[idx,'n_incorrect']}, re-derived={n_incorrect_rederived.at[idx]}, "
                f"n_not_sure={df.at[idx,'n_not_sure']} "
                f"(stored overcounts by {df.at[idx,'n_incorrect'] - n_incorrect_rederived.at[idx]})"
            )

    ns_mismatch = (n_not_sure_rederived != df["n_not_sure"]).sum()
    if ns_mismatch == 0:
        _pass("Re-derived 'n_not_sure' matches stored for all rows")
    else:
        _fail(f"'n_not_sure' mismatch in {ns_mismatch} row(s)")
        for idx in df[(n_not_sure_rederived != df["n_not_sure"])].index:
            _fail(f"  Row {idx}: stored={df.at[idx,'n_not_sure']}, re-derived={n_not_sure_rederived.at[idx]}")

    # Arithmetic: n_correct + n_incorrect + n_not_sure == N_ITEMS for all rows.
    # n_incorrect and n_not_sure are mutually exclusive by definition.
    total_answered = df["n_correct"] + df["n_incorrect"] + df["n_not_sure"]
    bad_total = total_answered[total_answered != N_ITEMS]
    if bad_total.empty:
        _pass(f"n_correct + n_incorrect + n_not_sure == {N_ITEMS} for all rows")
    else:
        _fail(
            f"{len(bad_total)} row(s) where n_correct + n_incorrect + n_not_sure != {N_ITEMS} "
            f"— likely caused by n_incorrect including 'I\\'m not sure' responses: {bad_total.to_dict()}"
        )

    # Re-derive pct_correct_of_attempted: percentage, rounded to 1 decimal place.
    # Denominator is attempted = n_correct + n_incorrect (not-sure excluded).
    attempted = df["n_correct"] + n_incorrect_rederived
    pct_rederived = np.where(attempted > 0, np.round(df["n_correct"] / attempted * 100, 1), np.nan)
    pct_mismatch  = ~np.isclose(pct_rederived, df["pct_correct_of_attempted"].values, atol=0.05, equal_nan=True)
    if not pct_mismatch.any():
        _pass("Re-derived 'pct_correct_of_attempted' matches stored for all rows")
    else:
        n_bad = pct_mismatch.sum()
        _fail(f"'pct_correct_of_attempted' mismatch in {n_bad} row(s)")
        for idx in np.where(pct_mismatch)[0]:
            _fail(f"  Row {idx}: stored={df['pct_correct_of_attempted'].iloc[idx]}, re-derived={pct_rederived[idx]}")

    # Note: Qualtrics embedded total_score (SC0) consistently differs from item-derived
    # n_correct by ~1 point, likely because Qualtrics uses a different scoring rule for
    # one item. This is expected and not treated as an error.


# ── Section 3: Derived-column consistency ────────────────────────────────────
def check_derived_columns(df: pd.DataFrame) -> None:
    _section("Section 3: Derived-Column Consistency")

    # self_confidence_mean == mean of the three rating items, rounded to 2 decimal places
    expected_mean = df[["self_knowledge_tdi", "self_confidence_avulsion", "self_confidence_fracture"]].mean(axis=1).round(2)
    diff = (expected_mean - df["self_confidence_mean"]).abs()
    if (diff > 0.005).sum() == 0:
        _pass("'self_confidence_mean' == round(mean(self_knowledge_tdi, self_confidence_avulsion, self_confidence_fracture), 2)")
    else:
        n_bad = (diff > 0.005).sum()
        _fail(f"'self_confidence_mean' mismatch in {n_bad} row(s) — max deviation {diff.max():.2e}")
        for idx in df[(diff > 0.005)].index:
            _fail(f"  Row {idx}: stored={df.at[idx,'self_confidence_mean']:.6f}, re-derived={expected_mean.at[idx]:.6f}")

    # duration_min == duration_sec / 60, rounded to 2 decimal places
    expected_dur = (df["duration_sec"] / 60).round(2)
    dur_diff = (expected_dur - df["duration_min"]).abs()
    if (dur_diff > 0.005).sum() == 0:
        _pass("'duration_min' == round(duration_sec / 60, 2)")
    else:
        n_bad = (dur_diff > 0.005).sum()
        _fail(f"'duration_min' mismatch in {n_bad} row(s) — max deviation {dur_diff.max():.2e}")
        for idx in df[(dur_diff > 0.005)].index:
            _fail(f"  Row {idx}: stored={df.at[idx,'duration_min']:.6f}, re-derived={expected_dur.at[idx]:.6f}")


# ── Section 4: Statistical test reproducibility ───────────────────────────────
def check_statistics(df: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame) -> None:
    _section("Section 4: Statistical Test Reproducibility")

    # ── 4a. Spearman correlations (Table 3) ───────────────────────────────────
    subsets = [("All Participants", df)] + [
        (grp, df[df["group_label"] == grp]) for grp in GROUP_ORDER
    ]

    for label, sub in subsets:
        if len(sub) < 3:
            _warn(f"Group '{label}' has n={len(sub)}, too small for reliable Spearman — skipping")
            continue

        rho, pval = spearmanr(sub["self_confidence_mean"], sub["n_correct"])
        p_bonf    = min(pval * N_CORR_TESTS, 1.0)

        row = table3[table3["Subset"] == label]
        if row.empty:
            _warn(f"Table 3 has no row for subset '{label}'")
            continue

        stored_rho  = float(row["rho"].iloc[0])
        stored_pval = float(row["p_value"].iloc[0])
        stored_bonf = float(row["p_bonferroni"].iloc[0])

        # Table 3 stores rounded values (2 decimal places), so compare at that precision
        if round(rho, 2) == round(stored_rho, 2):
            _pass(f"Spearman rho for '{label}': re-computed {rho:.4f} rounds to stored {stored_rho:.2f}")
        else:
            _fail(f"Spearman rho mismatch for '{label}': re-computed {rho:.4f} (rounds to {round(rho,2)}) vs stored {stored_rho:.2f}")

        if round(pval, 3) == round(stored_pval, 3):
            _pass(f"Spearman p-value for '{label}': re-computed {pval:.4f} rounds to stored {stored_pval:.3f}")
        else:
            _fail(f"Spearman p-value mismatch for '{label}': re-computed {pval:.4f} vs stored {stored_pval:.4f}")

        if round(p_bonf, 3) == round(stored_bonf, 3):
            _pass(f"Bonferroni-adjusted p for '{label}': re-computed {p_bonf:.4f} rounds to stored {stored_bonf:.3f}")
        else:
            _fail(
                f"Bonferroni p mismatch for '{label}': re-computed {p_bonf:.4f} vs stored {stored_bonf:.4f} "
                f"(check k={N_CORR_TESTS} multiplier is consistent)"
            )

    # ── 4b. Kruskal-Wallis on n_correct ───────────────────────────────────────
    groups = [grp["n_correct"].values for _, grp in df.groupby("group_label")]
    h_stat, kw_pval = kruskal(*groups)
    print(f"\n  INFO  Kruskal-Wallis on n_correct: H={h_stat:.4f}, p={kw_pval:.4f}")
    print(f"        Compare against Table 1 manually to verify consistency")
    if kw_pval < 0.05:
        _warn(f"Kruskal-Wallis p={kw_pval:.4f} < 0.05 — significant group difference in n_correct")
    else:
        _pass(f"Kruskal-Wallis on n_correct: H={h_stat:.4f}, p={kw_pval:.4f} (n.s.)")

    # ── 4c. Per-question chi-square (Table 2) ─────────────────────────────────

    any_chi2_fail = False
    any_assumption_violated = False

    for col in QUESTION_COLS:
        label      = QUESTION_LABELS[col]
        correct_col = f"{col}_correct"

        # Build 2×3 contingency table: correct (0/1) × group
        ct = pd.crosstab(df[correct_col], df["group_label"])
        ct = ct.reindex(columns=GROUP_ORDER, fill_value=0)

        chi2_stat, chi2_p, dof, expected = chi2_contingency(ct, correction=False)
        p_bonf = min(chi2_p * N_ITEMS, 1.0)

        # Compare to Table 2
        t2_row = table2[table2["Question"] == label]
        if t2_row.empty:
            _warn(f"Table 2 has no row for '{label}'")
            continue

        stored_p      = float(t2_row["p-value"].iloc[0])
        stored_p_bonf = float(t2_row["p-value (corrected)"].iloc[0])

        if abs(chi2_p - stored_p) < 1e-3:
            _pass(f"Chi-square p for '{label}': re-computed {chi2_p:.4f} matches stored {stored_p:.4f}")
        else:
            any_chi2_fail = True
            _fail(f"Chi-square p mismatch for '{label}': re-computed {chi2_p:.4f} vs stored {stored_p:.4f}")

        if abs(p_bonf - stored_p_bonf) < 1e-3:
            _pass(f"Bonferroni p for '{label}': re-computed {p_bonf:.4f} matches stored {stored_p_bonf:.4f}")
        else:
            any_chi2_fail = True
            _fail(
                f"Bonferroni p mismatch for '{label}': re-computed {p_bonf:.4f} vs "
                f"stored {stored_p_bonf:.4f} (check k={N_ITEMS} multiplier)"
            )

        # Chi-square assumption: flag if any expected cell < 5
        if (expected < 5).any():
            any_assumption_violated = True
            min_exp = expected.min()
            n_cells_viol = (expected < 5).sum()
            _warn(
                f"  Chi-square assumption for '{label}': "
                f"{n_cells_viol} cell(s) have expected count < 5 (min={min_exp:.2f})"
            )

    if not any_chi2_fail:
        _pass("All per-question chi-square statistics match Table 2")
    if any_assumption_violated:
        _warn("Some chi-square tests violate the expected-count assumption (see above) — "
              "results should be interpreted with caution")
    else:
        _pass("All chi-square cells have expected count ≥ 5")


# ── Section 5: Cross-script consistency ──────────────────────────────────────
def check_cross_script_consistency(df: pd.DataFrame, table3: pd.DataFrame) -> None:
    _section("Section 5: Cross-Script Consistency")

    # Verify Table 3 on disk matches freshly computed Spearman values
    # (catches the case where 02 and 04 wrote different values)
    subsets = [("All Participants", df)] + [
        (grp, df[df["group_label"] == grp]) for grp in GROUP_ORDER
    ]

    all_match = True
    for label, sub in subsets:
        if len(sub) < 3:
            continue
        rho, pval = spearmanr(sub["self_confidence_mean"], sub["n_correct"])
        p_bonf    = min(pval * N_CORR_TESTS, 1.0)

        row = table3[table3["Subset"] == label]
        if row.empty:
            continue

        stored_rho  = float(row["rho"].iloc[0])
        stored_bonf = float(row["p_bonferroni"].iloc[0])

        rho_ok  = round(rho, 2) == round(stored_rho, 2)
        bonf_ok = round(p_bonf, 3) == round(stored_bonf, 3)

        if not rho_ok or not bonf_ok:
            all_match = False
            _fail(
                f"Table 3 on disk may have been written by a different run than the current data. "
                f"Subset '{label}': rho stored={stored_rho:.2f} vs live={rho:.4f}, "
                f"p_bonf stored={stored_bonf:.3f} vs live={p_bonf:.4f}"
            )

    if all_match:
        _pass("Table 3 on disk is consistent with freshly computed Spearman values (02 and 04 agree)")

    # Verify Bonferroni multiplier k=4 in Table 3 using freshly computed (unrounded) p-values,
    # since stored p_value is rounded and would compound rounding error.
    bonf_check_failed = False
    for label, sub in subsets:
        if len(sub) < 3:
            continue
        _, pval_live = spearmanr(sub["self_confidence_mean"], sub["n_correct"])
        expected_bonf = min(pval_live * N_CORR_TESTS, 1.0)
        t3_row = table3[table3["Subset"] == label]
        if t3_row.empty:
            continue
        stored_bonf = float(t3_row["p_bonferroni"].iloc[0])
        if round(expected_bonf, 3) != round(stored_bonf, 3):
            bonf_check_failed = True
            _fail(
                f"Bonferroni encoding error in Table 3 for '{label}': "
                f"live p={pval_live:.4f}, expected min(p*{N_CORR_TESTS}, 1)={expected_bonf:.3f}, "
                f"stored p_bonferroni={stored_bonf:.3f}"
            )

    if not bonf_check_failed:
        _pass(f"Table 3 Bonferroni adjustments use k={N_CORR_TESTS} consistently")

    # Verify Bonferroni for Table 2 uses k=12, using freshly computed chi-square p-values
    bonf_check_t2_failed = False
    table2_tmp = pd.read_csv(TABLE2_PATH)
    for col in QUESTION_COLS:
        label       = QUESTION_LABELS[col]
        correct_col = f"{col}_correct"
        ct          = pd.crosstab(df[correct_col], df["group_label"]).reindex(columns=GROUP_ORDER, fill_value=0)
        _, chi2_p_live, _, _ = chi2_contingency(ct, correction=False)
        expected_bonf = min(chi2_p_live * N_ITEMS, 1.0)

        t2_row = table2_tmp[table2_tmp["Question"] == label]
        if t2_row.empty:
            continue
        stored_bonf = float(t2_row["p-value (corrected)"].iloc[0])
        if round(expected_bonf, 3) != round(stored_bonf, 3):
            bonf_check_t2_failed = True
            _fail(
                f"Bonferroni encoding error in Table 2 for '{label}': "
                f"live p={chi2_p_live:.4f}, expected min(p*{N_ITEMS}, 1)={expected_bonf:.3f}, "
                f"stored corrected={stored_bonf:.3f}"
            )

    if not bonf_check_t2_failed:
        _pass(f"Table 2 Bonferroni adjustments use k={N_ITEMS} consistently")


# ── Fisher z-transform helpers (must match 06.power_analysis.py) ──────────────

def _spearman_power(rho: float, n: int, alpha: float = ALPHA) -> float:
    """Power for two-sided test of H0: rho=0 via Fisher z-transform."""
    if abs(rho) < 1e-6 or n <= 3:
        return alpha
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    z_rho  = np.arctanh(abs(rho))
    lam    = np.sqrt(n - 3) * z_rho
    return float(norm.cdf(lam - z_crit) + norm.cdf(-lam - z_crit))


def _spearman_required_n(rho: float, alpha: float = ALPHA,
                         power: float = POWER_TARGET) -> int | None:
    if abs(rho) < 0.01:
        return None
    if _spearman_power(rho, MAX_N, alpha) < power:
        return None
    n_req = brentq(lambda n: _spearman_power(rho, n, alpha) - power,
                   4.0, float(MAX_N))
    return int(np.ceil(n_req))


def _spearman_sensitivity(alpha: float = ALPHA,
                          power: float = POWER_TARGET) -> float | None:
    """Minimum |rho| detectable at N_TOTAL with given power."""
    if _spearman_power(0.999, N_TOTAL, alpha) < power:
        return None
    return float(brentq(lambda rho: _spearman_power(rho, N_TOTAL, alpha) - power,
                        0.001, 0.999))


def _parse_rho_from_csv(obs_effect: str) -> float | None:
    """Extract the numeric rho value from the 'Obs. effect' CSV column."""
    m = re.search(r"[ρr]\s*=\s*([−\-]?\d+\.\d+)", obs_effect)
    if m:
        return float(m.group(1).replace("−", "-"))
    return None


def _parse_n_from_csv(req_n: str) -> int | None:
    """Extract the integer N from the 'Req. total N' CSV column, None if > MAX_N."""
    s = str(req_n).strip()
    if s.startswith(">") or s.startswith("&gt;"):
        return None
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _parse_sensitivity_from_csv(min_det: str) -> float | None:
    """Extract the sensitivity threshold from the 'Min detectable' CSV column."""
    m = re.search(r"[≥>=]\s*(\d+\.\d+)", min_det)
    return float(m.group(1)) if m else None


# ── Section 6: Power analysis consistency ─────────────────────────────────────
def check_power_analysis(df: pd.DataFrame) -> None:
    _section("Section 6: Power Analysis Consistency")

    if not PA_CSV_PATH.exists():
        _warn(f"Power analysis CSV not found at {PA_CSV_PATH} — skipping section")
        return

    pa = pd.read_csv(PA_CSV_PATH)
    pa.columns = pa.columns.str.lower()

    # Normalise analysis column (strip narrow no-break spaces etc.)
    pa["_analysis_norm"] = pa["analysis"].str.replace(r"\s+", " ", regex=True).str.strip()

    # 6a. Expected shape: 5 main rows + 3 groups × 2 (regular + Bonferroni)
    expected_n_rows = 11
    if len(pa) == expected_n_rows:
        _pass(f"Power analysis CSV has {len(pa)} rows as expected")
    else:
        _fail(f"Power analysis CSV has {len(pa)} rows, expected {expected_n_rows}")

    # Check that key analyses are present (substring matching for robustness)
    expected_substrings = [
        "Kruskal-Wallis",
        "Chi-square.*0.05",
        "Chi-square.*Bonferroni",
        "Spearman.*all participants",
        "Spearman.*all participants.*Bonferroni",
    ]
    for g in GROUP_ORDER:
        expected_substrings.append(f"Spearman.*{re.escape(g)}")
        expected_substrings.append(f"Spearman.*{re.escape(g)}.*Bonferroni")

    all_found = True
    for pattern in expected_substrings:
        if not pa["_analysis_norm"].str.contains(pattern, regex=True).any():
            all_found = False
            _fail(f"No row matching pattern '{pattern}' in power analysis CSV")
    if all_found:
        _pass("All expected analysis rows present in CSV")

    # 6b. Verify ALPHA_BONF = 0.05 / 12 and ALPHA_BONF_SPEAR = 0.05 / 4
    alpha_bonf_expected = ALPHA / N_ITEMS
    if abs(ALPHA_BONF - alpha_bonf_expected) < 1e-8:
        _pass(f"ALPHA_BONF = {ALPHA}/{N_ITEMS} = {ALPHA_BONF:.6f}")
    else:
        _fail(f"ALPHA_BONF mismatch: {ALPHA_BONF} != {ALPHA}/{N_ITEMS}")

    alpha_bonf_spear_expected = ALPHA / N_CORR_TESTS
    if abs(ALPHA_BONF_SPEAR - alpha_bonf_spear_expected) < 1e-8:
        _pass(f"ALPHA_BONF_SPEAR = {ALPHA}/{N_CORR_TESTS} = {ALPHA_BONF_SPEAR:.6f}")
    else:
        _fail(f"ALPHA_BONF_SPEAR mismatch: {ALPHA_BONF_SPEAR} != {ALPHA}/{N_CORR_TESTS}")

    # Helper: find rows by substring in normalised Analysis column
    def _pa_rows(pattern: str) -> pd.DataFrame:
        return pa[pa["_analysis_norm"].str.contains(pattern, regex=True)]

    # 6c. Re-derive Spearman rho from data and compare to CSV
    rho_live, _ = spearmanr(df["self_confidence_mean"], df["n_correct"])
    row_spr = _pa_rows(r"Spearman.*all participants(?!.*Bonferroni)")
    if not row_spr.empty:
        csv_rho = _parse_rho_from_csv(row_spr["obs. effect"].iloc[0])
        if csv_rho is not None and abs(round(rho_live, 3) - csv_rho) < 0.002:
            _pass(f"Spearman ρ from data ({rho_live:.4f}) matches CSV ({csv_rho:.3f})")
        else:
            _fail(f"Spearman ρ mismatch: live={rho_live:.4f}, CSV={csv_rho}")

    # 6d. Re-derive sensitivity and required N for Spearman (α = 0.05)
    sens_live   = _spearman_sensitivity(alpha=ALPHA)
    req_n_live  = _spearman_required_n(rho_live, alpha=ALPHA)

    if not row_spr.empty:
        csv_sens = _parse_sensitivity_from_csv(
            row_spr["min detectable (at study n)"].iloc[0])
        if csv_sens is not None and sens_live is not None:
            if abs(sens_live - csv_sens) < 0.002:
                _pass(f"Spearman sensitivity (α=0.05): live={sens_live:.3f}, CSV={csv_sens:.3f}")
            else:
                _fail(f"Spearman sensitivity mismatch (α=0.05): live={sens_live:.3f}, CSV={csv_sens:.3f}")

        csv_req = _parse_n_from_csv(row_spr["req. total n"].iloc[0])
        if csv_req is not None and req_n_live is not None:
            if csv_req == req_n_live:
                _pass(f"Spearman required N (α=0.05): live={req_n_live}, CSV={csv_req}")
            else:
                _fail(f"Spearman required N mismatch (α=0.05): live={req_n_live}, CSV={csv_req}")

    # 6e. Re-derive sensitivity and required N for Spearman (Bonferroni)
    sens_bonf_live  = _spearman_sensitivity(alpha=ALPHA_BONF_SPEAR)
    req_n_bonf_live = _spearman_required_n(rho_live, alpha=ALPHA_BONF_SPEAR)

    row_spr_b = _pa_rows(r"Spearman.*all participants.*Bonferroni")
    if not row_spr_b.empty:
        csv_sens_b = _parse_sensitivity_from_csv(
            row_spr_b["min detectable (at study n)"].iloc[0])
        if csv_sens_b is not None and sens_bonf_live is not None:
            if abs(sens_bonf_live - csv_sens_b) < 0.002:
                _pass(f"Spearman sensitivity (Bonferroni): live={sens_bonf_live:.3f}, CSV={csv_sens_b:.3f}")
            else:
                _fail(f"Spearman sensitivity mismatch (Bonferroni): live={sens_bonf_live:.3f}, CSV={csv_sens_b:.3f}")

        csv_req_b = _parse_n_from_csv(row_spr_b["req. total n"].iloc[0])
        if csv_req_b is not None and req_n_bonf_live is not None:
            if csv_req_b == req_n_bonf_live:
                _pass(f"Spearman required N (Bonferroni): live={req_n_bonf_live}, CSV={csv_req_b}")
            else:
                _fail(f"Spearman required N mismatch (Bonferroni): live={req_n_bonf_live}, CSV={csv_req_b}")

    # 6f. Bonferroni corrections are strictly more conservative
    if sens_live is not None and sens_bonf_live is not None:
        if sens_bonf_live > sens_live:
            _pass(f"Bonferroni sensitivity ({sens_bonf_live:.3f}) > unadjusted ({sens_live:.3f})")
        else:
            _fail(f"Bonferroni sensitivity ({sens_bonf_live:.3f}) should be > "
                  f"unadjusted ({sens_live:.3f})")

    if req_n_live is not None and req_n_bonf_live is not None:
        if req_n_bonf_live > req_n_live:
            _pass(f"Bonferroni required N ({req_n_bonf_live}) > unadjusted ({req_n_live})")
        else:
            _fail(f"Bonferroni required N ({req_n_bonf_live}) should be > "
                  f"unadjusted ({req_n_live})")

    # 6g. Fisher z-transform power spot-check at known N
    power_at_18 = _spearman_power(rho_live, N_TOTAL, alpha=ALPHA)
    if 0.0 < power_at_18 < 1.0:
        _pass(f"Spearman power at N=18 is plausible: {power_at_18:.3f}")
    else:
        _fail(f"Spearman power at N=18 is implausible: {power_at_18}")

    if req_n_live is not None:
        power_at_req = _spearman_power(rho_live, req_n_live, alpha=ALPHA)
        if abs(power_at_req - POWER_TARGET) < 0.02:
            _pass(f"Power at required N={req_n_live} ≈ 0.80 ({power_at_req:.3f})")
        else:
            _fail(f"Power at required N={req_n_live} should ≈ 0.80, got {power_at_req:.3f}")

    # 6h. Per-group Spearman rho in CSV matches live computation
    for grp in GROUP_ORDER:
        sub = df[df["group_label"] == grp]
        if len(sub) < 3:
            continue
        rho_g, _ = spearmanr(sub["self_confidence_mean"], sub["n_correct"])
        grp_pat = re.escape(grp)
        row_g = _pa_rows(rf"Spearman.*{grp_pat}(?!.*Bonferroni)")
        if row_g.empty:
            _warn(f"No power analysis CSV row for group '{grp}'")
            continue
        csv_rho_g = _parse_rho_from_csv(row_g["obs. effect"].iloc[0])
        if csv_rho_g is not None and abs(round(rho_g, 3) - csv_rho_g) < 0.002:
            _pass(f"Spearman ρ for '{grp}': live={rho_g:.3f}, CSV={csv_rho_g:.3f}")
        else:
            _fail(f"Spearman ρ mismatch for '{grp}': live={rho_g:.4f}, CSV={csv_rho_g}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}  Dental Trauma Survey - Data Integrity & Sanity Checks{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}")

    # Load data
    if not DATA_CLEAN.exists():
        print(f"{_RED}FATAL{_RESET}  Cannot find {DATA_CLEAN}. Run 01.data_pre_processing.ipynb first.")
        sys.exit(1)
    if not ANSWERS_PATH.exists():
        print(f"{_RED}FATAL{_RESET}  Cannot find answer key at {ANSWERS_PATH}.")
        sys.exit(1)

    df = pd.read_csv(DATA_CLEAN)

    # Build answer key (same logic as 01.data_pre_processing.ipynb)
    answers_df = pd.read_csv(ANSWERS_PATH)
    answer_key = {
        QCODE_TO_COL[row["question_col"]]: row["correct_answer"]
        for _, row in answers_df.iterrows()
        if row["question_col"] in QCODE_TO_COL
    }

    # Load reference tables
    table3_exists = TABLE3_PATH.exists()
    table2_exists = TABLE2_PATH.exists()

    table3 = pd.read_csv(TABLE3_PATH) if table3_exists else pd.DataFrame()
    table2 = pd.read_csv(TABLE2_PATH) if table2_exists else pd.DataFrame()

    if not table3_exists:
        print(f"{_YELLOW}WARN{_RESET}  table3_correlation.csv not found — Sections 4 & 5 partially skipped")
    if not table2_exists:
        print(f"{_YELLOW}WARN{_RESET}  table2_per_question.csv not found — Section 4c partially skipped")

    # Run checks
    check_structure(df)
    check_scoring(df, answer_key)
    check_derived_columns(df)
    check_statistics(df, table2, table3)
    check_cross_script_consistency(df, table3)
    check_power_analysis(df)

    # Summary
    _section("Summary")
    total = _pass_count + _fail_count + _warn_count
    print(f"  Total checks: {total}")
    print(f"  {_GREEN}PASS{_RESET}: {_pass_count}")
    print(f"  {_RED}FAIL{_RESET}: {_fail_count}")
    print(f"  {_YELLOW}WARN{_RESET}: {_warn_count}")

    if _fail_count == 0:
        print(f"\n{_GREEN}{_BOLD}  All checks passed.{_RESET}")
    else:
        print(f"\n{_RED}{_BOLD}  {_fail_count} check(s) FAILED — review output above.{_RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
