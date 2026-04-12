"""
anonymise_raw_survey.py

Strips personally identifiable and geographically sensitive columns from the
raw Qualtrics export and writes a de-identified copy to data/raw/anonymised/.

Columns removed
---------------
- IPAddress              — direct identifier
- LocationLatitude       — precise geolocation
- LocationLongitude      — precise geolocation
- StartDate              — timestamped with timezone
- EndDate                — timestamped with timezone
- RecordedDate           — timestamped with timezone
- RecipientLastName      — PII
- RecipientFirstName     — PII
- RecipientEmail         — PII
- ExternalReference      — PII link field
- ResponseId             — Qualtrics-internal identifier
- Status                 — always "IP Address"; no analytic value
- DistributionChannel    — always "anonymous"; no analytic value
- UserLanguage           — always "EN" in this dataset; no analytic value

Columns kept
------------
Everything else: duration, progress, completion flag, all survey responses,
score, randomisation group, and block progress.

Usage
-----
    python scripts/00.anonymise_raw_survey.py

Pipeline position
-----------------
**Step 0 of the analysis pipeline.**  Run once before any other script or
notebook.  All subsequent notebooks read from the anonymised output written
by this script (data/raw/anonymised/dental_trauma_survey_responses.csv).
"""

import csv
import sys
from pathlib import Path

import pandas as pd

RAW_FILE = Path("data/raw/Medical Provider Dental Trauma Assessment_April 9, 2026_20.07.csv")
OUT_DIR  = Path("data/raw/anonymised")
OUT_FILE = OUT_DIR / "dental_trauma_survey_responses.csv"

PII_COLUMNS = [
    "StartDate",
    "EndDate",
    "Status",
    "IPAddress",
    "RecordedDate",
    "ResponseId",
    "RecipientLastName",
    "RecipientFirstName",
    "RecipientEmail",
    "ExternalReference",
    "LocationLatitude",
    "LocationLongitude",
    "DistributionChannel",
    "UserLanguage",
]

# Columns whose non-null counts and unique values must be identical between
# the raw and anonymised files (spot-check that data was not shifted or lost).
INVARIANT_COLUMNS = [
    "Duration (in seconds)",
    "Progress",
    "Finished",
    "SC0",
    "random",
]


def load_qualtrics(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a Qualtrics CSV that has three header rows.

    Row 0 — short variable names (used as column headers)
    Row 1 — full question text
    Row 2 — Qualtrics import IDs

    Returns the data DataFrame and the two metadata rows as a separate
    DataFrame so they can be re-attached to the anonymised file.
    """
    header    = pd.read_csv(path, nrows=1, header=None)
    meta_rows = pd.read_csv(path, skiprows=1, nrows=2, header=None)
    meta_rows.columns = header.iloc[0]

    df_raw = pd.read_csv(path, header=0, skiprows=[1, 2], low_memory=False)
    return df_raw, meta_rows


def drop_pii(
    df: pd.DataFrame, meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove all PII/redundant columns from the data and metadata rows."""
    cols_to_drop = [c for c in PII_COLUMNS if c in df.columns]
    return (
        df.drop(columns=cols_to_drop),
        meta.drop(columns=cols_to_drop, errors="ignore"),
    )


def save(df_clean: pd.DataFrame, meta_clean: pd.DataFrame, out_path: Path) -> None:
    """Write the anonymised dataset preserving the two Qualtrics metadata rows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = df_clean.columns.tolist()

    # meta has duplicate column names (Qualtrics branching), so extract
    # the metadata rows by position rather than by label.
    question_row = meta_clean.iloc[0].values.tolist()
    import_row   = meta_clean.iloc[1].values.tolist()

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        writer.writerow(question_row)
        writer.writerow(import_row)
        for _, row in df_clean.iterrows():
            writer.writerow(row.tolist())


def verify_integrity(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    out_path: Path,
) -> bool:
    """
    Run a suite of checks to confirm that anonymisation did not corrupt data.

    Checks performed
    ----------------
    1. Row count preserved between raw and clean in-memory DataFrames.
    2. PII columns are absent from the clean DataFrame.
    3. All expected PII columns were actually present in the raw file.
    4. Kept column count equals raw minus number of PII columns removed.
    5. Invariant columns have identical non-null counts in raw vs. clean.
    6. Invariant columns have identical unique value sets in raw vs. clean.
    7. Output file exists and is non-empty.
    8. Output file round-trips correctly: row count and column count match
       the clean DataFrame after re-reading (skipping the two metadata rows).

    Returns True when all checks pass; prints failures and returns False otherwise.
    """
    failures = []

    # 1. Row count
    if len(df_raw) != len(df_clean):
        failures.append(
            f"Row count changed: {len(df_raw)} raw -> {len(df_clean)} clean"
        )

    # 2. No PII columns remain
    pii_still_present = [c for c in PII_COLUMNS if c in df_clean.columns]
    if pii_still_present:
        failures.append(f"PII columns still present: {pii_still_present}")

    # 3. All declared PII columns were found in the raw file
    pii_not_in_raw = [c for c in PII_COLUMNS if c not in df_raw.columns]
    if pii_not_in_raw:
        failures.append(
            f"Declared PII columns missing from raw file (typo?): {pii_not_in_raw}"
        )

    # 4. Column count arithmetic
    expected_cols = df_raw.shape[1] - len([c for c in PII_COLUMNS if c in df_raw.columns])
    if df_clean.shape[1] != expected_cols:
        failures.append(
            f"Column count mismatch: expected {expected_cols}, got {df_clean.shape[1]}"
        )

    # 5 & 6. Invariant column spot-checks
    for col in INVARIANT_COLUMNS:
        if col not in df_raw.columns:
            failures.append(f"Invariant column '{col}' missing from raw data")
            continue
        if col not in df_clean.columns:
            failures.append(f"Invariant column '{col}' missing from clean data")
            continue

        raw_nonnull   = df_raw[col].notna().sum()
        clean_nonnull = df_clean[col].notna().sum()
        if raw_nonnull != clean_nonnull:
            failures.append(
                f"Non-null count changed for '{col}': {raw_nonnull} -> {clean_nonnull}"
            )

        raw_vals   = set(df_raw[col].dropna().astype(str).unique())
        clean_vals = set(df_clean[col].dropna().astype(str).unique())
        if raw_vals != clean_vals:
            failures.append(
                f"Unique values changed for '{col}':\n"
                f"  raw:   {sorted(raw_vals)}\n"
                f"  clean: {sorted(clean_vals)}"
            )

    # 7. Output file exists and is non-empty
    if not out_path.exists():
        failures.append(f"Output file does not exist: {out_path}")
    elif out_path.stat().st_size == 0:
        failures.append(f"Output file is empty: {out_path}")
    else:
        # 8. Round-trip check
        df_roundtrip = pd.read_csv(out_path, header=0, skiprows=[1, 2], low_memory=False)

        if len(df_roundtrip) != len(df_clean):
            failures.append(
                f"Round-trip row count mismatch: "
                f"wrote {len(df_clean)}, read back {len(df_roundtrip)}"
            )

        if df_roundtrip.shape[1] != df_clean.shape[1]:
            failures.append(
                f"Round-trip column count mismatch: "
                f"wrote {df_clean.shape[1]}, read back {df_roundtrip.shape[1]}"
            )

        rt_pii = [c for c in PII_COLUMNS if c in df_roundtrip.columns]
        if rt_pii:
            failures.append(f"PII columns present in saved file: {rt_pii}")

    if failures:
        print("\n[INTEGRITY CHECKS FAILED]")
        for msg in failures:
            print(f"  FAIL: {msg}")
        return False

    print("[INTEGRITY CHECKS PASSED] All checks OK.")
    return True


if __name__ == "__main__":
    # ── Pipeline: Step 0 ─────────────────────────────────────────────────────
    # Read raw Qualtrics export → drop PII columns → write anonymised CSV.
    # Integrity checks verify no data rows were lost or corrupted.
    print(f"Loading: {RAW_FILE}")
    df_raw, meta = load_qualtrics(RAW_FILE)
    print(f"  {len(df_raw)} responses, {df_raw.shape[1]} columns")

    df_clean, meta_clean = drop_pii(df_raw, meta)
    removed = df_raw.shape[1] - df_clean.shape[1]
    print(f"  Removed {removed} PII/redundant columns -> {df_clean.shape[1]} remaining")

    save(df_clean, meta_clean, OUT_FILE)
    print(f"  Saved anonymised file to: {OUT_FILE}")

    print()
    passed = verify_integrity(df_raw, df_clean, OUT_FILE)
    if not passed:
        sys.exit(1)
