
# ============================================================
# threshold_class_analysis.py
#
# Phase 2.5 - Class-wise Threshold Analysis
#
# Input:
#   1) verifier_test_results.csv
#   2) threshold_optimizer_top.csv
#
# Target classes:
#   01_Good
#   02_Out_of_Focus
#   03_Tracking_Error
#
# IMPORTANT:
#   - CNN prediction remains authoritative.
#   - Verifier does NOT change cnn_class.
#   - Existing "verdict" column is NOT used as ground truth.
#   - Verdict is recalculated from candidate thresholds.
#
# Output:
#   threshold_class_analysis.csv
#   threshold_class_summary.csv
#   threshold_class_analysis_report.txt
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

# BASE_DIR = Path(
#     r"D:\Internship\Test_Rule\test_rule_new_v2"
# )

# ANALYSIS_DIR = BASE_DIR / "verifier_analysis_v2"

# # ใช้โฟลเดอร์ล่าสุดที่ Optimizer สร้างผลลัพธ์
# INPUT_DIR = ANALYSIS_DIR / "verifier_test_V2"

# RESULTS_CSV = r"D:\Internship\Test_Rule\test_rule_new_v2\verifier_analysis_v2\verifier_test\verifier_test_results.csv"
# TOP_CSV = r"D:\Internship\Test_Rule\test_rule_new_v2\verifier_analysis_v2\verifier_test_V2\threshold_optimizer_top.csv"

# OUTPUT_DIR = INPUT_DIR

# TARGET_CLASSES = [
#     "01_Good",
#     "02_Out_of_Focus",
#     "03_Tracking_Error",
# ]

BASE_DIR = Path(
    r"D:\Internship\Test_Rule\test_rule_new_v2"
)

ANALYSIS_DIR = BASE_DIR / "verifier_analysis_v2"

# ไฟล์ผลการทดสอบ Verifier อยู่ในโฟลเดอร์ verifier_test
RESULTS_CSV = ANALYSIS_DIR / "verifier_test" / "verifier_test_results.csv"

# ไฟล์ผล Optimizer อยู่ในโฟลเดอร์ verifier_test_V2
TOP_CSV = ANALYSIS_DIR / "verifier_test_V2" / "threshold_optimizer_top.csv"

# ให้ผลการวิเคราะห์ใหม่ถูกบันทึกในโฟลเดอร์ V2
OUTPUT_DIR = ANALYSIS_DIR / "verifier_test_V2"

TARGET_CLASSES = [
    "01_Good",
    "02_Out_of_Focus",
    "03_Tracking_Error",
]

# จำนวน threshold candidates จาก threshold_optimizer_top.csv
TOP_N = 10

# Threshold ปัจจุบันของ Verifier
CURRENT_THRESHOLD = {
    "weak_evidence": 0.40,
    "supported_evidence": 0.60,
    "supported_margin": 0.15,
    "ambiguous_margin": 0.08,
}


# ============================================================
# UTILS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def load_csv(path: Path, name: str):
    if not path.exists():
        raise SystemExit(
            f"\n[ERROR] ไม่พบไฟล์ {name}:\n{path}\n"
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(
            f"\n[ERROR] อ่านไฟล์ {name} ไม่สำเร็จ:\n"
            f"{path}\n"
            f"Error: {repr(exc)}\n"
        ) from exc

    print(f"[OK] Loaded {name}: {len(df):,} rows")
    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_results(df: pd.DataFrame):

    required = [
        "true_class",
        "cnn_class",
        "cnn_confidence",
        "cnn_class_evidence",
        "best_rival_evidence",
        "pairwise_margin",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise SystemExit(
            "\n[ERROR] verifier_test_results.csv "
            "ขาด columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    work = df.copy()

    # สนใจเฉพาะภาพที่มี true class อยู่ใน target
    work = work[
        work["true_class"].isin(TARGET_CLASSES)
    ].copy()

    # และ CNN ต้องทำนายอยู่ใน verifier scope
    work = work[
        work["cnn_class"].isin(TARGET_CLASSES)
    ].copy()

    numeric_cols = [
        "cnn_confidence",
        "cnn_class_evidence",
        "best_rival_evidence",
        "pairwise_margin",
    ]

    for col in numeric_cols:
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce"
        )

    work = work.dropna(
        subset=[
            "true_class",
            "cnn_class",
            "cnn_class_evidence",
            "best_rival_evidence",
            "pairwise_margin",
        ]
    ).copy()

    # CNN correctness
    work["cnn_correct"] = (
        work["true_class"] == work["cnn_class"]
    )

    return work


# ============================================================
# VERDICT FROM THRESHOLDS
# ============================================================

def calculate_verdict(
    evidence,
    margin,
    weak_evidence,
    supported_evidence,
    supported_margin,
    ambiguous_margin,
):
    """
    Verdict policy เดียวกับ Threshold Optimizer

    1. SUPPORTED
       evidence >= supported_evidence
       AND
       margin >= supported_margin

    2. AMBIGUOUS
       margin <= ambiguous_margin

    3. WEAK
       evidence >= weak_evidence

    4. otherwise AMBIGUOUS
    """

    if (
        evidence >= supported_evidence
        and margin >= supported_margin
    ):
        return "SUPPORTED"

    if margin <= ambiguous_margin:
        return "AMBIGUOUS"

    if evidence >= weak_evidence:
        return "WEAK"

    return "AMBIGUOUS"


# ============================================================
# APPLY THRESHOLD
# ============================================================

def apply_threshold(df, threshold):

    work = df.copy()

    weak_evidence = threshold["weak_evidence"]
    supported_evidence = threshold["supported_evidence"]
    supported_margin = threshold["supported_margin"]
    ambiguous_margin = threshold["ambiguous_margin"]

    work["analysis_verdict"] = [
        calculate_verdict(
            evidence=row["cnn_class_evidence"],
            margin=row["pairwise_margin"],
            weak_evidence=weak_evidence,
            supported_evidence=supported_evidence,
            supported_margin=supported_margin,
            ambiguous_margin=ambiguous_margin,
        )
        for _, row in work.iterrows()
    ]

    return work


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_overall_metrics(
    df,
    threshold,
    candidate_rank,
):

    work = apply_threshold(
        df,
        threshold
    )

    total = len(work)

    supported = work[
        work["analysis_verdict"] == "SUPPORTED"
    ]

    weak = work[
        work["analysis_verdict"] == "WEAK"
    ]

    ambiguous = work[
        work["analysis_verdict"] == "AMBIGUOUS"
    ]

    cnn_correct = work[
        work["cnn_correct"]
    ]

    supported_correct = supported[
        supported["cnn_correct"]
    ]

    false_supported = supported[
        ~supported["cnn_correct"]
    ]

    supported_count = len(supported)
    weak_count = len(weak)
    ambiguous_count = len(ambiguous)

    if supported_count > 0:
        supported_precision = (
            len(supported_correct)
            / supported_count
        )
    else:
        supported_precision = np.nan

    if len(cnn_correct) > 0:
        supported_coverage = (
            len(supported_correct)
            / len(cnn_correct)
        )
    else:
        supported_coverage = np.nan

    if total > 0:
        supported_rate = (
            supported_count / total
        )
        weak_rate = (
            weak_count / total
        )
        ambiguous_rate = (
            ambiguous_count / total
        )
    else:
        supported_rate = np.nan
        weak_rate = np.nan
        ambiguous_rate = np.nan

    return {
        "candidate_rank": candidate_rank,

        "weak_evidence": threshold["weak_evidence"],
        "supported_evidence": threshold["supported_evidence"],
        "supported_margin": threshold["supported_margin"],
        "ambiguous_margin": threshold["ambiguous_margin"],

        "total": total,

        "supported_count": supported_count,
        "weak_count": weak_count,
        "ambiguous_count": ambiguous_count,

        "supported_rate": supported_rate,
        "weak_rate": weak_rate,
        "ambiguous_rate": ambiguous_rate,

        "cnn_correct_count": len(cnn_correct),

        "supported_correct": len(supported_correct),
        "false_supported": len(false_supported),

        "supported_precision": supported_precision,
        "supported_coverage": supported_coverage,

        "false_supported_rate": (
            len(false_supported) / total
            if total > 0 else np.nan
        ),
    }


# ============================================================
# CLASS-WISE METRICS
# ============================================================

def calculate_class_metrics(
    df,
    threshold,
    candidate_rank,
):

    work = apply_threshold(
        df,
        threshold
    )

    rows = []

    for true_class in TARGET_CLASSES:

        class_df = work[
            work["true_class"] == true_class
        ].copy()

        total = len(class_df)

        cnn_correct = class_df[
            class_df["cnn_correct"]
        ]

        cnn_wrong = class_df[
            ~class_df["cnn_correct"]
        ]

        supported = class_df[
            class_df["analysis_verdict"] == "SUPPORTED"
        ]

        weak = class_df[
            class_df["analysis_verdict"] == "WEAK"
        ]

        ambiguous = class_df[
            class_df["analysis_verdict"] == "AMBIGUOUS"
        ]

        supported_correct = supported[
            supported["cnn_correct"]
        ]

        false_supported = supported[
            ~supported["cnn_correct"]
        ]

        supported_count = len(supported)

        if supported_count > 0:
            supported_precision = (
                len(supported_correct)
                / supported_count
            )
        else:
            supported_precision = np.nan

        if len(cnn_correct) > 0:
            supported_coverage = (
                len(supported_correct)
                / len(cnn_correct)
            )
        else:
            supported_coverage = np.nan

        rows.append({

            "candidate_rank": candidate_rank,

            "true_class": true_class,

            "weak_evidence": threshold["weak_evidence"],
            "supported_evidence": threshold["supported_evidence"],
            "supported_margin": threshold["supported_margin"],
            "ambiguous_margin": threshold["ambiguous_margin"],

            # ----------------------------
            # Counts
            # ----------------------------

            "total": total,

            "cnn_correct": len(cnn_correct),
            "cnn_wrong": len(cnn_wrong),

            "supported": supported_count,
            "weak": len(weak),
            "ambiguous": len(ambiguous),

            # ----------------------------
            # SUPPORTED
            # ----------------------------

            "supported_correct": len(
                supported_correct
            ),

            "false_supported": len(
                false_supported
            ),

            "supported_precision":
                supported_precision,

            "supported_coverage":
                supported_coverage,

            # ----------------------------
            # Rates
            # ----------------------------

            "supported_rate":
                supported_count / total
                if total > 0 else np.nan,

            "weak_rate":
                len(weak) / total
                if total > 0 else np.nan,

            "ambiguous_rate":
                len(ambiguous) / total
                if total > 0 else np.nan,

            "false_supported_rate":
                len(false_supported) / total
                if total > 0 else np.nan,

            # False SUPPORTED เฉพาะกรณี CNN ผิด
            "false_supported_among_cnn_wrong":
                (
                    len(false_supported)
                    / len(cnn_wrong)
                    if len(cnn_wrong) > 0
                    else 0.0
                ),

        })

    return pd.DataFrame(rows)


# ============================================================
# CNN-CLASS-WISE METRICS
# ============================================================

def calculate_cnn_class_metrics(
    df,
    threshold,
    candidate_rank,
):

    work = apply_threshold(
        df,
        threshold
    )

    rows = []

    for cnn_class in TARGET_CLASSES:

        class_df = work[
            work["cnn_class"] == cnn_class
        ].copy()

        total = len(class_df)

        cnn_correct = class_df[
            class_df["cnn_correct"]
        ]

        cnn_wrong = class_df[
            ~class_df["cnn_correct"]
        ]

        supported = class_df[
            class_df["analysis_verdict"] == "SUPPORTED"
        ]

        supported_correct = supported[
            supported["cnn_correct"]
        ]

        false_supported = supported[
            ~supported["cnn_correct"]
        ]

        rows.append({

            "candidate_rank": candidate_rank,

            "cnn_class": cnn_class,

            "total": total,

            "cnn_correct": len(cnn_correct),
            "cnn_wrong": len(cnn_wrong),

            "supported": len(supported),
            "supported_correct":
                len(supported_correct),

            "false_supported":
                len(false_supported),

            "supported_precision":
                (
                    len(supported_correct)
                    / len(supported)
                    if len(supported) > 0
                    else np.nan
                ),

            "supported_rate":
                (
                    len(supported)
                    / total
                    if total > 0
                    else np.nan
                ),

            "false_supported_rate":
                (
                    len(false_supported)
                    / total
                    if total > 0
                    else np.nan
                ),

        })

    return pd.DataFrame(rows)


# ============================================================
# LOAD TOP CANDIDATES
# ============================================================

def load_candidates(top_df):

    required = [
        "weak_evidence",
        "supported_evidence",
        "supported_margin",
        "ambiguous_margin",
    ]

    missing = [
        col
        for col in required
        if col not in top_df.columns
    ]

    if missing:
        raise SystemExit(
            "\n[ERROR] threshold_optimizer_top.csv "
            "ขาด columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    candidates = []

    # ----------------------------------------
    # Current threshold เป็น baseline
    # ----------------------------------------

    candidates.append({
        "rank": 0,
        **CURRENT_THRESHOLD,
    })

    # ----------------------------------------
    # Top N optimizer candidates
    # ----------------------------------------

    top_df = top_df.head(TOP_N).copy()

    for idx, row in top_df.iterrows():

        candidate = {
            "rank": int(
                row["rank"]
            )
            if "rank" in top_df.columns
            and pd.notna(row.get("rank"))
            else int(idx) + 1,

            "weak_evidence":
                safe_float(
                    row["weak_evidence"]
                ),

            "supported_evidence":
                safe_float(
                    row["supported_evidence"]
                ),

            "supported_margin":
                safe_float(
                    row["supported_margin"]
                ),

            "ambiguous_margin":
                safe_float(
                    row["ambiguous_margin"]
                ),
        }

        candidates.append(candidate)

    return candidates


# ============================================================
# REPORT
# ============================================================

def create_report(
    overall_df,
    class_df,
    cnn_class_df,
    output_path,
):

    lines = []

    lines.append(
        "=" * 78
    )
    lines.append(
        "THRESHOLD CLASS-WISE ANALYSIS REPORT"
    )
    lines.append(
        "=" * 78
    )

    lines.append("")
    lines.append(
        "Purpose:"
    )
    lines.append(
        "Evaluate optimizer threshold candidates "
        "separately for 01_Good, 02_Out_of_Focus, "
        "and 03_Tracking_Error."
    )

    lines.append("")
    lines.append(
        "Important:"
    )
    lines.append(
        "- CNN prediction remains authoritative."
    )
    lines.append(
        "- Verifier does not change cnn_class."
    )
    lines.append(
        "- Existing verdict column is not used as ground truth."
    )
    lines.append(
        "- SUPPORTED is considered correct only when CNN class "
        "matches true_class."
    )

    # ========================================================
    # OVERALL
    # ========================================================

    lines.append("")
    lines.append(
        "=" * 78
    )
    lines.append(
        "OVERALL CANDIDATE COMPARISON"
    )
    lines.append(
        "=" * 78
    )

    overall_sorted = overall_df.sort_values(
        by=[
            "false_supported",
            "supported_precision",
            "supported_coverage",
        ],
        ascending=[
            True,
            False,
            False,
        ],
        na_position="last",
    )

    for _, row in overall_sorted.iterrows():

        lines.append("")
        lines.append(
            f"Candidate rank : {int(row['candidate_rank'])}"
        )

        lines.append(
            "Threshold       : "
            f"weak={row['weak_evidence']:.2f}, "
            f"supported={row['supported_evidence']:.2f}, "
            f"supported_margin={row['supported_margin']:.2f}, "
            f"ambiguous_margin={row['ambiguous_margin']:.2f}"
        )

        lines.append(
            "Verdicts        : "
            f"SUPPORTED={int(row['supported_count'])}, "
            f"WEAK={int(row['weak_count'])}, "
            f"AMBIGUOUS={int(row['ambiguous_count'])}"
        )

        lines.append(
            "SUPPORTED       : "
            f"precision={row['supported_precision']:.3f}, "
            f"coverage={row['supported_coverage']:.3f}"
        )

        lines.append(
            "False SUPPORTED : "
            f"{int(row['false_supported'])} "
            f"({row['false_supported_rate']:.3f})"
        )

    # ========================================================
    # CLASS-WISE
    # ========================================================

    lines.append("")
    lines.append(
        "=" * 78
    )
    lines.append(
        "CLASS-WISE ANALYSIS"
    )
    lines.append(
        "=" * 78
    )

    for rank in class_df[
        "candidate_rank"
    ].drop_duplicates():

        candidate_rows = class_df[
            class_df["candidate_rank"] == rank
        ]

        if candidate_rows.empty:
            continue

        first = candidate_rows.iloc[0]

        lines.append("")
        lines.append(
            f"Candidate rank {int(rank)}"
        )

        lines.append(
            "Threshold       : "
            f"weak={first['weak_evidence']:.2f}, "
            f"supported={first['supported_evidence']:.2f}, "
            f"supported_margin={first['supported_margin']:.2f}, "
            f"ambiguous_margin={first['ambiguous_margin']:.2f}"
        )

        for _, row in candidate_rows.iterrows():

            precision = row["supported_precision"]
            coverage = row["supported_coverage"]

            precision_text = (
                f"{precision:.3f}"
                if pd.notna(precision)
                else "N/A"
            )

            coverage_text = (
                f"{coverage:.3f}"
                if pd.notna(coverage)
                else "N/A"
            )

            lines.append(
                f"  {row['true_class']:<20} "
                f"total={int(row['total']):3d} | "
                f"CNN_correct={int(row['cnn_correct']):3d} | "
                f"SUPPORTED={int(row['supported']):3d} | "
                f"WEAK={int(row['weak']):3d} | "
                f"AMB={int(row['ambiguous']):3d} | "
                f"false_SUP={int(row['false_supported']):2d} | "
                f"precision={precision_text} | "
                f"coverage={coverage_text}"
            )

    # ========================================================
    # BEST CANDIDATE HEURISTIC
    # ========================================================

    lines.append("")
    lines.append(
        "=" * 78
    )
    lines.append(
        "INTERPRETATION"
    )
    lines.append(
        "=" * 78
    )

    zero_false = overall_df[
        overall_df["false_supported"] == 0
    ].copy()

    if not zero_false.empty:

        best = zero_false.sort_values(
            by=[
                "supported_coverage",
                "supported_precision",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        ).iloc[0]

        lines.append("")
        lines.append(
            "Best zero-false-SUPPORTED candidate "
            "by coverage:"
        )

        lines.append(
            f"  rank               = "
            f"{int(best['candidate_rank'])}"
        )

        lines.append(
            f"  weak_evidence      = "
            f"{best['weak_evidence']:.2f}"
        )

        lines.append(
            f"  supported_evidence = "
            f"{best['supported_evidence']:.2f}"
        )

        lines.append(
            f"  supported_margin   = "
            f"{best['supported_margin']:.2f}"
        )

        lines.append(
            f"  ambiguous_margin   = "
            f"{best['ambiguous_margin']:.2f}"
        )

        lines.append(
            f"  supported          = "
            f"{int(best['supported_count'])}"
        )

        lines.append(
            f"  coverage           = "
            f"{best['supported_coverage']:.3f}"
        )

        lines.append(
            "NOTE: This is still evaluated on the same "
            "dataset used by the optimizer and should not "
            "be treated as a final unbiased validation."
        )

    else:

        lines.append(
            "No candidate achieved zero false SUPPORTED."
        )

        lines.append(
            "Prefer the candidate with the best "
            "precision/coverage trade-off."
        )

    lines.append("")
    lines.append(
        "Recommended next step:"
    )
    lines.append(
        "Inspect class-wise results before changing "
        "the production verifier thresholds."
    )

    lines.append("")
    lines.append(
        "=" * 78
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("=" * 78)
    print("THRESHOLD CLASS-WISE ANALYSIS")
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 1. LOAD
    # ========================================================

    print("")
    print("[1/5] Loading CSV files...")

    results_df = load_csv(
        RESULTS_CSV,
        "verifier_test_results.csv"
    )

    top_df = load_csv(
        TOP_CSV,
        "threshold_optimizer_top.csv"
    )

    # ========================================================
    # 2. PREPARE
    # ========================================================

    print("")
    print("[2/5] Preparing evaluation data...")

    eval_df = prepare_results(
        results_df
    )

    print(
        f"[OK] Evaluation rows : "
        f"{len(eval_df):,}"
    )

    print("")
    print("Class distribution:")

    for cls in TARGET_CLASSES:

        count = int(
            (eval_df["true_class"] == cls).sum()
        )

        print(
            f"  {cls:<22}: {count}"
        )

    print("")
    print(
        f"CNN correct          : "
        f"{int(eval_df['cnn_correct'].sum())}"
    )

    print(
        f"CNN wrong            : "
        f"{int((~eval_df['cnn_correct']).sum())}"
    )

    # ========================================================
    # 3. LOAD CANDIDATES
    # ========================================================

    print("")
    print("[3/5] Loading threshold candidates...")

    candidates = load_candidates(
        top_df
    )

    print(
        f"[OK] Candidates        : "
        f"{len(candidates)}"
    )

    print("")
    print(
        "Candidate 0 = CURRENT threshold"
    )

    print(
        "Candidate 1+ = optimizer TOP candidates"
    )

    # ========================================================
    # 4. ANALYSIS
    # ========================================================

    print("")
    print("[4/5] Running class-wise analysis...")

    overall_rows = []
    class_rows = []
    cnn_class_rows = []

    for candidate in candidates:

        rank = candidate["rank"]

        threshold = {
            "weak_evidence":
                candidate["weak_evidence"],

            "supported_evidence":
                candidate["supported_evidence"],

            "supported_margin":
                candidate["supported_margin"],

            "ambiguous_margin":
                candidate["ambiguous_margin"],
        }

        overall_rows.append(
            calculate_overall_metrics(
                eval_df,
                threshold,
                rank,
            )
        )

        class_rows.append(
            calculate_class_metrics(
                eval_df,
                threshold,
                rank,
            )
        )

        cnn_class_rows.append(
            calculate_cnn_class_metrics(
                eval_df,
                threshold,
                rank,
            )
        )

    overall_df = pd.DataFrame(
        overall_rows
    )

    class_df = pd.concat(
        class_rows,
        ignore_index=True
    )

    cnn_class_df = pd.concat(
        cnn_class_rows,
        ignore_index=True
    )

    # ========================================================
    # 5. SAVE
    # ========================================================

    print("")
    print("[5/5] Saving analysis files...")

    overall_path = (
        OUTPUT_DIR
        / "threshold_class_summary.csv"
    )

    class_path = (
        OUTPUT_DIR
        / "threshold_class_analysis.csv"
    )

    cnn_class_path = (
        OUTPUT_DIR
        / "threshold_cnn_class_analysis.csv"
    )

    report_path = (
        OUTPUT_DIR
        / "threshold_class_analysis_report.txt"
    )

    overall_df.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
    )

    class_df.to_csv(
        class_path,
        index=False,
        encoding="utf-8-sig",
    )

    cnn_class_df.to_csv(
        cnn_class_path,
        index=False,
        encoding="utf-8-sig",
    )

    create_report(
        overall_df,
        class_df,
        cnn_class_df,
        report_path,
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print("")
    print("=" * 78)
    print("CURRENT THRESHOLD")
    print("=" * 78)

    current = overall_df[
        overall_df["candidate_rank"] == 0
    ].iloc[0]

    print(
        f"weak_evidence      = "
        f"{current['weak_evidence']:.2f}"
    )

    print(
        f"supported_evidence = "
        f"{current['supported_evidence']:.2f}"
    )

    print(
        f"supported_margin   = "
        f"{current['supported_margin']:.2f}"
    )

    print(
        f"ambiguous_margin   = "
        f"{current['ambiguous_margin']:.2f}"
    )

    print(
        f"SUPPORTED          = "
        f"{int(current['supported_count'])}"
    )

    print(
        f"WEAK               = "
        f"{int(current['weak_count'])}"
    )

    print(
        f"AMBIGUOUS          = "
        f"{int(current['ambiguous_count'])}"
    )

    print("")
    print("=" * 78)
    print("CLASS-WISE CURRENT THRESHOLD")
    print("=" * 78)

    current_class = class_df[
        class_df["candidate_rank"] == 0
    ]

    for _, row in current_class.iterrows():

        precision = row["supported_precision"]
        coverage = row["supported_coverage"]

        print("")
        print(
            f"{row['true_class']}"
        )

        print(
            f"  total             = "
            f"{int(row['total'])}"
        )

        print(
            f"  CNN correct       = "
            f"{int(row['cnn_correct'])}"
        )

        print(
            f"  SUPPORTED         = "
            f"{int(row['supported'])}"
        )

        print(
            f"  WEAK              = "
            f"{int(row['weak'])}"
        )

        print(
            f"  AMBIGUOUS         = "
            f"{int(row['ambiguous'])}"
        )

        print(
            f"  false SUPPORTED   = "
            f"{int(row['false_supported'])}"
        )

        print(
            f"  precision         = "
            f"{precision:.3f}"
            if pd.notna(precision)
            else "  precision         = N/A"
        )

        print(
            f"  coverage          = "
            f"{coverage:.3f}"
            if pd.notna(coverage)
            else "  coverage          = N/A"
        )

    print("")
    print("=" * 78)
    print("TOP CANDIDATES")
    print("=" * 78)

    display_cols = [
        "candidate_rank",
        "weak_evidence",
        "supported_evidence",
        "supported_margin",
        "ambiguous_margin",
        "supported_count",
        "weak_count",
        "ambiguous_count",
        "supported_precision",
        "supported_coverage",
        "false_supported",
    ]

    print(
        overall_df[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("")
    print("=" * 78)
    print("SAVED FILES")
    print("=" * 78)

    print(
        overall_path
    )

    print(
        class_path
    )

    print(
        cnn_class_path
    )

    print(
        report_path
    )

    print("")
    print(
        "[DONE] Class-wise threshold analysis completed."
    )


if __name__ == "__main__":
    main()

