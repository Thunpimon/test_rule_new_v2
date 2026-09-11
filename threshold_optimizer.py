
# ============================================================
# threshold_optimizer.py
#
# Purpose:
#   Optimize Verifier thresholds using verifier_test_results.csv
#
# Important design:
#   CNN prediction remains authoritative.
#   This script ONLY optimizes:
#       SUPPORTED / WEAK / AMBIGUOUS
#
# It does NOT:
#   - retrain CNN
#   - change CNN class
#   - fuse CNN + rule prediction
#   - use the existing "verdict" column as ground truth
#
# Input:
#   verifier_test_results.csv
#
# Output:
#   threshold_optimizer_results.csv
#   threshold_optimizer_top.csv
#   threshold_optimizer_report.txt
#
# ============================================================

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DEFAULT_INPUT = Path(
    r"D:\Internship\Test_Rule\test_rule_new_v2"
    r"\\verifier_analysis_v2\verifier_test\verifier_test_results.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    r"D:\Internship\Test_Rule\test_rule_new_v2"
    r"\\verifier_analysis_v2\verifier_test_V2"
)

TARGET_CLASSES = [
    "01_Good",
    "02_Out_of_Focus",
    "03_Tracking_Error",
]


# ------------------------------------------------------------
# Current thresholds
# ใช้เป็น baseline เพื่อเปรียบเทียบ
# ------------------------------------------------------------

CURRENT_SUPPORTED_EVIDENCE = 0.60
CURRENT_SUPPORTED_MARGIN = 0.15
CURRENT_WEAK_EVIDENCE = 0.40
CURRENT_AMBIGUOUS_MARGIN = 0.08


# ------------------------------------------------------------
# Candidate threshold ranges
#
# เราไม่ได้สุ่มค่าแบบมั่ว ๆ แต่ทดลองจากช่วงที่สมเหตุสมผล
# ตาม distribution ของ evidence ที่มีอยู่จริง
# ------------------------------------------------------------

WEAK_EVIDENCE_VALUES = np.round(
    np.arange(0.30, 0.66, 0.02),
    3
)

SUPPORTED_EVIDENCE_VALUES = np.round(
    np.arange(0.45, 0.76, 0.02),
    3
)

SUPPORTED_MARGIN_VALUES = np.round(
    np.arange(0.03, 0.26, 0.01),
    3
)

AMBIGUOUS_MARGIN_VALUES = np.round(
    np.arange(0.03, 0.16, 0.01),
    3
)


# ============================================================
# LOAD DATA
# ============================================================

def load_results(csv_path: Path) -> pd.DataFrame:

    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nไม่พบไฟล์:\n{csv_path}\n"
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "true_class",
        "cnn_class",
        "cnn_confidence",
        "cnn_class_evidence",
        "best_rival_evidence",
        "pairwise_margin",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "\nCSV ไม่มี column ที่ต้องใช้:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df: pd.DataFrame):

    work = df.copy()

    # --------------------------------------------------------
    # แปลง numeric
    # --------------------------------------------------------

    numeric_columns = [
        "cnn_confidence",
        "cnn_class_evidence",
        "best_rival_evidence",
        "pairwise_margin",
    ]

    for col in numeric_columns:
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # true_class ต้องเป็น target class
    #
    # เนื่องจาก dataset ปัจจุบันมีบางรูปที่ true_class ว่าง
    # และบาง class อยู่นอก verifier scope
    # --------------------------------------------------------

    labeled = work[
        work["true_class"].isin(TARGET_CLASSES)
    ].copy()

    labeled = labeled.dropna(
        subset=[
            "cnn_class_evidence",
            "best_rival_evidence",
            "pairwise_margin",
        ]
    ).copy()

    # --------------------------------------------------------
    # CNN ถูกต้องหรือไม่
    #
    # สำคัญ:
    # เราไม่ได้เปลี่ยน cnn_class
    # เพียงใช้ true_class เพื่อวัดว่า evidence
    # สนับสนุน prediction ที่ถูกต้องหรือไม่
    # --------------------------------------------------------

    labeled["cnn_correct"] = (
        labeled["cnn_class"] == labeled["true_class"]
    )

    labeled["cnn_in_scope"] = (
        labeled["cnn_class"].isin(TARGET_CLASSES)
    )

    # Optimizer หลักจะใช้เฉพาะ:
    #   true_class อยู่ใน target
    #   CNN prediction อยู่ใน verifier scope
    #
    # เพราะ verifier ไม่ได้ออกแบบมาให้ตัดสิน 04/05/06
    eval_df = labeled[
        labeled["cnn_in_scope"]
    ].copy()

    return labeled, eval_df


# ============================================================
# VERDICT POLICY
# ============================================================

def apply_thresholds(
    evidence,
    margin,
    weak_evidence,
    supported_evidence,
    supported_margin,
    ambiguous_margin,
):
    """
    กติกาของ optimizer

    1. SUPPORTED
       evidence >= supported_evidence
       และ margin >= supported_margin

    2. AMBIGUOUS
       ถ้า margin <= ambiguous_margin
       หรือคู่แข่งชนะ

    3. WEAK
       evidence >= weak_evidence

    4. นอกเหนือจากนั้น
       AMBIGUOUS

    หมายเหตุ:
    ambiguous_margin เป็น threshold ด้านบน
    ดังนั้น margin ติดลบจะเป็น AMBIGUOUS อยู่แล้ว
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
# APPLY TO WHOLE DATAFRAME
# ============================================================

def generate_verdicts(
    df,
    weak_evidence,
    supported_evidence,
    supported_margin,
    ambiguous_margin,
):

    return [
        apply_thresholds(
            evidence=row["cnn_class_evidence"],
            margin=row["pairwise_margin"],
            weak_evidence=weak_evidence,
            supported_evidence=supported_evidence,
            supported_margin=supported_margin,
            ambiguous_margin=ambiguous_margin,
        )
        for _, row in df.iterrows()
    ]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    df,
    weak_evidence,
    supported_evidence,
    supported_margin,
    ambiguous_margin,
):

    if len(df) == 0:
        return None

    verdicts = generate_verdicts(
        df,
        weak_evidence,
        supported_evidence,
        supported_margin,
        ambiguous_margin,
    )

    tmp = df.copy()
    tmp["optimized_verdict"] = verdicts

    total = len(tmp)

    supported_mask = (
        tmp["optimized_verdict"] == "SUPPORTED"
    )

    weak_mask = (
        tmp["optimized_verdict"] == "WEAK"
    )

    ambiguous_mask = (
        tmp["optimized_verdict"] == "AMBIGUOUS"
    )

    supported_count = int(supported_mask.sum())
    weak_count = int(weak_mask.sum())
    ambiguous_count = int(ambiguous_mask.sum())

    # --------------------------------------------------------
    # SUPPORTED quality
    #
    # SUPPORTED ไม่ควรเกิดกับ CNN prediction ที่ผิดบ่อย
    # --------------------------------------------------------

    if supported_count > 0:

        supported_correct = int(
            tmp.loc[
                supported_mask,
                "cnn_correct"
            ].sum()
        )

        supported_precision = (
            supported_correct / supported_count
        )

    else:
        supported_correct = 0
        supported_precision = np.nan

    # --------------------------------------------------------
    # Coverage ของ SUPPORTED
    #
    # ในบรรดา CNN prediction ที่ถูกต้อง
    # มีสัดส่วนเท่าไรที่ verifier กล้ายืนยัน SUPPORTED
    # --------------------------------------------------------

    cnn_correct_count = int(
        tmp["cnn_correct"].sum()
    )

    if cnn_correct_count > 0:
        supported_coverage = (
            supported_correct / cnn_correct_count
        )
    else:
        supported_coverage = 0.0

    # --------------------------------------------------------
    # False SUPPORTED
    #
    # SUPPORTED แต่ CNN prediction ผิด
    # --------------------------------------------------------

    false_supported = int(
        (
            supported_mask
            & (~tmp["cnn_correct"])
        ).sum()
    )

    false_supported_rate = (
        false_supported / total
    )

    # --------------------------------------------------------
    # Verdict rates
    # --------------------------------------------------------

    supported_rate = supported_count / total
    weak_rate = weak_count / total
    ambiguous_rate = ambiguous_count / total

    # --------------------------------------------------------
    # Average evidence / margin by verdict
    # --------------------------------------------------------

    def mean_or_nan(mask, column):
        if mask.sum() == 0:
            return np.nan
        return float(
            tmp.loc[mask, column].mean()
        )

    supported_mean_evidence = mean_or_nan(
        supported_mask,
        "cnn_class_evidence"
    )

    weak_mean_evidence = mean_or_nan(
        weak_mask,
        "cnn_class_evidence"
    )

    ambiguous_mean_evidence = mean_or_nan(
        ambiguous_mask,
        "cnn_class_evidence"
    )

    supported_mean_margin = mean_or_nan(
        supported_mask,
        "pairwise_margin"
    )

    weak_mean_margin = mean_or_nan(
        weak_mask,
        "pairwise_margin"
    )

    ambiguous_mean_margin = mean_or_nan(
        ambiguous_mask,
        "pairwise_margin"
    )

    # --------------------------------------------------------
    # Agreement rate
    #
    # SUPPORTED + WEAK ที่ CNN ถูกต้อง
    # --------------------------------------------------------

    useful_mask = (
        tmp["optimized_verdict"].isin(
            ["SUPPORTED", "WEAK"]
        )
    )

    useful_count = int(useful_mask.sum())

    if useful_count > 0:
        useful_correct = int(
            tmp.loc[
                useful_mask,
                "cnn_correct"
            ].sum()
        )

        useful_precision = (
            useful_correct / useful_count
        )
    else:
        useful_precision = np.nan

    # --------------------------------------------------------
    # Objective
    #
    # เราต้องการ:
    #   1. SUPPORTED precision สูง
    #   2. SUPPORTED coverage ไม่ต่ำจนไม่มี SUPPORTED
    #   3. false SUPPORTED ต่ำ
    #   4. ไม่อยากให้ทุกอย่างกลายเป็น AMBIGUOUS
    #
    # ไม่ได้ optimize เพื่อให้ SUPPORTED เยอะที่สุด
    # --------------------------------------------------------

    if np.isnan(supported_precision):
        precision_component = 0.0
    else:
        precision_component = supported_precision

    # coverage มีน้ำหนักน้อยกว่า precision
    coverage_component = supported_coverage

    false_supported_penalty = false_supported_rate

    # ถ้าไม่มี SUPPORTED เลย ให้ penalty
    no_supported_penalty = (
        0.15 if supported_count == 0 else 0.0
    )

    # ถ้า AMBIGUOUS เกือบทั้งหมด ให้ penalty เล็กน้อย
    excessive_ambiguous_penalty = max(
        0.0,
        ambiguous_rate - 0.90
    )

    objective = (
        0.60 * precision_component
        + 0.25 * coverage_component
        - 1.50 * false_supported_penalty
        - no_supported_penalty
        - excessive_ambiguous_penalty
    )

    return {
        "total": total,

        "supported_count": supported_count,
        "weak_count": weak_count,
        "ambiguous_count": ambiguous_count,

        "supported_rate": supported_rate,
        "weak_rate": weak_rate,
        "ambiguous_rate": ambiguous_rate,

        "cnn_correct_count": cnn_correct_count,

        "supported_correct": supported_correct,
        "supported_precision": supported_precision,
        "supported_coverage": supported_coverage,

        "false_supported": false_supported,
        "false_supported_rate": false_supported_rate,

        "useful_count": useful_count,
        "useful_precision": useful_precision,

        "supported_mean_evidence": supported_mean_evidence,
        "weak_mean_evidence": weak_mean_evidence,
        "ambiguous_mean_evidence": ambiguous_mean_evidence,

        "supported_mean_margin": supported_mean_margin,
        "weak_mean_margin": weak_mean_margin,
        "ambiguous_mean_margin": ambiguous_mean_margin,

        "objective": objective,
    }


# ============================================================
# BASELINE
# ============================================================

def evaluate_baseline(df):

    metrics = calculate_metrics(
        df=df,
        weak_evidence=CURRENT_WEAK_EVIDENCE,
        supported_evidence=CURRENT_SUPPORTED_EVIDENCE,
        supported_margin=CURRENT_SUPPORTED_MARGIN,
        ambiguous_margin=CURRENT_AMBIGUOUS_MARGIN,
    )

    return metrics


# ============================================================
# OPTIMIZER
# ============================================================

def optimize_thresholds(df):

    results = []

    total_combinations = (
        len(WEAK_EVIDENCE_VALUES)
        * len(SUPPORTED_EVIDENCE_VALUES)
        * len(SUPPORTED_MARGIN_VALUES)
        * len(AMBIGUOUS_MARGIN_VALUES)
    )

    print(
        f"\n[INFO] จำนวน threshold combinations: "
        f"{total_combinations:,}"
    )

    counter = 0

    for weak_evidence in WEAK_EVIDENCE_VALUES:

        for supported_evidence in SUPPORTED_EVIDENCE_VALUES:

            # ------------------------------------------------
            # Strong threshold ไม่ควรต่ำกว่า Weak threshold
            # ------------------------------------------------

            if supported_evidence < weak_evidence:
                continue

            for supported_margin in SUPPORTED_MARGIN_VALUES:

                for ambiguous_margin in AMBIGUOUS_MARGIN_VALUES:

                    # ------------------------------------------------
                    # Supported margin ควรมากกว่า ambiguous margin
                    # ------------------------------------------------

                    if supported_margin <= ambiguous_margin:
                        continue

                    metrics = calculate_metrics(
                        df=df,
                        weak_evidence=weak_evidence,
                        supported_evidence=supported_evidence,
                        supported_margin=supported_margin,
                        ambiguous_margin=ambiguous_margin,
                    )

                    if metrics is None:
                        continue

                    row = {
                        "weak_evidence": weak_evidence,
                        "supported_evidence": supported_evidence,
                        "supported_margin": supported_margin,
                        "ambiguous_margin": ambiguous_margin,
                        **metrics,
                    }

                    results.append(row)

                    counter += 1

    print(
        f"[INFO] evaluated combinations: "
        f"{counter:,}"
    )

    return pd.DataFrame(results)


# ============================================================
# RANK RESULTS
# ============================================================

def rank_results(results: pd.DataFrame):

    ranked = results.copy()

    # --------------------------------------------------------
    # Quality gates
    #
    # เราไม่เอา threshold ที่ทำให้ SUPPORTED ผิดเยอะ
    # --------------------------------------------------------

    ranked["passes_precision_90"] = (
        ranked["supported_precision"].fillna(0.0)
        >= 0.90
    )

    ranked["passes_false_supported_10"] = (
        ranked["false_supported_rate"] <= 0.10
    )

    ranked["passes_basic_quality"] = (
        ranked["passes_precision_90"]
        & ranked["passes_false_supported_10"]
    )

    # --------------------------------------------------------
    # Priority:
    #
    # 1. ผ่าน quality gate
    # 2. objective สูง
    # 3. supported precision สูง
    # 4. supported coverage สูง
    # --------------------------------------------------------

    ranked = ranked.sort_values(
        by=[
            "passes_basic_quality",
            "objective",
            "supported_precision",
            "supported_coverage",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    ranked["rank"] = (
        np.arange(len(ranked)) + 1
    )

    return ranked


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    output_dir: Path,
    baseline,
    ranked,
    eval_df,
    labeled_df,
):

    report_path = (
        output_dir
        / "threshold_optimizer_report.txt"
    )

    best = ranked.iloc[0]

    lines = []

    lines.append(
        "=" * 70
    )
    lines.append(
        "THRESHOLD OPTIMIZER REPORT"
    )
    lines.append(
        "=" * 70
    )

    lines.append("")
    lines.append(
        "Dataset"
    )
    lines.append(
        f"  Total rows              : {len(labeled_df)}"
    )
    lines.append(
        f"  Labeled target rows     : {len(labeled_df)}"
    )
    lines.append(
        f"  CNN in verifier scope  : {len(eval_df)}"
    )
    lines.append(
        f"  CNN correct             : "
        f"{int(eval_df['cnn_correct'].sum())}"
    )

    lines.append("")
    lines.append(
        "-" * 70
    )
    lines.append(
        "CURRENT THRESHOLDS"
    )
    lines.append(
        "-" * 70
    )

    lines.append(
        f"  weak_evidence       = "
        f"{CURRENT_WEAK_EVIDENCE:.2f}"
    )
    lines.append(
        f"  supported_evidence  = "
        f"{CURRENT_SUPPORTED_EVIDENCE:.2f}"
    )
    lines.append(
        f"  supported_margin    = "
        f"{CURRENT_SUPPORTED_MARGIN:.2f}"
    )
    lines.append(
        f"  ambiguous_margin    = "
        f"{CURRENT_AMBIGUOUS_MARGIN:.2f}"
    )

    if baseline is not None:

        lines.append("")
        lines.append(
            "Current performance:"
        )

        lines.append(
            f"  SUPPORTED           : "
            f"{baseline['supported_count']}"
        )

        lines.append(
            f"  WEAK                : "
            f"{baseline['weak_count']}"
        )

        lines.append(
            f"  AMBIGUOUS           : "
            f"{baseline['ambiguous_count']}"
        )

        lines.append(
            f"  SUPPORTED precision : "
            f"{baseline['supported_precision']:.4f}"
            if not np.isnan(
                baseline["supported_precision"]
            )
            else
            "  SUPPORTED precision : NaN"
        )

        lines.append(
            f"  SUPPORTED coverage  : "
            f"{baseline['supported_coverage']:.4f}"
        )

        lines.append(
            f"  False SUPPORTED     : "
            f"{baseline['false_supported']}"
        )

    lines.append("")
    lines.append(
        "=" * 70
    )
    lines.append(
        "BEST THRESHOLD CANDIDATE"
    )
    lines.append(
        "=" * 70
    )

    lines.append(
        f"  weak_evidence       = "
        f"{best['weak_evidence']:.2f}"
    )

    lines.append(
        f"  supported_evidence  = "
        f"{best['supported_evidence']:.2f}"
    )

    lines.append(
        f"  supported_margin    = "
        f"{best['supported_margin']:.2f}"
    )

    lines.append(
        f"  ambiguous_margin    = "
        f"{best['ambiguous_margin']:.2f}"
    )

    lines.append("")
    lines.append(
        "Performance:"
    )

    lines.append(
        f"  SUPPORTED           : "
        f"{int(best['supported_count'])}"
    )

    lines.append(
        f"  WEAK                : "
        f"{int(best['weak_count'])}"
    )

    lines.append(
        f"  AMBIGUOUS           : "
        f"{int(best['ambiguous_count'])}"
    )

    if pd.notna(
        best["supported_precision"]
    ):
        lines.append(
            f"  SUPPORTED precision : "
            f"{best['supported_precision']:.4f}"
        )
    else:
        lines.append(
            "  SUPPORTED precision : NaN"
        )

    lines.append(
        f"  SUPPORTED coverage  : "
        f"{best['supported_coverage']:.4f}"
    )

    lines.append(
        f"  False SUPPORTED     : "
        f"{int(best['false_supported'])}"
    )

    lines.append(
        f"  False SUPPORTED rate: "
        f"{best['false_supported_rate']:.4f}"
    )

    lines.append(
        f"  Objective           : "
        f"{best['objective']:.6f}"
    )

    lines.append("")
    lines.append(
        "=" * 70
    )
    lines.append(
        "IMPORTANT"
    )
    lines.append(
        "=" * 70
    )

    lines.append(
        "The optimizer does NOT modify CNN predictions."
    )

    lines.append(
        "The optimizer does NOT use the existing verdict "
        "column as ground truth."
    )

    lines.append(
        "Thresholds should be validated on a separate "
        "validation/test split before being considered final."
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return report_path


# ============================================================
# PRINT TOP RESULTS
# ============================================================

def print_results(
    baseline,
    ranked,
    eval_df,
):

    print("\n")
    print("=" * 70)
    print("THRESHOLD OPTIMIZER")
    print("=" * 70)

    print(
        f"\nEvaluation rows : {len(eval_df)}"
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print("\nCURRENT THRESHOLDS")
    print("-" * 70)

    print(
        f"weak_evidence      = "
        f"{CURRENT_WEAK_EVIDENCE:.2f}"
    )

    print(
        f"supported_evidence = "
        f"{CURRENT_SUPPORTED_EVIDENCE:.2f}"
    )

    print(
        f"supported_margin   = "
        f"{CURRENT_SUPPORTED_MARGIN:.2f}"
    )

    print(
        f"ambiguous_margin   = "
        f"{CURRENT_AMBIGUOUS_MARGIN:.2f}"
    )

    if baseline:

        print(
            f"\nSUPPORTED : "
            f"{baseline['supported_count']}"
        )

        print(
            f"WEAK      : "
            f"{baseline['weak_count']}"
        )

        print(
            f"AMBIGUOUS : "
            f"{baseline['ambiguous_count']}"
        )

        if not np.isnan(
            baseline["supported_precision"]
        ):
            print(
                f"SUPPORTED precision : "
                f"{baseline['supported_precision']:.3f}"
            )

        print(
            f"SUPPORTED coverage  : "
            f"{baseline['supported_coverage']:.3f}"
        )

        print(
            f"False SUPPORTED     : "
            f"{baseline['false_supported']}"
        )

    # --------------------------------------------------------
    # Best
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP 10 THRESHOLD CANDIDATES")
    print("=" * 70)

    columns = [
        "rank",
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
        "objective",
    ]

    top = ranked.head(10)[columns].copy()

    print(
        top.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Optimize Verifier thresholds using "
            "verifier_test_results.csv"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to verifier_test_results.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for optimizer outputs",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n[1/5] Loading CSV..."
    )

    df = load_results(input_path)

    print(
        f"[OK] Loaded {len(df)} rows"
    )

    print(
        "\n[2/5] Preparing evaluation data..."
    )

    labeled_df, eval_df = prepare_data(df)

    print(
        f"[OK] Labeled target rows : "
        f"{len(labeled_df)}"
    )

    print(
        f"[OK] CNN in-scope rows   : "
        f"{len(eval_df)}"
    )

    if len(eval_df) == 0:
        raise SystemExit(
            "\n[ERROR] ไม่มีข้อมูลที่ใช้ optimize ได้"
        )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print(
        "\n[3/5] Evaluating current thresholds..."
    )

    baseline = evaluate_baseline(
        eval_df
    )

    # --------------------------------------------------------
    # Optimize
    # --------------------------------------------------------

    print(
        "\n[4/5] Searching thresholds..."
    )

    results = optimize_thresholds(
        eval_df
    )

    if len(results) == 0:
        raise SystemExit(
            "\n[ERROR] ไม่พบ threshold combination"
        )

    ranked = rank_results(
        results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\n[5/5] Saving results..."
    )

    all_results_path = (
        output_dir
        / "threshold_optimizer_results.csv"
    )

    top_results_path = (
        output_dir
        / "threshold_optimizer_top.csv"
    )

    results.to_csv(
        all_results_path,
        index=False,
        encoding="utf-8-sig"
    )

    ranked.head(50).to_csv(
        top_results_path,
        index=False,
        encoding="utf-8-sig"
    )

    report_path = save_report(
        output_dir=output_dir,
        baseline=baseline,
        ranked=ranked,
        eval_df=eval_df,
        labeled_df=labeled_df,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        baseline=baseline,
        ranked=ranked,
        eval_df=eval_df,
    )

    print("\n")
    print("=" * 70)
    print("SAVED FILES")
    print("=" * 70)

    print(
        f"\n[SAVED] {all_results_path}"
    )

    print(
        f"[SAVED] {top_results_path}"
    )

    print(
        f"[SAVED] {report_path}"
    )

    best = ranked.iloc[0]

    print("\n")
    print("=" * 70)
    print("RECOMMENDED THRESHOLD CANDIDATE")
    print("=" * 70)

    print(
        f"\nweak_evidence      = "
        f"{best['weak_evidence']:.2f}"
    )

    print(
        f"supported_evidence = "
        f"{best['supported_evidence']:.2f}"
    )

    print(
        f"supported_margin   = "
        f"{best['supported_margin']:.2f}"
    )

    print(
        f"ambiguous_margin   = "
        f"{best['ambiguous_margin']:.2f}"
    )

    print("\n*** ยังไม่ควรเอาค่านี้ไปแทนใน test_verifier.py ทันที ***")
    print(
        "ต้องดู TOP candidates และ distribution "
        "ก่อนเลือก threshold สุดท้าย"
    )


if __name__ == "__main__":
    main()

