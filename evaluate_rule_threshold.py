# ============================================================
# evaluate_rule_threshold.py
#
# Purpose:
#   Evaluate class-specific Rule threshold for
#   CNN -> predicted class -> corresponding Rule -> APPROVE/REJECT
#
# IMPORTANT:
#   - Does NOT modify Feature extraction
#   - Does NOT modify Rule definitions
#   - Does NOT modify CNN
#   - Uses ONLY the Rule corresponding to CNN predicted class
#
# Outputs:
#   1. rule_gate_results.csv
#   2. threshold_sweep.csv
#   3. threshold_evaluation_report.txt
#
# ============================================================

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Internship\Test_Rule\test_rule_new_v2"
)

DEFAULT_CLASSIFIER = (
    PROJECT_ROOT / "astro_rule_classifier_new_V2.py"
)

DEFAULT_DATASET = (
    PROJECT_ROOT / "Dataset_For_Rule_Base"
)

DEFAULT_ONNX = (
    PROJECT_ROOT / "eff_b0_kfold_add_focal_r2.onnx"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "verifier_analysis_v2" / "threshold_evaluation"
)


# Thresholds to test
DEFAULT_THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# IMPORT V2 CLASSIFIER
# ============================================================

def import_classifier(classifier_path: Path):
    """
    Dynamically import astro_rule_classifier_new_V2.py

    Important:
    For Python 3.12 + dataclass,
    module must be registered in sys.modules
    BEFORE exec_module().
    """

    classifier_path = classifier_path.resolve()

    if not classifier_path.exists():
        raise SystemExit(
            "\n[ERROR] ไม่พบ classifier:\n"
            f"{classifier_path}\n"
        )

    module_name = "astro_rule_classifier_new_V2"

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(classifier_path),
    )

    if spec is None or spec.loader is None:
        raise SystemExit(
            "\n[ERROR] ไม่สามารถสร้าง import spec ได้\n"
            f"Path: {classifier_path}\n"
        )

    module = importlib.util.module_from_spec(spec)

    # สำคัญสำหรับ Python 3.12 + dataclass
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)

    except Exception as exc:
        sys.modules.pop(module_name, None)

        raise SystemExit(
            "\n[ERROR] import classifier ไม่สำเร็จ\n"
            f"Path : {classifier_path}\n"
            f"Error: {repr(exc)}\n"
        ) from exc

    required = [
        "CLASS_NAMES",
        "predict_with_rules",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise SystemExit(
            "\n[ERROR] classifier ไม่มีสิ่งที่ต้องใช้:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    print("[OK] import classifier สำเร็จ")
    print(f"     {classifier_path}")

    return module


# ============================================================
# DATASET
# ============================================================

def collect_images(
    dataset_dir: Path,
    class_names: list[str],
) -> list[tuple[Path, str]]:
    """
    Scan dataset recursively.

    Expected structure:

    Dataset_For_Rule_Base/
        01_Good/
            image1.jpg
            image2.jpg

        02_Out_of_Focus/
            image3.jpg

        ...

    true_class is inferred from the first directory
    under dataset_dir that matches CLASS_NAMES.
    """

    dataset_dir = dataset_dir.resolve()

    if not dataset_dir.exists():
        raise SystemExit(
            "\n[ERROR] ไม่พบ Dataset:\n"
            f"{dataset_dir}\n"
        )

    class_set = set(class_names)

    samples = []

    for path in sorted(dataset_dir.rglob("*")):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            relative = path.relative_to(dataset_dir)
        except ValueError:
            continue

        parts = relative.parts

        true_class = None

        for part in parts[:-1]:
            if part in class_set:
                true_class = part
                break

        if true_class is None:
            continue

        samples.append(
            (
                path,
                true_class,
            )
        )

    return samples


# ============================================================
# SAFE VALUE
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert value to float safely.
    """

    try:
        result = float(value)

        if result != result:  # NaN
            return default

        return result

    except Exception:
        return default


# ============================================================
# RUN CNN + RULE
# ============================================================

def evaluate_images(
    classifier,
    samples: list[tuple[Path, str]],
    onnx_path: Path,
) -> list[dict[str, Any]]:
    """
    Run each image through:

        CNN
         ↓
        cnn_class
         ↓
        Rule[c nn_class]
         ↓
        rule_score

    Threshold is NOT applied here.

    We calculate rule_score only once.
    This allows threshold sweep without running
    CNN and feature extraction repeatedly.
    """

    results = []

    total = len(samples)

    print()
    print("=" * 70)
    print("STEP 1: CNN + CLASS-SPECIFIC RULE")
    print("=" * 70)
    print(f"Images : {total}")
    print(f"ONNX   : {onnx_path}")
    print()

    if not onnx_path.exists():
        raise SystemExit(
            "\n[ERROR] ไม่พบ ONNX model:\n"
            f"{onnx_path}\n"
        )

    for index, (image_path, true_class) in enumerate(
        samples,
        start=1,
    ):

        print(
            f"[{index:>4}/{total}] "
            f"{image_path.name}"
        )

        try:

            output = classifier.predict_with_rules(
                image_path=image_path,
                onnx_path=onnx_path,
                approve_threshold=0.45,
            )

            cnn_class = output.get(
                "model_class",
                output.get("cnn_class"),
            )

            cnn_confidence = output.get(
                "model_confidence",
                output.get("cnn_confidence"),
            )

            rule_score = output.get(
                "rule_score"
            )

            if cnn_class is None:
                raise ValueError(
                    "ไม่พบ CNN predicted class"
                )

            if rule_score is None:
                raise ValueError(
                    "ไม่พบ rule_score"
                )

            cnn_confidence = safe_float(
                cnn_confidence
            )

            rule_score = safe_float(
                rule_score
            )

            cnn_correct = (
                cnn_class == true_class
            )

            results.append(
                {
                    "image_path": str(image_path),
                    "true_class": true_class,
                    "cnn_class": cnn_class,
                    "cnn_confidence": cnn_confidence,
                    "rule_score": rule_score,
                    "cnn_correct": cnn_correct,
                    "error": "",
                }
            )

        except Exception as exc:

            print(
                f"       [ERROR] {repr(exc)}"
            )

            results.append(
                {
                    "image_path": str(image_path),
                    "true_class": true_class,
                    "cnn_class": "",
                    "cnn_confidence": 0.0,
                    "rule_score": 0.0,
                    "cnn_correct": False,
                    "error": repr(exc),
                }
            )

    return results


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    results: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """
    Evaluate one threshold.

    Definitions:

    CNN correct:
        cnn_class == true_class

    Correct Approve:
        CNN correct AND Rule says APPROVE

    False Reject:
        CNN correct AND Rule says REJECT

    False Approve:
        CNN wrong AND Rule says APPROVE

    Correct Reject:
        CNN wrong AND Rule says REJECT
    """

    valid = [
        row
        for row in results
        if row["error"] == ""
    ]

    total = len(valid)

    if total == 0:
        return {
            "threshold": threshold,
            "total": 0,
        }

    correct_cnn = sum(
        row["cnn_correct"]
        for row in valid
    )

    wrong_cnn = total - correct_cnn

    approved = 0
    rejected = 0

    correct_approve = 0
    false_reject = 0

    false_approve = 0
    correct_reject = 0

    for row in valid:

        approve = (
            row["rule_score"] >= threshold
        )

        if approve:
            approved += 1
        else:
            rejected += 1

        if row["cnn_correct"]:

            if approve:
                correct_approve += 1
            else:
                false_reject += 1

        else:

            if approve:
                false_approve += 1
            else:
                correct_reject += 1

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    approve_rate = (
        approved / total
        if total > 0
        else 0.0
    )

    reject_rate = (
        rejected / total
        if total > 0
        else 0.0
    )

    cnn_accuracy = (
        correct_cnn / total
        if total > 0
        else 0.0
    )

    # Among CNN-correct images:
    # how many are accepted by Rule?
    approve_recall_on_cnn_correct = (
        correct_approve / correct_cnn
        if correct_cnn > 0
        else 0.0
    )

    # Among CNN-correct images:
    # how many were incorrectly rejected?
    false_reject_rate = (
        false_reject / correct_cnn
        if correct_cnn > 0
        else 0.0
    )

    # Among CNN-wrong images:
    # how many were incorrectly accepted?
    false_approve_rate = (
        false_approve / wrong_cnn
        if wrong_cnn > 0
        else 0.0
    )

    # Among all APPROVE decisions:
    # how many are actually CNN-correct?
    approve_precision = (
        correct_approve / approved
        if approved > 0
        else 0.0
    )

    # Gate accuracy:
    # correct approve + correct reject
    gate_accuracy = (
        (correct_approve + correct_reject)
        / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Balanced gate score
    #
    # We want:
    #
    #   High approval of CNN-correct images
    #   High rejection of CNN-wrong images
    #
    # TPR = correct approve / CNN correct
    # TNR = correct reject / CNN wrong
    #
    # Balanced Accuracy =
    #     (TPR + TNR) / 2
    # --------------------------------------------------------

    tpr = (
        correct_approve / correct_cnn
        if correct_cnn > 0
        else 0.0
    )

    tnr = (
        correct_reject / wrong_cnn
        if wrong_cnn > 0
        else 0.0
    )

    balanced_gate_accuracy = (
        (tpr + tnr) / 2
    )

    return {
        "threshold": threshold,
        "total": total,

        "cnn_correct": correct_cnn,
        "cnn_wrong": wrong_cnn,
        "cnn_accuracy": cnn_accuracy,

        "approved": approved,
        "rejected": rejected,
        "approve_rate": approve_rate,
        "reject_rate": reject_rate,

        "correct_approve": correct_approve,
        "false_reject": false_reject,

        "false_approve": false_approve,
        "correct_reject": correct_reject,

        "approve_precision": approve_precision,

        "approve_recall_on_cnn_correct":
            approve_recall_on_cnn_correct,

        "false_reject_rate":
            false_reject_rate,

        "false_approve_rate":
            false_approve_rate,

        "gate_accuracy":
            gate_accuracy,

        "balanced_gate_accuracy":
            balanced_gate_accuracy,
    }


# ============================================================
# APPLY THRESHOLD TO IMAGE RESULTS
# ============================================================

def create_threshold_results(
    results: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """
    Create row-level results for one threshold.
    """

    output = []

    for row in results:

        if row["error"] != "":
            decision = "ERROR"

        else:
            if row["rule_score"] >= threshold:
                decision = "APPROVE"
            else:
                decision = "REJECT"

        output.append(
            {
                "image_path":
                    row["image_path"],

                "true_class":
                    row["true_class"],

                "cnn_class":
                    row["cnn_class"],

                "cnn_confidence":
                    row["cnn_confidence"],

                "rule_score":
                    row["rule_score"],

                "approve_threshold":
                    threshold,

                "decision":
                    decision,

                "cnn_correct":
                    row["cnn_correct"],

                "error":
                    row["error"],
            }
        )

    return output


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
):
    """
    Save list of dictionaries to CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    path: Path,
    results: list[dict[str, Any]],
    sweep: list[dict[str, Any]],
):
    """
    Save human-readable threshold evaluation report.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid = [
        row
        for row in results
        if row["error"] == ""
    ]

    errors = [
        row
        for row in results
        if row["error"] != ""
    ]

    total = len(results)
    valid_count = len(valid)
    error_count = len(errors)

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write("=" * 80 + "\n")
        f.write("RULE THRESHOLD EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("Dataset summary\n")
        f.write("-" * 80 + "\n")

        f.write(
            f"Total images : {total}\n"
        )

        f.write(
            f"Valid images : {valid_count}\n"
        )

        f.write(
            f"Errors       : {error_count}\n\n"
        )

        if not sweep:
            f.write(
                "No threshold results.\n"
            )
            return

        # ----------------------------------------------------
        # Best threshold by balanced gate accuracy
        # ----------------------------------------------------

        best_balanced = max(
            sweep,
            key=lambda x:
                x["balanced_gate_accuracy"],
        )

        # Best threshold by lowest false approve rate
        # while keeping at least 80% approval recall
        candidates = [
            x
            for x in sweep
            if x[
                "approve_recall_on_cnn_correct"
            ] >= 0.80
        ]

        if candidates:

            best_safe = min(
                candidates,
                key=lambda x:
                    x["false_approve_rate"],
            )

        else:
            best_safe = None

        # ----------------------------------------------------
        # Recommended threshold
        # ----------------------------------------------------

        f.write(
            "BEST THRESHOLD BY BALANCED GATE ACCURACY\n"
        )
        f.write("-" * 80 + "\n")

        f.write(
            f"Threshold : "
            f"{best_balanced['threshold']:.2f}\n"
        )

        f.write(
            f"Balanced gate accuracy : "
            f"{best_balanced['balanced_gate_accuracy']:.4f}\n"
        )

        f.write(
            f"Approve recall on CNN-correct : "
            f"{best_balanced['approve_recall_on_cnn_correct']:.4f}\n"
        )

        f.write(
            f"False approve rate : "
            f"{best_balanced['false_approve_rate']:.4f}\n"
        )

        f.write(
            f"False reject rate : "
            f"{best_balanced['false_reject_rate']:.4f}\n"
        )

        f.write("\n")

        if best_safe is not None:

            f.write(
                "BEST LOWER-RISK THRESHOLD\n"
            )
            f.write("-" * 80 + "\n")

            f.write(
                "Criteria: "
                "approve recall on CNN-correct >= 80%\n"
            )

            f.write(
                f"Threshold : "
                f"{best_safe['threshold']:.2f}\n"
            )

            f.write(
                f"Approve recall : "
                f"{best_safe['approve_recall_on_cnn_correct']:.4f}\n"
            )

            f.write(
                f"False approve rate : "
                f"{best_safe['false_approve_rate']:.4f}\n"
            )

            f.write(
                f"False reject rate : "
                f"{best_safe['false_reject_rate']:.4f}\n"
            )

            f.write("\n")

        # ----------------------------------------------------
        # Full table
        # ----------------------------------------------------

        f.write(
            "THRESHOLD COMPARISON\n"
        )
        f.write("-" * 80 + "\n")

        headers = [
            "Threshold",
            "Approve%",
            "Reject%",
            "FR%",
            "FA%",
            "ApprovePrec%",
            "GateAcc%",
            "Balanced%",
        ]

        f.write(
            " | ".join(
                f"{h:>12}"
                for h in headers
            )
            + "\n"
        )

        f.write("-" * 110 + "\n")

        for row in sweep:

            f.write(
                f"{row['threshold']:>12.2f} | "
                f"{row['approve_rate'] * 100:>11.2f} | "
                f"{row['reject_rate'] * 100:>11.2f} | "
                f"{row['false_reject_rate'] * 100:>11.2f} | "
                f"{row['false_approve_rate'] * 100:>11.2f} | "
                f"{row['approve_precision'] * 100:>11.2f} | "
                f"{row['gate_accuracy'] * 100:>11.2f} | "
                f"{row['balanced_gate_accuracy'] * 100:>11.2f}\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        f.write(
            "INTERPRETATION\n"
        )
        f.write("-" * 80 + "\n")

        f.write(
            "CNN correctness is evaluated first:\n"
            "  cnn_class == true_class\n\n"
        )

        f.write(
            "For CNN-correct images:\n"
            "  APPROVE = desired\n"
            "  REJECT  = False Reject\n\n"
        )

        f.write(
            "For CNN-wrong images:\n"
            "  REJECT  = desired\n"
            "  APPROVE = False Approve\n\n"
        )

        f.write(
            "Therefore, threshold selection should consider "
            "both False Reject and False Approve,\n"
            "not only the overall approve rate.\n"
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    sweep: list[dict[str, Any]],
):
    """
    Print threshold comparison to terminal.
    """

    if not sweep:
        return

    print()
    print("=" * 110)
    print("THRESHOLD SWEEP")
    print("=" * 110)

    print(
        f"{'Threshold':>10} "
        f"{'Approve%':>10} "
        f"{'Reject%':>10} "
        f"{'FR%':>10} "
        f"{'FA%':>10} "
        f"{'ApprovePrec%':>14} "
        f"{'GateAcc%':>12} "
        f"{'Balanced%':>12}"
    )

    print("-" * 110)

    for row in sweep:

        print(
            f"{row['threshold']:>10.2f} "
            f"{row['approve_rate'] * 100:>9.2f}% "
            f"{row['reject_rate'] * 100:>9.2f}% "
            f"{row['false_reject_rate'] * 100:>9.2f}% "
            f"{row['false_approve_rate'] * 100:>9.2f}% "
            f"{row['approve_precision'] * 100:>13.2f}% "
            f"{row['gate_accuracy'] * 100:>11.2f}% "
            f"{row['balanced_gate_accuracy'] * 100:>11.2f}%"
        )

    # Best
    best = max(
        sweep,
        key=lambda x:
            x["balanced_gate_accuracy"],
    )

    print()
    print("=" * 110)
    print("BEST THRESHOLD")
    print("=" * 110)

    print(
        f"Threshold = {best['threshold']:.2f}"
    )

    print(
        f"Balanced Gate Accuracy = "
        f"{best['balanced_gate_accuracy'] * 100:.2f}%"
    )

    print(
        f"Approve Recall on CNN-correct = "
        f"{best['approve_recall_on_cnn_correct'] * 100:.2f}%"
    )

    print(
        f"False Reject Rate = "
        f"{best['false_reject_rate'] * 100:.2f}%"
    )

    print(
        f"False Approve Rate = "
        f"{best['false_approve_rate'] * 100:.2f}%"
    )

    print(
        f"Approve Precision = "
        f"{best['approve_precision'] * 100:.2f}%"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate class-specific Rule threshold "
            "for CNN -> Rule -> APPROVE/REJECT"
        )
    )

    parser.add_argument(
        "--classifier",
        type=Path,
        default=DEFAULT_CLASSIFIER,
        help=(
            "Path to astro_rule_classifier_new_V2.py"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Dataset root directory"
        ),
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX,
        help=(
            "Path to ONNX model"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory"
        ),
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help=(
            "Thresholds to evaluate, "
            "example: --thresholds 0.3 0.4 0.45 0.5"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print()
    print("=" * 80)
    print("ASTRO RULE THRESHOLD EVALUATION")
    print("=" * 80)

    print()
    print("Configuration")
    print("-" * 80)

    print(
        f"Classifier : {args.classifier}"
    )

    print(
        f"Dataset    : {args.dataset}"
    )

    print(
        f"ONNX       : {args.onnx}"
    )

    print(
        f"Output     : {args.output_dir}"
    )

    print(
        "Thresholds : "
        + ", ".join(
            f"{x:.2f}"
            for x in args.thresholds
        )
    )

    # --------------------------------------------------------
    # Import classifier
    # --------------------------------------------------------

    classifier = import_classifier(
        args.classifier
    )

    class_names = list(
        classifier.CLASS_NAMES
    )

    print()
    print("Classes")
    print("-" * 80)

    for i, name in enumerate(
        class_names,
        start=1,
    ):
        print(
            f"{i}. {name}"
        )

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 0: SCAN DATASET")
    print("=" * 80)

    samples = collect_images(
        args.dataset,
        class_names,
    )

    if not samples:
        raise SystemExit(
            "\n[ERROR] ไม่พบ image ที่อยู่ใน class folder"
        )

    print(
        f"[OK] พบรูปทั้งหมด {len(samples)} รูป"
    )

    # --------------------------------------------------------
    # Show class distribution
    # --------------------------------------------------------

    distribution = {
        class_name: 0
        for class_name in class_names
    }

    for _, true_class in samples:

        if true_class in distribution:
            distribution[true_class] += 1

    print()
    print("Dataset distribution")
    print("-" * 80)

    for class_name in class_names:

        print(
            f"{class_name:<25} "
            f"{distribution[class_name]:>6}"
        )

    # --------------------------------------------------------
    # CNN + Rule
    # --------------------------------------------------------

    results = evaluate_images(
        classifier=classifier,
        samples=samples,
        onnx_path=args.onnx,
    )

    valid_results = [
        row
        for row in results
        if row["error"] == ""
    ]

    error_results = [
        row
        for row in results
        if row["error"] != ""
    ]

    # --------------------------------------------------------
    # CNN baseline
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CNN BASELINE")
    print("=" * 80)

    if valid_results:

        cnn_correct = sum(
            row["cnn_correct"]
            for row in valid_results
        )

        cnn_accuracy = (
            cnn_correct / len(valid_results)
        )

        print(
            f"Valid images : "
            f"{len(valid_results)}"
        )

        print(
            f"CNN correct  : "
            f"{cnn_correct}"
        )

        print(
            f"CNN wrong    : "
            f"{len(valid_results) - cnn_correct}"
        )

        print(
            f"CNN accuracy : "
            f"{cnn_accuracy * 100:.2f}%"
        )

    print(
        f"Errors       : "
        f"{len(error_results)}"
    )

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 2: THRESHOLD SWEEP")
    print("=" * 80)

    sweep = []

    for threshold in sorted(
        set(args.thresholds)
    ):

        metrics = evaluate_threshold(
            results,
            threshold,
        )

        sweep.append(metrics)

    # --------------------------------------------------------
    # Save row-level result
    #
    # Use 0.45 as the initial reference result.
    # This is NOT declaring 0.45 as final.
    # --------------------------------------------------------

    reference_threshold = 0.45

    closest_threshold = min(
        sweep,
        key=lambda x:
            abs(
                x["threshold"]
                - reference_threshold
            ),
    )

    row_results = create_threshold_results(
        results,
        closest_threshold["threshold"],
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rule_results_path = (
        output_dir /
        "rule_gate_results.csv"
    )

    threshold_sweep_path = (
        output_dir /
        "threshold_sweep.csv"
    )

    report_path = (
        output_dir /
        "threshold_evaluation_report.txt"
    )

    save_csv(
        rule_results_path,
        row_results,
    )

    save_csv(
        threshold_sweep_path,
        sweep,
    )

    save_report(
        report_path,
        results,
        sweep,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_summary(
        sweep
    )

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        f"1. {rule_results_path}"
    )

    print(
        f"2. {threshold_sweep_path}"
    )

    print(
        f"3. {report_path}"
    )

    print()
    print(
        "[DONE] Threshold evaluation เสร็จแล้ว"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()