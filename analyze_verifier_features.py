from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CLASSIFIER_DIR = Path(r"D:\Internship\Test_Rule\test_rule_new")
DEFAULT_DATASET_DIR = Path(
    r"D:\Internship\Test_Rule\test_rule_new\Dataset_For_Rule_Base"
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# จำนวนภาพต่อ class
# None = ใช้ทุกภาพ
DEFAULT_MAX_IMAGES_PER_CLASS = None

# Feature ที่ไม่ควรเอามาเป็น evidence
# width / height เป็นขนาดภาพ ไม่ใช่ลักษณะของ defect
EXCLUDED_FEATURES = {
    "width",
    "height",
}


# ============================================================================
# Evidence ranking thresholds
# ============================================================================
#
# ใช้เป็น "heuristic สำหรับจัดกลุ่ม" เท่านั้น
# ยังไม่ใช่ threshold สุดท้ายของ Rule Verifier
#
# AUC = 0.50 -> แยกไม่ได้
# AUC = 1.00 -> แยกได้สมบูรณ์
#
# score = abs(AUC - 0.5) * 2
#
# score:
#   0.00 = ไม่มีความสามารถในการแยก
#   1.00 = แยกได้สมบูรณ์
#
CORE_SCORE = 0.70
SECONDARY_SCORE = 0.55


# ============================================================================
# Import existing feature extractor
# ============================================================================

def import_classifier(classifier_dir: Path):
    sys.path.insert(0, str(classifier_dir))

    try:
        from astro_rule_classifier_new import (
            CLASS_NAMES,
            extract_features,
            read_gray_image,
        )
    except ImportError as e:
        raise SystemExit(
            "\n[ERROR] import astro_rule_classifier_new ไม่สำเร็จ\n"
            f"CLASSIFIER_DIR = {classifier_dir}\n"
            f"รายละเอียด: {e}\n"
        )

    return CLASS_NAMES, extract_features, read_gray_image


# ============================================================================
# Utility
# ============================================================================

def safe_float(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")

    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")

    return float(np.median(np.asarray(values, dtype=np.float64)))


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"

    return f"{value:.{digits}f}"


# ============================================================================
# AUC
# ============================================================================

def auc_one_vs_rest(
    positive_values: Sequence[float],
    negative_values: Sequence[float],
) -> float:
    """
    คำนวณ ROC AUC สำหรับ feature หนึ่งตัว

    positive = ภาพของ class ที่กำลังวิเคราะห์
    negative = ภาพของ class อื่นทั้งหมด

    AUC:
        0.5 = แยกไม่ได้
        >0.5 = ค่าที่สูงมีแนวโน้มเป็น positive
        <0.5 = ค่าที่ต่ำมีแนวโน้มเป็น positive

    ใช้ rank-based implementation
    ไม่ต้องพึ่ง scipy / sklearn
    """

    pos = np.asarray(positive_values, dtype=np.float64)
    neg = np.asarray(negative_values, dtype=np.float64)

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    values = np.concatenate([pos, neg])

    # rank แบบรองรับ ties
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(sorted_values):
        j = i + 1

        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1

        # average rank แบบ 1-based
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank

        i = j

    pos_ranks = ranks[: len(pos)]

    u = np.sum(pos_ranks) - len(pos) * (len(pos) + 1) / 2.0

    auc = u / (len(pos) * len(neg))

    return float(auc)


def auc_strength(auc: float) -> float:
    """
    แปลง AUC ให้เป็น strength 0..1

    0.5 -> 0
    1.0 หรือ 0.0 -> 1
    """

    if not math.isfinite(auc):
        return float("nan")

    return abs(auc - 0.5) * 2.0


def auc_direction(auc: float) -> str:
    """
    HIGH = class มีแนวโน้มมีค่ามากกว่า class อื่น
    LOW  = class มีแนวโน้มมีค่าต่ำกว่า class อื่น
    """

    if not math.isfinite(auc):
        return "UNKNOWN"

    if auc > 0.5:
        return "HIGH"

    if auc < 0.5:
        return "LOW"

    return "NONE"


# ============================================================================
# Feature statistics
# ============================================================================

def compute_distribution(values: Sequence[float]) -> Dict[str, float]:
    return {
        "n": len(values),
        "min": float(np.min(values)) if values else float("nan"),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "max": float(np.max(values)) if values else float("nan"),
        "mean": (
            float(np.mean(np.asarray(values, dtype=np.float64)))
            if values
            else float("nan")
        ),
        "std": (
            float(np.std(np.asarray(values, dtype=np.float64)))
            if values
            else float("nan")
        ),
    }


# ============================================================================
# Feature ranking
# ============================================================================

def classify_feature_strength(score: float) -> str:
    if not math.isfinite(score):
        return "UNKNOWN"

    if score >= CORE_SCORE:
        return "CORE"

    if score >= SECONDARY_SCORE:
        return "SECONDARY"

    return "DROP"


def analyze_feature_for_class(
    target_class: str,
    feature_name: str,
    class_values: Dict[str, List[float]],
    class_names: Sequence[str],
) -> Dict[str, object]:

    positive = class_values.get(target_class, [])

    negative: List[float] = []

    for other_class in class_names:
        if other_class == target_class:
            continue

        negative.extend(class_values.get(other_class, []))

    auc = auc_one_vs_rest(positive, negative)
    strength = auc_strength(auc)
    direction = auc_direction(auc)

    positive_median = median(positive)
    negative_median = median(negative)

    positive_dist = compute_distribution(positive)
    negative_dist = compute_distribution(negative)

    median_gap = float("nan")

    if (
        math.isfinite(positive_median)
        and math.isfinite(negative_median)
    ):
        median_gap = positive_median - negative_median

    return {
        "class": target_class,
        "feature": feature_name,

        "n_positive": len(positive),
        "n_negative": len(negative),

        "class_min": positive_dist["min"],
        "class_p10": positive_dist["p10"],
        "class_p25": positive_dist["p25"],
        "class_median": positive_dist["p50"],
        "class_p75": positive_dist["p75"],
        "class_p90": positive_dist["p90"],
        "class_max": positive_dist["max"],
        "class_mean": positive_dist["mean"],
        "class_std": positive_dist["std"],

        "other_min": negative_dist["min"],
        "other_p10": negative_dist["p10"],
        "other_p25": negative_dist["p25"],
        "other_median": negative_dist["p50"],
        "other_p75": negative_dist["p75"],
        "other_p90": negative_dist["p90"],
        "other_max": negative_dist["max"],
        "other_mean": negative_dist["mean"],
        "other_std": negative_dist["std"],

        "median_gap": median_gap,

        "auc": auc,
        "strength": strength,
        "direction": direction,

        "recommendation": classify_feature_strength(strength),
    }


# ============================================================================
# Pairwise analysis
# ============================================================================

def pairwise_auc(
    class_a: str,
    class_b: str,
    feature_name: str,
    class_values: Dict[str, List[float]],
) -> Dict[str, object]:

    values_a = class_values.get(class_a, [])
    values_b = class_values.get(class_b, [])

    auc = auc_one_vs_rest(values_a, values_b)
    strength = auc_strength(auc)

    return {
        "class_a": class_a,
        "class_b": class_b,
        "feature": feature_name,
        "auc_a_vs_b": auc,
        "strength": strength,
        "direction_for_a": auc_direction(auc),
    }


# ============================================================================
# Dataset extraction
# ============================================================================

def collect_features(
    dataset_dir: Path,
    class_names: Sequence[str],
    read_gray_image,
    extract_features,
    max_images_per_class: int | None,
):
    """
    อ่านภาพทั้งหมดแล้วเก็บ feature เป็น:

        feature_values[class_name][feature_name] = [values...]
    """

    feature_values: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    image_counts: Dict[str, int] = defaultdict(int)
    error_counts: Dict[str, int] = defaultdict(int)

    print()
    print("=" * 80)
    print("STEP 1: Extract features")
    print("=" * 80)

    for class_name in class_names:

        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            print(f"[SKIP] ไม่พบ folder: {class_dir}")
            continue

        paths = [
            p
            for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]

        if max_images_per_class is not None:
            paths = paths[:max_images_per_class]

        print(
            f"\n[{class_name}] "
            f"จำนวนภาพ = {len(paths)}"
        )

        for idx, image_path in enumerate(paths, start=1):

            try:
                gray = read_gray_image(image_path)
                features = extract_features(gray)

                feat_dict = features.__dict__

                for feature_name, value in feat_dict.items():

                    if feature_name in EXCLUDED_FEATURES:
                        continue

                    value = safe_float(value)

                    if value is None:
                        continue

                    feature_values[class_name][feature_name].append(value)

                image_counts[class_name] += 1

            except Exception as e:
                error_counts[class_name] += 1

                print(
                    f"  [ERROR] {image_path.name}: {e}"
                )

            if idx % 50 == 0 or idx == len(paths):
                print(
                    f"  processed {idx}/{len(paths)}",
                    end="\r",
                )

        print()

    print()
    print("Extraction complete.")

    return feature_values, image_counts, error_counts


# ============================================================================
# CSV export
# ============================================================================

def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

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
        writer.writerows(rows)

    print(f"[saved] {path}")


# ============================================================================
# Console report
# ============================================================================

def print_summary(
    ranking_rows: List[Dict[str, object]],
    class_names: Sequence[str],
):
    print()
    print("=" * 80)
    print("STEP 2: Verifier Feature Ranking")
    print("=" * 80)

    for class_name in class_names:

        rows = [
            r
            for r in ranking_rows
            if r["class"] == class_name
        ]

        rows.sort(
            key=lambda r: (
                r["strength"]
                if math.isfinite(float(r["strength"]))
                else -1.0
            ),
            reverse=True,
        )

        print()
        print("-" * 80)
        print(f"{class_name}")
        print("-" * 80)

        print(
            f"{'Feature':<32}"
            f"{'AUC':>8}"
            f"{'Strength':>10}"
            f"{'Dir':>8}"
            f"{'Recommendation':>16}"
        )

        print("-" * 80)

        for row in rows:

            print(
                f"{str(row['feature']):<32}"
                f"{float(row['auc']):>8.3f}"
                f"{float(row['strength']):>10.3f}"
                f"{str(row['direction']):>8}"
                f"{str(row['recommendation']):>16}"
            )

        core = [
            r for r in rows
            if r["recommendation"] == "CORE"
        ]

        secondary = [
            r for r in rows
            if r["recommendation"] == "SECONDARY"
        ]

        print()
        print(
            "CORE:      "
            + (
                ", ".join(str(r["feature"]) for r in core)
                if core
                else "-"
            )
        )

        print(
            "SECONDARY: "
            + (
                ", ".join(str(r["feature"]) for r in secondary)
                if secondary
                else "-"
            )
        )


# ============================================================================
# Pairwise confusion analysis
# ============================================================================

def print_pairwise_worst_features(
    class_names: Sequence[str],
    feature_values: Dict[str, Dict[str, List[float]]],
    top_k: int = 5,
):
    print()
    print("=" * 80)
    print("STEP 3: Pairwise Analysis")
    print("=" * 80)

    print(
        "\nใช้ดูว่า feature ไหนแยก class ที่มีแนวโน้มสับสนกันได้ดี"
    )

    all_features = set()

    for class_name in class_names:
        all_features.update(
            feature_values.get(class_name, {}).keys()
        )

    for i, class_a in enumerate(class_names):

        for class_b in class_names[i + 1:]:

            pair_rows = []

            for feature_name in all_features:

                class_a_values = feature_values.get(
                    class_a, {}
                ).get(feature_name, [])

                class_b_values = feature_values.get(
                    class_b, {}
                ).get(feature_name, [])

                if not class_a_values or not class_b_values:
                    continue

                auc = auc_one_vs_rest(
                    class_a_values,
                    class_b_values,
                )

                strength = auc_strength(auc)

                pair_rows.append(
                    {
                        "feature": feature_name,
                        "auc": auc,
                        "strength": strength,
                    }
                )

            pair_rows.sort(
                key=lambda x: x["strength"],
                reverse=True,
            )

            print()
            print(
                f"{class_a}  VS  {class_b}"
            )

            for row in pair_rows[:top_k]:

                print(
                    f"  {row['feature']:<32}"
                    f"AUC={row['auc']:.3f} "
                    f"strength={row['strength']:.3f}"
                )


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze astronomical image features for "
            "Class-Specific Rule Verifier"
        )
    )

    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=DEFAULT_CLASSIFIER_DIR,
        help="Folder ที่มี astro_rule_classifier_new.py",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset root",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder สำหรับ CSV output",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_CLASS,
        help="จำนวนภาพสูงสุดต่อ class; ไม่ใส่ = ใช้ทั้งหมด",
    )

    parser.add_argument(
        "--pairwise-top",
        type=int,
        default=5,
        help="จำนวน feature สูงสุดที่แสดงใน pairwise analysis",
    )

    args = parser.parse_args()

    classifier_dir = args.classifier_dir
    dataset_dir = args.dataset

    if args.output_dir is None:
        output_dir = classifier_dir / "verifier_analysis"
    else:
        output_dir = args.output_dir

    if not classifier_dir.exists():
        raise SystemExit(
            f"[ERROR] ไม่พบ classifier directory:\n"
            f"{classifier_dir}"
        )

    if not dataset_dir.exists():
        raise SystemExit(
            f"[ERROR] ไม่พบ dataset directory:\n"
            f"{dataset_dir}"
        )

    # ------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------

    (
        class_names,
        extract_features,
        read_gray_image,
    ) = import_classifier(classifier_dir)

    print()
    print("=" * 80)
    print("ASTRONOMY VERIFIER FEATURE ANALYZER")
    print("=" * 80)

    print(f"\nClassifier : {classifier_dir}")
    print(f"Dataset    : {dataset_dir}")
    print(f"Output     : {output_dir}")

    print("\nClasses:")
    for c in class_names:
        print(f"  - {c}")

    # ------------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------------

    (
        feature_values,
        image_counts,
        error_counts,
    ) = collect_features(
        dataset_dir=dataset_dir,
        class_names=class_names,
        read_gray_image=read_gray_image,
        extract_features=extract_features,
        max_images_per_class=args.max_images,
    )

    # ------------------------------------------------------------------------
    # Distribution CSV
    # ------------------------------------------------------------------------

    distribution_rows = []

    all_features = set()

    for class_name in class_names:
        all_features.update(
            feature_values.get(class_name, {}).keys()
        )

    for class_name in class_names:

        for feature_name in sorted(all_features):

            values = feature_values.get(
                class_name, {}
            ).get(feature_name, [])

            if not values:
                continue

            dist = compute_distribution(values)

            distribution_rows.append(
                {
                    "class": class_name,
                    "feature": feature_name,
                    **dist,
                }
            )

    write_csv(
        output_dir / "feature_distribution.csv",
        distribution_rows,
    )

    # ------------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------------

    ranking_rows = []

    for class_name in class_names:

        for feature_name in sorted(all_features):

            row = analyze_feature_for_class(
                target_class=class_name,
                feature_name=feature_name,
                class_values={
                    c: feature_values.get(c, {}).get(
                        feature_name, []
                    )
                    for c in class_names
                },
                class_names=class_names,
            )

            ranking_rows.append(row)

    write_csv(
        output_dir / "verifier_feature_ranking.csv",
        ranking_rows,
    )

    # ------------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------------

    print_summary(
        ranking_rows,
        class_names,
    )

    # ------------------------------------------------------------------------
    # Pairwise analysis
    # ------------------------------------------------------------------------

    print_pairwise_worst_features(
        class_names=class_names,
        feature_values=feature_values,
        top_k=args.pairwise_top,
    )

    # ------------------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(
        f"{'Class':<24}"
        f"{'Images':>10}"
        f"{'Errors':>10}"
    )

    for class_name in class_names:
        print(
            f"{class_name:<24}"
            f"{image_counts.get(class_name, 0):>10}"
            f"{error_counts.get(class_name, 0):>10}"
        )

    # ------------------------------------------------------------------------
    # Final recommendation
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("NEXT STEP")
    print("=" * 80)

    print(
        """
ไฟล์นี้ยังไม่ได้สร้าง Rule Verifier จริง

ผลลัพธ์ที่สำคัญที่สุดคือ:

    verifier_feature_ranking.csv

ให้ดู:
    recommendation = CORE
    recommendation = SECONDARY

โดย CORE คือ feature ที่มีแนวโน้มเหมาะกับการเป็น
"หลักฐานหลัก" ของ class นั้น

ส่วน SECONDARY คือ feature ที่ใช้เป็นหลักฐานเสริม

DROP คือ feature ที่แยก class นี้ออกจาก class อื่น
ได้ไม่ดีพอจากข้อมูลชุดนี้

หมายเหตุ:
    CORE / SECONDARY / DROP เป็น heuristic ranking
    ยังไม่ใช่ threshold สุดท้ายของ verifier
"""
    )

    print()
    print("[DONE]")


if __name__ == "__main__":
    main()