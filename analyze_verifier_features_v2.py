from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CLASSIFIER_DIR = Path(
    r"D:\Internship\Test_Rule\test_rule_new_v2"
)

DEFAULT_DATASET_DIR = (
    DEFAULT_CLASSIFIER_DIR / "Dataset_For_Rule_Base"
)

DEFAULT_OUTPUT_DIR = (
    DEFAULT_CLASSIFIER_DIR / "verifier_analysis_v2"
)

IMG_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

N_FOLDS = 5
RANDOM_SEED = 42

# None = ใช้ทุกภาพ
DEFAULT_MAX_IMAGES_PER_CLASS = None


# ============================================================================
# Feature exclusions
# ============================================================================

EXCLUDED_FEATURES = {
    "width",
    "height",
}


# ============================================================================
# Candidate thresholds
# ============================================================================
#
# IMPORTANT:
# สิ่งเหล่านี้เป็น "candidate screening thresholds"
# ไม่ใช่ threshold ของ Rule Verifier จริง
#
# strength:
#     0.00 = ไม่มี discrimination
#     1.00 = discrimination สมบูรณ์
#
# stability:
#     ยิ่งสูง = feature มีความสม่ำเสมอข้าม folds
#

CORE_MEAN_STRENGTH = 0.65
SECONDARY_MEAN_STRENGTH = 0.50

CORE_MIN_STRENGTH = 0.50
SECONDARY_MIN_STRENGTH = 0.35

MAX_CORE_STD = 0.15
MAX_SECONDARY_STD = 0.22

MIN_DIRECTION_CONSISTENCY = 0.80


# ============================================================================
# Import existing classifier
# ============================================================================

def import_classifier(classifier_dir: Path):

    sys.path.insert(0, str(classifier_dir))

    try:
        from astro_rule_classifier_new_GoodV2 import (
            CLASS_NAMES,
            extract_features,
            read_gray_image,
        )
    except ImportError as e:
        raise SystemExit(
            "\n[ERROR] ไม่สามารถ import astro_rule_classifier_new.py ได้\n"
            f"Folder: {classifier_dir}\n"
            f"Error : {e}\n"
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


def median(values: Sequence[float]) -> float:

    if not values:
        return float("nan")

    return float(
        np.median(
            np.asarray(values, dtype=np.float64)
        )
    )


def percentile(
    values: Sequence[float],
    q: float,
) -> float:

    if not values:
        return float("nan")

    return float(
        np.percentile(
            np.asarray(values, dtype=np.float64),
            q,
        )
    )


def mean_or_nan(values: Sequence[float]) -> float:

    if not values:
        return float("nan")

    return float(
        np.mean(
            np.asarray(values, dtype=np.float64)
        )
    )


def std_or_nan(values: Sequence[float]) -> float:

    if not values:
        return float("nan")

    return float(
        np.std(
            np.asarray(values, dtype=np.float64)
        )
    )


def min_or_nan(values: Sequence[float]) -> float:

    if not values:
        return float("nan")

    return float(
        np.min(
            np.asarray(values, dtype=np.float64)
        )
    )


def max_or_nan(values: Sequence[float]) -> float:

    if not values:
        return float("nan")

    return float(
        np.max(
            np.asarray(values, dtype=np.float64)
        )
    )

def fmt(value, digits=4):

    try:
        value = float(value)
    except Exception:
        return "nan"

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
    Rank-based ROC AUC

    positive = target class
    negative = all other classes
    """

    pos = np.asarray(
        positive_values,
        dtype=np.float64,
    )

    neg = np.asarray(
        negative_values,
        dtype=np.float64,
    )

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    values = np.concatenate(
        [pos, neg]
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    sorted_values = values[order]

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    i = 0

    while i < len(sorted_values):

        j = i + 1

        while (
            j < len(sorted_values)
            and sorted_values[j] == sorted_values[i]
        ):
            j += 1

        avg_rank = (
            (i + 1) + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = avg_rank

        i = j

    pos_ranks = ranks[: len(pos)]

    u = (
        np.sum(pos_ranks)
        - len(pos)
        * (len(pos) + 1)
        / 2.0
    )

    return float(
        u
        / (
            len(pos)
            * len(neg)
        )
    )


def auc_strength(auc: float) -> float:

    if not math.isfinite(auc):
        return float("nan")

    return abs(auc - 0.5) * 2.0


def auc_direction(auc: float) -> str:

    if not math.isfinite(auc):
        return "UNKNOWN"

    if auc > 0.5:
        return "HIGH"

    if auc < 0.5:
        return "LOW"

    return "NONE"


# ============================================================================
# Image collection
# ============================================================================

def collect_image_paths(
    dataset_dir: Path,
    class_names: Sequence[str],
    max_images_per_class: int | None,
) -> Dict[str, List[Path]]:

    result = {}

    for class_name in class_names:

        class_dir = (
            dataset_dir / class_name
        )

        if not class_dir.exists():

            print(
                f"[WARNING] ไม่พบ folder: "
                f"{class_dir}"
            )

            result[class_name] = []
            continue

        paths = [
            p
            for p in sorted(
                class_dir.iterdir()
            )
            if (
                p.is_file()
                and p.suffix.lower()
                in IMG_EXTS
            )
        ]

        if (
            max_images_per_class
            is not None
        ):
            paths = paths[
                :max_images_per_class
            ]

        result[class_name] = paths

    return result


# ============================================================================
# Feature extraction
# ============================================================================

def extract_dataset_features(
    image_paths: Dict[str, List[Path]],
    read_gray_image,
    extract_features,
):
    """
    Return:

        data[class_name][image_name][feature]
    """

    data = defaultdict(dict)

    errors = defaultdict(int)

    print()
    print("=" * 80)
    print("STEP 1: Extract features")
    print("=" * 80)

    for class_name, paths in image_paths.items():

        print(
            f"\n[{class_name}] "
            f"จำนวนภาพ = {len(paths)}"
        )

        for idx, image_path in enumerate(
            paths,
            start=1,
        ):

            try:

                gray = read_gray_image(
                    image_path
                )

                features = extract_features(
                    gray
                )

                feature_dict = (
                    features.__dict__
                )

                clean = {}

                for feature_name, value in (
                    feature_dict.items()
                ):

                    if (
                        feature_name
                        in EXCLUDED_FEATURES
                    ):
                        continue

                    value = safe_float(value)

                    if value is None:
                        continue

                    clean[
                        feature_name
                    ] = value

                data[
                    class_name
                ][
                    image_path.name
                ] = clean

            except Exception as e:

                errors[
                    class_name
                ] += 1

                print(
                    f"\n  [ERROR] "
                    f"{image_path.name}: "
                    f"{e}"
                )

            if (
                idx % 50 == 0
                or idx == len(paths)
            ):
                print(
                    f"  processed "
                    f"{idx}/{len(paths)}",
                    end="\r",
                )

        print()

    return data, errors


# ============================================================================
# Build feature matrix
# ============================================================================

def get_all_features(
    data,
    class_names,
) -> List[str]:

    features = set()

    for class_name in class_names:

        for image_features in (
            data
            .get(class_name, {})
            .values()
        ):

            features.update(
                image_features.keys()
            )

    return sorted(features)


def get_feature_values(
    data,
    class_name: str,
    feature_name: str,
    image_names: Sequence[str] | None = None,
) -> List[float]:

    class_data = data.get(
        class_name,
        {},
    )

    if image_names is None:

        image_names = class_data.keys()

    values = []

    for image_name in image_names:

        value = class_data.get(
            image_name,
            {},
        ).get(feature_name)

        if value is None:
            continue

        values.append(value)

    return values


# ============================================================================
# Stratified 5-fold split
# ============================================================================

def make_stratified_folds(
    data,
    class_names: Sequence[str],
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
):
    """
    สร้าง fold แบบ stratified ต่อ class

    Return:

        folds[fold_index][class_name] = image_names
    """

    rng = random.Random(seed)

    per_class = {}

    for class_name in class_names:

        names = list(
            data
            .get(class_name, {})
            .keys()
        )

        rng.shuffle(names)

        per_class[class_name] = names

    folds = [
        defaultdict(list)
        for _ in range(n_folds)
    ]

    for class_name in class_names:

        names = per_class[class_name]

        for idx, image_name in enumerate(
            names
        ):

            fold_idx = (
                idx % n_folds
            )

            folds[
                fold_idx
            ][
                class_name
            ].append(
                image_name
            )

    return folds


# ============================================================================
# Distribution
# ============================================================================

def distribution_row(
    class_name: str,
    feature_name: str,
    values: Sequence[float],
):

    return {
        "class": class_name,
        "feature": feature_name,
        "n": len(values),
        "min": min_or_nan(values),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "max": max_or_nan(values),
        "mean": mean_or_nan(values),
        "std": std_or_nan(values),
    }


# ============================================================================
# One-vs-rest fold analysis
# ============================================================================

def analyze_fold_one_vs_rest(
    data,
    class_names: Sequence[str],
    target_class: str,
    feature_name: str,
    validation_fold,
) -> Dict[str, object]:

    positive_names = validation_fold.get(
        target_class,
        [],
    )

    positive = get_feature_values(
        data,
        target_class,
        feature_name,
        positive_names,
    )

    negative = []

    for other_class in class_names:

        if other_class == target_class:
            continue

        names = validation_fold.get(
            other_class,
            [],
        )

        negative.extend(
            get_feature_values(
                data,
                other_class,
                feature_name,
                names,
            )
        )

    auc = auc_one_vs_rest(
        positive,
        negative,
    )

    return {
        "auc": auc,
        "strength": auc_strength(auc),
        "direction": auc_direction(auc),
        "n_positive": len(positive),
        "n_negative": len(negative),
    }


# ============================================================================
# Aggregate 5-fold stability
# ============================================================================

def aggregate_stability(
    fold_rows: List[Dict[str, object]],
) -> Dict[str, object]:

    aucs = [
        float(r["auc"])
        for r in fold_rows
        if math.isfinite(
            float(r["auc"])
        )
    ]

    strengths = [
        float(r["strength"])
        for r in fold_rows
        if math.isfinite(
            float(r["strength"])
        )
    ]

    directions = [
        str(r["direction"])
        for r in fold_rows
        if r["direction"]
        in {"HIGH", "LOW"}
    ]

    if not aucs:

        return {
            "mean_auc": float("nan"),
            "std_auc": float("nan"),
            "min_auc": float("nan"),
            "max_auc": float("nan"),
            "mean_strength": float("nan"),
            "std_strength": float("nan"),
            "min_strength": float("nan"),
            "max_strength": float("nan"),
            "direction": "UNKNOWN",
            "direction_consistency": 0.0,
        }

    # ------------------------------------------------------------------------
    # Direction consistency
    # ------------------------------------------------------------------------

    if directions:

        high_count = directions.count(
            "HIGH"
        )

        low_count = directions.count(
            "LOW"
        )

        if high_count >= low_count:

            final_direction = "HIGH"
            direction_consistency = (
                high_count
                / len(directions)
            )

        else:

            final_direction = "LOW"
            direction_consistency = (
                low_count
                / len(directions)
            )

    else:

        final_direction = "UNKNOWN"
        direction_consistency = 0.0

    return {
        "mean_auc": float(
            np.mean(aucs)
        ),
        "std_auc": float(
            np.std(aucs)
        ),
        "min_auc": float(
            np.min(aucs)
        ),
        "max_auc": float(
            np.max(aucs)
        ),

        "mean_strength": float(
            np.mean(strengths)
        ),
        "std_strength": float(
            np.std(strengths)
        ),
        "min_strength": float(
            np.min(strengths)
        ),
        "max_strength": float(
            np.max(strengths)
        ),

        "direction": final_direction,
        "direction_consistency": (
            direction_consistency
        ),
    }


# ============================================================================
# Pairwise fold analysis
# ============================================================================

def analyze_fold_pairwise(
    data,
    class_a: str,
    class_b: str,
    feature_name: str,
    validation_fold,
):

    names_a = validation_fold.get(
        class_a,
        [],
    )

    names_b = validation_fold.get(
        class_b,
        [],
    )

    values_a = get_feature_values(
        data,
        class_a,
        feature_name,
        names_a,
    )

    values_b = get_feature_values(
        data,
        class_b,
        feature_name,
        names_b,
    )

    auc = auc_one_vs_rest(
        values_a,
        values_b,
    )

    return {
        "auc": auc,
        "strength": auc_strength(auc),
        "direction": auc_direction(auc),
        "n_a": len(values_a),
        "n_b": len(values_b),
    }


# ============================================================================
# CSV
# ============================================================================

def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

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

    print(
        f"[saved] {path}"
    )


# ============================================================================
# Candidate classification
# ============================================================================

def classify_candidate(
    mean_strength: float,
    min_strength: float,
    std_strength: float,
    direction_consistency: float,
):

    if not all(
        math.isfinite(x)
        for x in [
            mean_strength,
            min_strength,
            std_strength,
            direction_consistency,
        ]
    ):
        return "DROP"

    # ------------------------------------------------------------------------
    # CORE
    # ------------------------------------------------------------------------

    if (
        mean_strength
        >= CORE_MEAN_STRENGTH
        and min_strength
        >= CORE_MIN_STRENGTH
        and std_strength
        <= MAX_CORE_STD
        and direction_consistency
        >= MIN_DIRECTION_CONSISTENCY
    ):
        return "CORE"

    # ------------------------------------------------------------------------
    # SECONDARY
    # ------------------------------------------------------------------------

    if (
        mean_strength
        >= SECONDARY_MEAN_STRENGTH
        and min_strength
        >= SECONDARY_MIN_STRENGTH
        and std_strength
        <= MAX_SECONDARY_STD
        and direction_consistency
        >= MIN_DIRECTION_CONSISTENCY
    ):
        return "SECONDARY"

    return "DROP"


# ============================================================================
# Candidate score
# ============================================================================

def candidate_score(
    mean_strength: float,
    min_strength: float,
    std_strength: float,
    direction_consistency: float,
) -> float:

    if not all(
        math.isfinite(x)
        for x in [
            mean_strength,
            min_strength,
            std_strength,
            direction_consistency,
        ]
    ):
        return float("nan")

    # ------------------------------------------------------------------------
    # เราให้ importance กับ:
    #
    # 1. average strength
    # 2. worst fold strength
    # 3. consistency
    # 4. low variance
    #
    # ------------------------------------------------------------------------

    stability_bonus = max(
        0.0,
        1.0 - std_strength,
    )

    score = (
        0.45 * mean_strength
        + 0.30 * min_strength
        + 0.15 * direction_consistency
        + 0.10 * stability_bonus
    )

    return float(score)


# ============================================================================
# Print 5-fold report
# ============================================================================

def print_stability_report(
    rows,
    class_names,
):

    print()
    print("=" * 80)
    print("5-FOLD STABILITY ANALYSIS")
    print("=" * 80)

    for class_name in class_names:

        class_rows = [
            r
            for r in rows
            if r["class"]
            == class_name
        ]

        class_rows.sort(
            key=lambda r: (
                float(
                    r["mean_strength"]
                )
                if math.isfinite(
                    float(
                        r[
                            "mean_strength"
                        ]
                    )
                )
                else -1
            ),
            reverse=True,
        )

        print()
        print("-" * 80)
        print(class_name)
        print("-" * 80)

        print(
            f"{'Feature':<32}"
            f"{'Mean':>8}"
            f"{'Std':>8}"
            f"{'Min':>8}"
            f"{'Dir':>8}"
            f"{'Cons.':>8}"
            f"{'Candidate':>14}"
        )

        print("-" * 80)

        for row in class_rows:

            print(
                f"{str(row['feature']):<32}"
                f"{float(row['mean_strength']):>8.3f}"
                f"{float(row['std_strength']):>8.3f}"
                f"{float(row['min_strength']):>8.3f}"
                f"{str(row['direction']):>8}"
                f"{float(row['direction_consistency']):>8.2f}"
                f"{str(row['candidate']):>14}"
            )


# ============================================================================
# Print pairwise stability
# ============================================================================

def print_pairwise_stability(
    rows,
    class_names,
    top_k=5,
):

    print()
    print("=" * 80)
    print("PAIRWISE 5-FOLD STABILITY")
    print("=" * 80)

    for i, class_a in enumerate(
        class_names
    ):

        for class_b in class_names[
            i + 1:
        ]:

            pair_rows = [
                r
                for r in rows
                if (
                    r["class_a"]
                    == class_a
                    and r["class_b"]
                    == class_b
                )
            ]

            pair_rows.sort(
                key=lambda r: (
                    float(
                        r["mean_strength"]
                    )
                    if math.isfinite(
                        float(
                            r[
                                "mean_strength"
                            ]
                        )
                    )
                    else -1
                ),
                reverse=True,
            )

            print()
            print(
                f"{class_a}  VS  {class_b}"
            )

            for row in pair_rows[
                :top_k
            ]:

                print(
                    f"  "
                    f"{str(row['feature']):<32}"
                    f"mean="
                    f"{float(row['mean_strength']):.3f} "
                    f"std="
                    f"{float(row['std_strength']):.3f} "
                    f"min="
                    f"{float(row['min_strength']):.3f} "
                    f"dir="
                    f"{row['direction']} "
                    f"cons="
                    f"{float(row['direction_consistency']):.2f}"
                )


# ============================================================================
# Build candidate CSV
# ============================================================================

def build_candidates(
    one_vs_rest_rows,
    pairwise_rows,
    class_names,
):

    candidates = []

    # ------------------------------------------------------------------------
    # 1. Class-specific candidates
    # ------------------------------------------------------------------------

    for row in one_vs_rest_rows:

        candidate = str(
            row["candidate"]
        )

        if candidate == "DROP":
            continue

        candidates.append(
            {
                "candidate_type": "CLASS_SPECIFIC",
                "class": row["class"],
                "against": "ALL_OTHER_CLASSES",
                "feature": row["feature"],

                "candidate": candidate,

                "mean_auc": row["mean_auc"],
                "std_auc": row["std_auc"],
                "min_auc": row["min_auc"],
                "max_auc": row["max_auc"],

                "mean_strength": row[
                    "mean_strength"
                ],
                "std_strength": row[
                    "std_strength"
                ],
                "min_strength": row[
                    "min_strength"
                ],
                "max_strength": row[
                    "max_strength"
                ],

                "direction": row[
                    "direction"
                ],

                "direction_consistency": row[
                    "direction_consistency"
                ],

                "candidate_score": row[
                    "candidate_score"
                ],
            }
        )

    # ------------------------------------------------------------------------
    # 2. Pairwise candidates
    #
    # Pairwise candidate ไม่ได้หมายความว่าใช้เป็น global evidence
    #
    # มันมีไว้บอกว่า:
    #
    # "feature นี้มีประโยชน์โดยเฉพาะตอนแยก A กับ B"
    # ------------------------------------------------------------------------

    for row in pairwise_rows:

        mean_strength = float(
            row["mean_strength"]
        )

        min_strength = float(
            row["min_strength"]
        )

        std_strength = float(
            row["std_strength"]
        )

        direction_consistency = float(
            row[
                "direction_consistency"
            ]
        )

        if (
            mean_strength
            < SECONDARY_MEAN_STRENGTH
        ):
            continue

        if (
            min_strength
            < SECONDARY_MIN_STRENGTH
        ):
            continue

        if (
            std_strength
            > MAX_SECONDARY_STD
        ):
            continue

        if (
            direction_consistency
            < MIN_DIRECTION_CONSISTENCY
        ):
            continue

        candidates.append(
            {
                "candidate_type": "PAIRWISE",
                "class": row["class_a"],
                "against": row["class_b"],
                "feature": row["feature"],

                "candidate": (
                    "PAIRWISE_CORE"
                    if (
                        mean_strength
                        >= CORE_MEAN_STRENGTH
                        and min_strength
                        >= CORE_MIN_STRENGTH
                        and std_strength
                        <= MAX_CORE_STD
                    )
                    else "PAIRWISE_SECONDARY"
                ),

                "mean_auc": row["mean_auc"],
                "std_auc": row["std_auc"],
                "min_auc": row["min_auc"],
                "max_auc": row["max_auc"],

                "mean_strength": row[
                    "mean_strength"
                ],
                "std_strength": row[
                    "std_strength"
                ],
                "min_strength": row[
                    "min_strength"
                ],
                "max_strength": row[
                    "max_strength"
                ],

                "direction": row[
                    "direction"
                ],

                "direction_consistency": row[
                    "direction_consistency"
                ],

                "candidate_score": row[
                    "candidate_score"
                ],
            }
        )

    candidates.sort(
        key=lambda r: (
            float(
                r["candidate_score"]
            )
            if math.isfinite(
                float(
                    r[
                        "candidate_score"
                    ]
                )
            )
            else -1
        ),
        reverse=True,
    )

    return candidates


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Astronomy Evidence Verifier "
            "Feature Analyzer v2"
        )
    )

    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=DEFAULT_CLASSIFIER_DIR,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_CLASS,
        help=(
            "จำนวนภาพสูงสุดต่อ class; "
            "ไม่ระบุ = ใช้ทุกภาพ"
        ),
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=N_FOLDS,
        help="จำนวน folds",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed",
    )

    parser.add_argument(
        "--pairwise-top",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    classifier_dir = (
        args.classifier_dir
    )

    dataset_dir = args.dataset
    output_dir = args.output_dir

    if not classifier_dir.exists():

        raise SystemExit(
            f"[ERROR] ไม่พบ classifier:\n"
            f"{classifier_dir}"
        )

    if not dataset_dir.exists():

        raise SystemExit(
            f"[ERROR] ไม่พบ dataset:\n"
            f"{dataset_dir}"
        )

    if args.folds < 2:

        raise SystemExit(
            "[ERROR] --folds ต้อง >= 2"
        )

    # ------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------

    (
        class_names,
        extract_features,
        read_gray_image,
    ) = import_classifier(
        classifier_dir
    )

    print()
    print("=" * 80)
    print("ASTRONOMY VERIFIER FEATURE ANALYZER v2")
    print("=" * 80)

    print(
        f"\nClassifier : "
        f"{classifier_dir}"
    )

    print(
        f"Dataset    : "
        f"{dataset_dir}"
    )

    print(
        f"Output     : "
        f"{output_dir}"
    )

    print(
        f"Folds      : "
        f"{args.folds}"
    )

    print(
        f"Seed       : "
        f"{args.seed}"
    )

    print("\nClasses:")

    for class_name in class_names:
        print(
            f"  - {class_name}"
        )

    # ------------------------------------------------------------------------
    # Image paths
    # ------------------------------------------------------------------------

    image_paths = collect_image_paths(
        dataset_dir,
        class_names,
        args.max_images,
    )

    # ------------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------------

    (
        data,
        errors,
    ) = extract_dataset_features(
        image_paths,
        read_gray_image,
        extract_features,
    )

    all_features = get_all_features(
        data,
        class_names,
    )

    print()
    print(
        f"Features discovered: "
        f"{len(all_features)}"
    )

    # ------------------------------------------------------------------------
    # Full distribution
    # ------------------------------------------------------------------------

    distribution_rows = []

    for class_name in class_names:

        for feature_name in all_features:

            values = get_feature_values(
                data,
                class_name,
                feature_name,
            )

            if not values:
                continue

            distribution_rows.append(
                distribution_row(
                    class_name,
                    feature_name,
                    values,
                )
            )

    write_csv(
        output_dir
        / "feature_distribution.csv",
        distribution_rows,
    )

    # ------------------------------------------------------------------------
    # 5-fold split
    # ------------------------------------------------------------------------

    folds = make_stratified_folds(
        data,
        class_names,
        n_folds=args.folds,
        seed=args.seed,
    )

    # ------------------------------------------------------------------------
    # One-vs-rest analysis
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 2: 5-FOLD ONE-VS-REST")
    print("=" * 80)

    one_vs_rest_fold_rows = []
    one_vs_rest_stability_rows = []

    for class_name in class_names:

        for feature_name in all_features:

            fold_results = []

            for fold_idx in range(
                args.folds
            ):

                result = (
                    analyze_fold_one_vs_rest(
                        data,
                        class_names,
                        class_name,
                        feature_name,
                        folds[fold_idx],
                    )
                )

                fold_results.append(
                    result
                )

                one_vs_rest_fold_rows.append(
                    {
                        "class": class_name,
                        "feature": feature_name,
                        "fold": fold_idx + 1,
                        **result,
                    }
                )

            aggregate = (
                aggregate_stability(
                    fold_results
                )
            )

            candidate = (
                classify_candidate(
                    aggregate[
                        "mean_strength"
                    ],
                    aggregate[
                        "min_strength"
                    ],
                    aggregate[
                        "std_strength"
                    ],
                    aggregate[
                        "direction_consistency"
                    ],
                )
            )

            score = candidate_score(
                aggregate[
                    "mean_strength"
                ],
                aggregate[
                    "min_strength"
                ],
                aggregate[
                    "std_strength"
                ],
                aggregate[
                    "direction_consistency"
                ],
            )

            one_vs_rest_stability_rows.append(
                {
                    "class": class_name,
                    "feature": feature_name,

                    **aggregate,

                    "candidate": candidate,
                    "candidate_score": score,
                }
            )

    write_csv(
        output_dir
        / "verifier_feature_5fold.csv",
        one_vs_rest_fold_rows,
    )

    write_csv(
        output_dir
        / "verifier_feature_stability.csv",
        one_vs_rest_stability_rows,
    )

    # ------------------------------------------------------------------------
    # Print stability
    # ------------------------------------------------------------------------

    print_stability_report(
        one_vs_rest_stability_rows,
        class_names,
    )

    # ------------------------------------------------------------------------
    # Pairwise stability
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 3: PAIRWISE 5-FOLD STABILITY")
    print("=" * 80)

    pairwise_fold_rows = []
    pairwise_stability_rows = []

    for i, class_a in enumerate(
        class_names
    ):

        for class_b in class_names[
            i + 1:
        ]:

            for feature_name in all_features:

                fold_results = []

                for fold_idx in range(
                    args.folds
                ):

                    result = (
                        analyze_fold_pairwise(
                            data,
                            class_a,
                            class_b,
                            feature_name,
                            folds[fold_idx],
                        )
                    )

                    fold_results.append(
                        result
                    )

                    pairwise_fold_rows.append(
                        {
                            "class_a": class_a,
                            "class_b": class_b,
                            "feature": feature_name,
                            "fold": fold_idx + 1,
                            **result,
                        }
                    )

                aggregate = (
                    aggregate_stability(
                        fold_results
                    )
                )

                score = candidate_score(
                    aggregate[
                        "mean_strength"
                    ],
                    aggregate[
                        "min_strength"
                    ],
                    aggregate[
                        "std_strength"
                    ],
                    aggregate[
                        "direction_consistency"
                    ],
                )

                pairwise_stability_rows.append(
                    {
                        "class_a": class_a,
                        "class_b": class_b,
                        "feature": feature_name,

                        **aggregate,

                        "candidate_score": score,
                    }
                )

    write_csv(
        output_dir
        / "pairwise_feature_5fold.csv",
        pairwise_fold_rows,
    )

    write_csv(
        output_dir
        / "pairwise_feature_stability.csv",
        pairwise_stability_rows,
    )

    print_pairwise_stability(
        pairwise_stability_rows,
        class_names,
        top_k=args.pairwise_top,
    )

    # ------------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 4: BUILD VERIFIER EVIDENCE CANDIDATES")
    print("=" * 80)

    candidates = build_candidates(
        one_vs_rest_stability_rows,
        pairwise_stability_rows,
        class_names,
    )

    write_csv(
        output_dir
        / "verifier_evidence_candidates.csv",
        candidates,
    )

    # ------------------------------------------------------------------------
    # Candidate summary
    # ------------------------------------------------------------------------

    print()

    for class_name in class_names:

        class_candidates = [
            r
            for r in candidates
            if (
                r["candidate_type"]
                == "CLASS_SPECIFIC"
                and r["class"]
                == class_name
            )
        ]

        core = [
            r
            for r in class_candidates
            if r["candidate"]
            == "CORE"
        ]

        secondary = [
            r
            for r in class_candidates
            if r["candidate"]
            == "SECONDARY"
        ]

        print()
        print(
            f"{class_name}"
        )

        print(
            "  CORE:"
        )

        if core:

            for row in core:
                print(
                    f"    - "
                    f"{row['feature']} "
                    f"(score="
                    f"{float(row['candidate_score']):.3f}, "
                    f"strength="
                    f"{float(row['mean_strength']):.3f})"
                )

        else:

            print(
                "    -"
            )

        print(
            "  SECONDARY:"
        )

        if secondary:

            for row in secondary:
                print(
                    f"    - "
                    f"{row['feature']} "
                    f"(score="
                    f"{float(row['candidate_score']):.3f}, "
                    f"strength="
                    f"{float(row['mean_strength']):.3f})"
                )

        else:

            print(
                "    -"
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
            f"{len(data.get(class_name, {})):>10}"
            f"{errors.get(class_name, 0):>10}"
        )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        f"""
{output_dir}

1. feature_distribution.csv
   └─ Distribution ของ feature ทุก class

2. verifier_feature_5fold.csv
   └─ ผล AUC/Strength ของแต่ละ fold

3. verifier_feature_stability.csv
   └─ ผลสรุป 5-fold ของแต่ละ feature/class

4. pairwise_feature_5fold.csv
   └─ ผล pairwise ของแต่ละ fold

5. pairwise_feature_stability.csv
   └─ ผลสรุป pairwise stability

6. verifier_evidence_candidates.csv
   └─ Candidate Evidence สำหรับ Verifier
"""
    )

    print()
    print("=" * 80)
    print("IMPORTANT")
    print("=" * 80)

    print(
        """
CORE / SECONDARY ในตอนนี้ยังเป็นเพียง
candidate screening

ยังไม่ใช่ Rule threshold จริง

ขั้นต่อไปเราจะเอา candidate เหล่านี้
ไปสร้าง Evidence Verifier สำหรับ CNN
โดยไม่ใช้ fuse_scores()
และไม่เปลี่ยน class prediction ของ CNN
"""
    )

    print()
    print("[DONE]")


if __name__ == "__main__":
    main()