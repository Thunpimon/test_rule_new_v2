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
    r"D:\Internship\Test_Rule\test_rule_new"
)

DEFAULT_DATASET_DIR = Path(
    r"D:\Internship\Test_Rule\test_rule_new\Dataset_For_Rule_Base"
)

IMG_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

DEFAULT_MAX_IMAGES_PER_CLASS = None

N_FOLDS = 5
RANDOM_SEED = 42

# Feature ที่ไม่ควรเป็น evidence
EXCLUDED_FEATURES = {
    "width",
    "height",
}

# ---------------------------------------------------------------------------
# Candidate thresholds
# ---------------------------------------------------------------------------

CORE_SCORE = 0.70
SECONDARY_SCORE = 0.55

MIN_MEAN_STRENGTH_CORE = 0.65
MIN_WORST_STRENGTH_CORE = 0.50
MIN_STABILITY_CORE = 0.80

MIN_MEAN_STRENGTH_SECONDARY = 0.50
MIN_WORST_STRENGTH_SECONDARY = 0.35
MIN_STABILITY_SECONDARY = 0.60


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


def median(
    values: Sequence[float],
) -> float:

    if not values:

        return float("nan")

    return float(
        np.median(
            np.asarray(values, dtype=np.float64)
        )
    )


def mean(
    values: Sequence[float],
) -> float:

    if not values:

        return float("nan")

    return float(
        np.mean(
            np.asarray(values, dtype=np.float64)
        )
    )


def std(
    values: Sequence[float],
) -> float:

    if not values:

        return float("nan")

    return float(
        np.std(
            np.asarray(values, dtype=np.float64)
        )
    )


def fmt(
    value: float,
    digits: int = 4,
) -> str:

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

    0.5 = แยกไม่ได้
    1.0 = positive มีค่ามากกว่า negative เกือบสมบูรณ์
    0.0 = positive มีค่าต่ำกว่า negative เกือบสมบูรณ์
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

    pos_ranks = ranks[
        : len(pos)
    ]

    u = (
        np.sum(pos_ranks)
        - len(pos)
        * (len(pos) + 1)
        / 2.0
    )

    auc = u / (
        len(pos) * len(neg)
    )

    return float(auc)


def auc_strength(
    auc: float,
) -> float:

    if not math.isfinite(auc):

        return float("nan")

    return abs(auc - 0.5) * 2.0


def auc_direction(
    auc: float,
) -> str:

    if not math.isfinite(auc):

        return "UNKNOWN"

    if auc > 0.5:

        return "HIGH"

    if auc < 0.5:

        return "LOW"

    return "NONE"


# ============================================================================
# Distribution
# ============================================================================

def compute_distribution(
    values: Sequence[float],
) -> Dict[str, float]:

    return {

        "n": len(values),

        "min": (
            float(np.min(values))
            if values
            else float("nan")
        ),

        "p10": percentile(
            values,
            10,
        ),

        "p25": percentile(
            values,
            25,
        ),

        "p50": percentile(
            values,
            50,
        ),

        "p75": percentile(
            values,
            75,
        ),

        "p90": percentile(
            values,
            90,
        ),

        "max": (
            float(np.max(values))
            if values
            else float("nan")
        ),

        "mean": mean(values),

        "std": std(values),
    }


# ============================================================================
# Records
# ============================================================================

def collect_feature_records(
    dataset_dir: Path,
    class_names: Sequence[str],
    read_gray_image,
    extract_features,
    max_images_per_class: int | None,
):
    """
    เก็บ feature แบบรายภาพ

    records = [
        {
            "class": "...",
            "file": "...",
            "features": {
                "sharpness": ...,
                ...
            }
        }
    ]
    """

    records = []

    image_counts = defaultdict(int)
    error_counts = defaultdict(int)

    print()
    print("=" * 80)
    print("STEP 1: Extract per-image features")
    print("=" * 80)

    for class_name in class_names:

        class_dir = dataset_dir / class_name

        if not class_dir.exists():

            print(
                f"[SKIP] ไม่พบ folder: {class_dir}"
            )

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

        if max_images_per_class is not None:

            paths = paths[
                :max_images_per_class
            ]

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

                feature_obj = extract_features(
                    gray
                )

                feat_dict = (
                    feature_obj.__dict__
                )

                clean_features = {}

                for feature_name, value in (
                    feat_dict.items()
                ):

                    if (
                        feature_name
                        in EXCLUDED_FEATURES
                    ):
                        continue

                    value = safe_float(value)

                    if value is None:
                        continue

                    clean_features[
                        feature_name
                    ] = value

                records.append(
                    {
                        "class": class_name,
                        "file": image_path.name,
                        "features": clean_features,
                    }
                )

                image_counts[
                    class_name
                ] += 1

            except Exception as e:

                error_counts[
                    class_name
                ] += 1

                print(
                    f"\n  [ERROR] "
                    f"{image_path.name}: {e}"
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

    print()
    print(
        f"Extraction complete: "
        f"{len(records)} images"
    )

    return (
        records,
        dict(image_counts),
        dict(error_counts),
    )


# ============================================================================
# Convert records
# ============================================================================

def get_all_features(
    records,
) -> List[str]:

    names = set()

    for record in records:

        names.update(
            record["features"].keys()
        )

    return sorted(names)


def build_class_feature_values(
    records,
    class_names,
    feature_name,
):
    """
    Return:

        {
            class_name: [value, value, ...]
        }
    """

    result = {}

    for class_name in class_names:

        result[class_name] = []

    for record in records:

        class_name = record["class"]

        value = record[
            "features"
        ].get(feature_name)

        if value is None:

            continue

        result[
            class_name
        ].append(value)

    return result


# ============================================================================
# One-vs-Rest
# ============================================================================

def analyze_feature_for_class(
    target_class: str,
    feature_name: str,
    class_values,
    class_names,
):

    positive = class_values.get(
        target_class,
        [],
    )

    negative = []

    for other_class in class_names:

        if other_class == target_class:

            continue

        negative.extend(
            class_values.get(
                other_class,
                [],
            )
        )

    auc = auc_one_vs_rest(
        positive,
        negative,
    )

    strength = auc_strength(
        auc
    )

    direction = auc_direction(
        auc
    )

    positive_dist = compute_distribution(
        positive
    )

    negative_dist = compute_distribution(
        negative
    )

    median_gap = float("nan")

    if (
        math.isfinite(
            positive_dist["p50"]
        )
        and math.isfinite(
            negative_dist["p50"]
        )
    ):

        median_gap = (
            positive_dist["p50"]
            - negative_dist["p50"]
        )

    return {

        "class": target_class,

        "feature": feature_name,

        "n_positive": len(
            positive
        ),

        "n_negative": len(
            negative
        ),

        "class_min":
            positive_dist["min"],

        "class_p10":
            positive_dist["p10"],

        "class_p25":
            positive_dist["p25"],

        "class_median":
            positive_dist["p50"],

        "class_p75":
            positive_dist["p75"],

        "class_p90":
            positive_dist["p90"],

        "class_max":
            positive_dist["max"],

        "class_mean":
            positive_dist["mean"],

        "class_std":
            positive_dist["std"],

        "other_min":
            negative_dist["min"],

        "other_p10":
            negative_dist["p10"],

        "other_p25":
            negative_dist["p25"],

        "other_median":
            negative_dist["p50"],

        "other_p75":
            negative_dist["p75"],

        "other_p90":
            negative_dist["p90"],

        "other_max":
            negative_dist["max"],

        "other_mean":
            negative_dist["mean"],

        "other_std":
            negative_dist["std"],

        "median_gap":
            median_gap,

        "auc":
            auc,

        "strength":
            strength,

        "direction":
            direction,
    }


# ============================================================================
# Fold creation
# ============================================================================

def make_stratified_folds(
    records,
    class_names,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
):
    """
    Stratified K-Fold แบบง่าย
    แต่ละ class จะถูกกระจายเข้า fold อย่างสมดุล
    """

    rng = random.Random(seed)

    fold_records = [
        []
        for _ in range(n_folds)
    ]

    for class_name in class_names:

        class_records = [
            r
            for r in records
            if r["class"] == class_name
        ]

        rng.shuffle(
            class_records
        )

        for index, record in enumerate(
            class_records
        ):

            fold_index = (
                index % n_folds
            )

            fold_records[
                fold_index
            ].append(record)

    return fold_records


# ============================================================================
# Build values from records
# ============================================================================

def values_from_records(
    records,
    target_class,
    feature_name,
):

    positive = []
    negative = []

    for record in records:

        value = record[
            "features"
        ].get(feature_name)

        if value is None:

            continue

        if (
            record["class"]
            == target_class
        ):

            positive.append(value)

        else:

            negative.append(value)

    return positive, negative


# ============================================================================
# Threshold search
# ============================================================================

def threshold_candidates(
    positive,
    negative,
):

    values = sorted(
        set(
            positive
            + negative
        )
    )

    if not values:

        return []

    if len(values) <= 100:

        return values

    quantiles = np.linspace(
        0.01,
        0.99,
        99,
    )

    all_values = np.asarray(
        values,
        dtype=np.float64,
    )

    return sorted(
        set(
            float(
                np.quantile(
                    all_values,
                    q,
                )
            )
            for q in quantiles
        )
    )


def evaluate_threshold(
    positive,
    negative,
    threshold,
    direction,
):
    """
    Balanced accuracy

    direction = HIGH
        value >= threshold -> positive

    direction = LOW
        value <= threshold -> positive
    """

    if (
        not positive
        or not negative
    ):

        return float("nan")

    if direction == "HIGH":

        tp = sum(
            x >= threshold
            for x in positive
        )

        fn = len(positive) - tp

        fp = sum(
            x >= threshold
            for x in negative
        )

        tn = len(negative) - fp

    elif direction == "LOW":

        tp = sum(
            x <= threshold
            for x in positive
        )

        fn = len(positive) - tp

        fp = sum(
            x <= threshold
            for x in negative
        )

        tn = len(negative) - fp

    else:

        return float("nan")

    tpr = (
        tp / len(positive)
    )

    tnr = (
        tn / len(negative)
    )

    return (
        tpr + tnr
    ) / 2.0


def find_best_threshold(
    positive,
    negative,
):

    auc = auc_one_vs_rest(
        positive,
        negative,
    )

    direction = auc_direction(
        auc
    )

    if direction == "UNKNOWN":

        return (
            float("nan"),
            direction,
            float("nan"),
        )

    candidates = threshold_candidates(
        positive,
        negative,
    )

    best_threshold = float("nan")
    best_score = -1.0

    for threshold in candidates:

        score = evaluate_threshold(
            positive,
            negative,
            threshold,
            direction,
        )

        if (
            math.isfinite(score)
            and score > best_score
        ):

            best_score = score
            best_threshold = threshold

    return (
        best_threshold,
        direction,
        best_score,
    )


# ============================================================================
# 5-Fold Stability
# ============================================================================

def analyze_5fold_feature(
    records,
    class_names,
    feature_name,
    n_folds=N_FOLDS,
    seed=RANDOM_SEED,
):

    folds = make_stratified_folds(
        records,
        class_names,
        n_folds,
        seed,
    )

    rows = []

    for target_class in class_names:

        fold_strengths = []
        fold_aucs = []
        fold_thresholds = []
        fold_val_scores = []
        fold_directions = []

        for fold_index in range(
            n_folds
        ):

            validation = folds[
                fold_index
            ]

            training = []

            for j, fold in enumerate(
                folds
            ):

                if j != fold_index:

                    training.extend(
                        fold
                    )

            train_pos, train_neg = (
                values_from_records(
                    training,
                    target_class,
                    feature_name,
                )
            )

            val_pos, val_neg = (
                values_from_records(
                    validation,
                    target_class,
                    feature_name,
                )
            )

            auc = auc_one_vs_rest(
                val_pos,
                val_neg,
            )

            strength = auc_strength(
                auc
            )

            direction = auc_direction(
                auc
            )

            (
                threshold,
                train_direction,
                train_score,
            ) = find_best_threshold(
                train_pos,
                train_neg,
            )

            val_score = float("nan")

            if (
                math.isfinite(
                    threshold
                )
                and train_direction
                in {"HIGH", "LOW"}
            ):

                val_score = evaluate_threshold(
                    val_pos,
                    val_neg,
                    threshold,
                    train_direction,
                )

            fold_aucs.append(
                auc
            )

            fold_strengths.append(
                strength
            )

            fold_thresholds.append(
                threshold
            )

            fold_val_scores.append(
                val_score
            )

            fold_directions.append(
                direction
            )

        valid_strengths = [
            x
            for x in fold_strengths
            if math.isfinite(x)
        ]

        valid_aucs = [
            x
            for x in fold_aucs
            if math.isfinite(x)
        ]

        valid_thresholds = [
            x
            for x in fold_thresholds
            if math.isfinite(x)
        ]

        valid_val_scores = [
            x
            for x in fold_val_scores
            if math.isfinite(x)
        ]

        mean_strength = mean(
            valid_strengths
        )

        std_strength = std(
            valid_strengths
        )

        worst_strength = (
            min(valid_strengths)
            if valid_strengths
            else float("nan")
        )

        mean_auc = mean(
            valid_aucs
        )

        mean_threshold = mean(
            valid_thresholds
        )

        threshold_std = std(
            valid_thresholds
        )

        mean_val_score = mean(
            valid_val_scores
        )

        # -------------------------------------------------------------------
        # Direction stability
        # -------------------------------------------------------------------

        valid_directions = [
            d
            for d in fold_directions
            if d in {"HIGH", "LOW"}
        ]

        direction = "UNKNOWN"

        direction_stability = 0.0

        if valid_directions:

            high_count = (
                valid_directions.count(
                    "HIGH"
                )
            )

            low_count = (
                valid_directions.count(
                    "LOW"
                )
            )

            if high_count >= low_count:

                direction = "HIGH"

                direction_stability = (
                    high_count
                    / len(
                        valid_directions
                    )
                )

            else:

                direction = "LOW"

                direction_stability = (
                    low_count
                    / len(
                        valid_directions
                    )
                )

        return_row = {

            "class": target_class,

            "feature": feature_name,

            "n_folds":
                n_folds,

            "mean_auc":
                mean_auc,

            "mean_strength":
                mean_strength,

            "std_strength":
                std_strength,

            "worst_strength":
                worst_strength,

            "mean_threshold":
                mean_threshold,

            "threshold_std":
                threshold_std,

            "mean_validation_balanced_accuracy":
                mean_val_score,

            "direction":
                direction,

            "direction_stability":
                direction_stability,

        }

        # ---------------------------------------------------------------
        # fold details
        # ---------------------------------------------------------------

        for i in range(n_folds):

            return_row[
                f"fold_{i+1}_auc"
            ] = fold_aucs[i]

            return_row[
                f"fold_{i+1}_strength"
            ] = fold_strengths[i]

            return_row[
                f"fold_{i+1}_threshold"
            ] = fold_thresholds[i]

            return_row[
                f"fold_{i+1}_val_balanced_accuracy"
            ] = fold_val_scores[i]

            return_row[
                f"fold_{i+1}_direction"
            ] = fold_directions[i]

        rows.append(
            return_row
        )

    return rows


# ============================================================================
# Pairwise analysis
# ============================================================================

def pairwise_feature_analysis(
    records,
    class_a,
    class_b,
    feature_name,
):

    values_a = []

    values_b = []

    for record in records:

        if record["class"] not in {
            class_a,
            class_b,
        }:

            continue

        value = record[
            "features"
        ].get(feature_name)

        if value is None:

            continue

        if (
            record["class"]
            == class_a
        ):

            values_a.append(value)

        else:

            values_b.append(value)

    auc = auc_one_vs_rest(
        values_a,
        values_b,
    )

    return {

        "class_a":
            class_a,

        "class_b":
            class_b,

        "feature":
            feature_name,

        "n_a":
            len(values_a),

        "n_b":
            len(values_b),

        "auc_a_vs_b":
            auc,

        "strength":
            auc_strength(auc),

        "direction_for_a":
            auc_direction(auc),
    }


def pairwise_5fold_analysis(
    records,
    class_names,
    feature_name,
    n_folds=N_FOLDS,
    seed=RANDOM_SEED,
):

    folds = make_stratified_folds(
        records,
        class_names,
        n_folds,
        seed,
    )

    rows = []

    for i, class_a in enumerate(
        class_names
    ):

        for class_b in class_names[
            i + 1:
        ]:

            fold_strengths = []
            fold_aucs = []
            fold_directions = []

            for fold_index in range(
                n_folds
            ):

                validation = folds[
                    fold_index
                ]

                values_a = []
                values_b = []

                for record in validation:

                    if record[
                        "class"
                    ] == class_a:

                        value = record[
                            "features"
                        ].get(
                            feature_name
                        )

                        if value is not None:

                            values_a.append(
                                value
                            )

                    elif record[
                        "class"
                    ] == class_b:

                        value = record[
                            "features"
                        ].get(
                            feature_name
                        )

                        if value is not None:

                            values_b.append(
                                value
                            )

                auc = auc_one_vs_rest(
                    values_a,
                    values_b,
                )

                strength = auc_strength(
                    auc
                )

                direction = auc_direction(
                    auc
                )

                fold_aucs.append(
                    auc
                )

                fold_strengths.append(
                    strength
                )

                fold_directions.append(
                    direction
                )

            valid_strengths = [
                x
                for x in fold_strengths
                if math.isfinite(x)
            ]

            valid_aucs = [
                x
                for x in fold_aucs
                if math.isfinite(x)
            ]

            valid_directions = [
                x
                for x in fold_directions
                if x in {"HIGH", "LOW"}
            ]

            mean_strength = mean(
                valid_strengths
            )

            std_strength = std(
                valid_strengths
            )

            worst_strength = (
                min(valid_strengths)
                if valid_strengths
                else float("nan")
            )

            mean_auc = mean(
                valid_aucs
            )

            direction = "UNKNOWN"

            direction_stability = 0.0

            if valid_directions:

                high_count = (
                    valid_directions.count(
                        "HIGH"
                    )
                )

                low_count = (
                    valid_directions.count(
                        "LOW"
                    )
                )

                if high_count >= low_count:

                    direction = "HIGH"

                    direction_stability = (
                        high_count
                        / len(
                            valid_directions
                        )
                    )

                else:

                    direction = "LOW"

                    direction_stability = (
                        low_count
                        / len(
                            valid_directions
                        )
                    )

            row = {

                "class_a":
                    class_a,

                "class_b":
                    class_b,

                "feature":
                    feature_name,

                "mean_auc":
                    mean_auc,

                "mean_strength":
                    mean_strength,

                "std_strength":
                    std_strength,

                "worst_strength":
                    worst_strength,

                "direction":
                    direction,

                "direction_stability":
                    direction_stability,
            }

            for fold_index in range(
                n_folds
            ):

                row[
                    f"fold_{fold_index+1}_auc"
                ] = fold_aucs[
                    fold_index
                ]

                row[
                    f"fold_{fold_index+1}_strength"
                ] = fold_strengths[
                    fold_index
                ]

                row[
                    f"fold_{fold_index+1}_direction"
                ] = fold_directions[
                    fold_index
                ]

            rows.append(row)

    return rows


# ============================================================================
# Evidence candidate generation
# ============================================================================

def candidate_level(
    mean_strength,
    worst_strength,
    stability,
):

    if not all(
        math.isfinite(x)
        for x in [
            mean_strength,
            worst_strength,
            stability,
        ]
    ):

        return "DROP"

    if (
        mean_strength
        >= MIN_MEAN_STRENGTH_CORE
        and worst_strength
        >= MIN_WORST_STRENGTH_CORE
        and stability
        >= MIN_STABILITY_CORE
    ):

        return "CORE"

    if (
        mean_strength
        >= MIN_MEAN_STRENGTH_SECONDARY
        and worst_strength
        >= MIN_WORST_STRENGTH_SECONDARY
        and stability
        >= MIN_STABILITY_SECONDARY
    ):

        return "SECONDARY"

    return "DROP"


def generate_evidence_candidates(
    stability_rows,
    pairwise_rows,
    class_names,
):

    candidates = []

    # ------------------------------------------------------------------------
    # Pairwise map
    # ------------------------------------------------------------------------

    pair_strength_map = defaultdict(list)

    for row in pairwise_rows:

        key = (
            row["class_a"],
            row["class_b"],
            row["feature"],
        )

        pair_strength_map[
            key
        ].append(
            row["mean_strength"]
        )

    # ------------------------------------------------------------------------
    # Class-wise
    # ------------------------------------------------------------------------

    for class_name in class_names:

        rows = [
            r
            for r in stability_rows
            if r["class"]
            == class_name
        ]

        for row in rows:

            mean_strength = safe_float(
                row["mean_strength"]
            )

            worst_strength = safe_float(
                row["worst_strength"]
            )

            stability = safe_float(
                row[
                    "direction_stability"
                ]
            )

            level = candidate_level(
                mean_strength,
                worst_strength,
                stability,
            )

            # ---------------------------------------------------------------
            # Pairwise support
            # ---------------------------------------------------------------

            pair_values = []

            for other_class in class_names:

                if (
                    other_class
                    == class_name
                ):

                    continue

                pair_key = (
                    class_name,
                    other_class,
                    row["feature"],
                )

                reverse_key = (
                    other_class,
                    class_name,
                    row["feature"],
                )

                if pair_key in pair_strength_map:

                    pair_values.extend(
                        pair_strength_map[
                            pair_key
                        ]
                    )

                elif (
                    reverse_key
                    in pair_strength_map
                ):

                    pair_values.extend(
                        pair_strength_map[
                            reverse_key
                        ]
                    )

            pairwise_mean = mean(
                pair_values
            )

            pairwise_min = (
                min(pair_values)
                if pair_values
                else float("nan")
            )

            # ---------------------------------------------------------------
            # Final score
            # ---------------------------------------------------------------

            components = []

            if math.isfinite(
                mean_strength
            ):

                components.append(
                    mean_strength
                )

            if math.isfinite(
                worst_strength
            ):

                components.append(
                    worst_strength
                )

            if math.isfinite(
                stability
            ):

                components.append(
                    stability
                )

            if math.isfinite(
                pairwise_mean
            ):

                components.append(
                    pairwise_mean
                )

            final_score = (
                mean(components)
                if components
                else float("nan")
            )

            candidates.append(
                {

                    "class":
                        class_name,

                    "feature":
                        row["feature"],

                    "recommendation":
                        level,

                    "final_score":
                        final_score,

                    "mean_strength":
                        mean_strength,

                    "worst_strength":
                        worst_strength,

                    "direction_stability":
                        stability,

                    "direction":
                        row["direction"],

                    "mean_threshold":
                        row[
                            "mean_threshold"
                        ],

                    "threshold_std":
                        row[
                            "threshold_std"
                        ],

                    "mean_validation_balanced_accuracy":
                        row[
                            "mean_validation_balanced_accuracy"
                        ],

                    "pairwise_mean_strength":
                        pairwise_mean,

                    "pairwise_worst_strength":
                        pairwise_min,
                }
            )

    candidates.sort(
        key=lambda r: (
            r["class"],
            -(
                r["final_score"]
                if math.isfinite(
                    float(
                        r["final_score"]
                    )
                )
                else -1
            ),
        )
    )

    return candidates


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

        writer.writerows(
            rows
        )

    print(
        f"[saved] {path}"
    )


# ============================================================================
# SVG helper
# ============================================================================

def write_svg(
    path: Path,
    svg_content: str,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        svg_content,
        encoding="utf-8",
    )

    print(
        f"[saved] {path}"
    )


# ============================================================================
# SVG: Feature Stability
# ============================================================================

def create_feature_stability_svg(
    stability_rows,
    class_names,
    output_path,
    top_n=12,
):

    width = 1200
    row_height = 24
    left = 280
    top = 60

    rows = []

    for class_name in class_names:

        class_rows = [
            r
            for r in stability_rows
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

        rows.extend(
            class_rows[
                :top_n
            ]
        )

    height = max(
        200,
        top
        + len(rows)
        * row_height
        + 60,
    )

    svg = []

    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}">'
    )

    svg.append(
        f'<rect width="{width}" '
        f'height="{height}" fill="white"/>'
    )

    svg.append(
        '<text x="20" y="32" '
        'font-size="20" '
        'font-family="Arial">'
        'Feature Stability — Mean Strength'
        '</text>'
    )

    y = top

    for row in rows:

        strength = safe_float(
            row["mean_strength"]
        )

        stability = safe_float(
            row[
                "direction_stability"
            ]
        )

        if not math.isfinite(
            strength
        ):

            strength = 0

        if not math.isfinite(
            stability
        ):

            stability = 0

        bar_width = (
            max(
                0,
                min(
                    1,
                    strength,
                ),
            )
            * 600
        )

        svg.append(
            f'<text x="20" y="{y+16}" '
            f'font-size="13" '
            f'font-family="Arial">'
            f'{row["class"]} / '
            f'{row["feature"]}'
            f'</text>'
        )

        svg.append(
            f'<rect x="{left}" '
            f'y="{y+3}" '
            f'width="{bar_width:.1f}" '
            f'height="16" '
            f'fill="#4C78A8"/>'
        )

        svg.append(
            f'<text x="{left+620}" '
            f'y="{y+16}" '
            f'font-size="12" '
            f'font-family="Arial">'
            f'S={strength:.3f} '
            f'Stable={stability:.2f}'
            f'</text>'
        )

        y += row_height

    svg.append(
        '</svg>'
    )

    write_svg(
        output_path,
        "\n".join(svg),
    )


# ============================================================================
# SVG: Pairwise Stability
# ============================================================================

def create_pairwise_svg(
    pairwise_rows,
    class_names,
    output_path,
    top_n=8,
):

    width = 1300

    pair_names = []

    for i, class_a in enumerate(
        class_names
    ):

        for class_b in class_names[
            i + 1:
        ]:

            pair_names.append(
                (
                    class_a,
                    class_b,
                )
            )

    row_height = 22

    total_rows = (
        len(pair_names)
        * top_n
    )

    height = max(
        250,
        70
        + total_rows
        * row_height,
    )

    svg = []

    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}">'
    )

    svg.append(
        f'<rect width="{width}" '
        f'height="{height}" '
        f'fill="white"/>'
    )

    svg.append(
        '<text x="20" y="32" '
        'font-size="20" '
        'font-family="Arial">'
        'Pairwise Feature Stability'
        '</text>'
    )

    y = 55

    for class_a, class_b in pair_names:

        rows = [
            r
            for r in pairwise_rows
            if (
                r["class_a"]
                == class_a
                and
                r["class_b"]
                == class_b
            )
        ]

        rows.sort(
            key=lambda r: (
                float(
                    r[
                        "mean_strength"
                    ]
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

        svg.append(
            f'<text x="20" y="{y+15}" '
            f'font-size="15" '
            f'font-family="Arial">'
            f'{class_a} VS {class_b}'
            f'</text>'
        )

        y += 22

        for row in rows[:top_n]:

            strength = safe_float(
                row[
                    "mean_strength"
                ]
            )

            stability = safe_float(
                row[
                    "direction_stability"
                ]
            )

            if not math.isfinite(
                strength
            ):

                strength = 0

            if not math.isfinite(
                stability
            ):

                stability = 0

            bar_width = (
                max(
                    0,
                    min(
                        1,
                        strength,
                    ),
                )
                * 500
            )

            svg.append(
                f'<text x="40" '
                f'y="{y+14}" '
                f'font-size="12" '
                f'font-family="Arial">'
                f'{row["feature"]}'
                f'</text>'
            )

            svg.append(
                f'<rect x="350" '
                f'y="{y+2}" '
                f'width="{bar_width:.1f}" '
                f'height="14" '
                f'fill="#F58518"/>'
            )

            svg.append(
                f'<text x="870" '
                f'y="{y+14}" '
                f'font-size="12" '
                f'font-family="Arial">'
                f'S={strength:.3f} '
                f'Stable={stability:.2f}'
                f'</text>'
            )

            y += row_height

        y += 12

    svg.append(
        '</svg>'
    )

    write_svg(
        output_path,
        "\n".join(svg),
    )


# ============================================================================
# SVG: Evidence Matrix
# ============================================================================

def create_evidence_matrix_svg(
    candidates,
    class_names,
    output_path,
):

    features = sorted(
        {
            row["feature"]
            for row in candidates
        }
    )

    width = 1200

    cell_w = 85
    cell_h = 28

    left = 280
    top = 90

    height = (
        top
        + len(class_names)
        * cell_h
        + 80
    )

    width = max(
        width,
        left
        + len(features)
        * cell_w
        + 100,
    )

    svg = []

    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}">'
    )

    svg.append(
        f'<rect width="{width}" '
        f'height="{height}" '
        f'fill="white"/>'
    )

    svg.append(
        '<text x="20" y="32" '
        'font-size="20" '
        'font-family="Arial">'
        'Verifier Evidence Candidate Matrix'
        '</text>'
    )

    # feature headers

    for j, feature in enumerate(
        features
    ):

        x = left + j * cell_w

        svg.append(
            f'<text x="{x}" y="75" '
            f'font-size="10" '
            f'font-family="Arial" '
            f'transform="rotate(-55 {x} 75)">'
            f'{feature}'
            f'</text>'
        )

    level_value = {
        "DROP": 0,
        "SECONDARY": 1,
        "CORE": 2,
    }

    for i, class_name in enumerate(
        class_names
    ):

        y = top + i * cell_h

        svg.append(
            f'<text x="10" '
            f'y="{y+19}" '
            f'font-size="12" '
            f'font-family="Arial">'
            f'{class_name}'
            f'</text>'
        )

        for j, feature in enumerate(
            features
        ):

            row = next(
                (
                    r
                    for r in candidates
                    if (
                        r["class"]
                        == class_name
                        and
                        r["feature"]
                        == feature
                    )
                ),
                None,
            )

            level = (
                row[
                    "recommendation"
                ]
                if row
                else "DROP"
            )

            value = level_value[
                level
            ]

            # monochrome intensity
            gray = int(
                245 - value * 70
            )

            x = left + j * cell_w

            svg.append(
                f'<rect x="{x}" '
                f'y="{y}" '
                f'width="{cell_w-2}" '
                f'height="{cell_h-2}" '
                f'fill="rgb({gray},{gray},{gray})" '
                f'stroke="#BBBBBB"/>'
            )

            label = {
                "CORE": "C",
                "SECONDARY": "S",
                "DROP": "-",
            }[level]

            svg.append(
                f'<text x="{x+32}" '
                f'y="{y+18}" '
                f'font-size="11" '
                f'font-family="Arial">'
                f'{label}'
                f'</text>'
            )

    svg.append(
        '</svg>'
    )

    write_svg(
        output_path,
        "\n".join(svg),
    )


# ============================================================================
# Console reports
# ============================================================================

def print_stability_summary(
    stability_rows,
    class_names,
    top_k=10,
):

    print()
    print("=" * 100)
    print("STEP 2: 5-FOLD FEATURE STABILITY")
    print("=" * 100)

    for class_name in class_names:

        rows = [
            r
            for r in stability_rows
            if r["class"]
            == class_name
        ]

        rows.sort(
            key=lambda r: (
                float(
                    r[
                        "mean_strength"
                    ]
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
            f"--- {class_name} ---"
        )

        print(
            f"{'Feature':<34}"
            f"{'MeanS':>8}"
            f"{'StdS':>8}"
            f"{'Worst':>8}"
            f"{'Stable':>9}"
            f"{'ValBA':>9}"
            f"{'Dir':>7}"
        )

        print("-" * 100)

        for row in rows[:top_k]:

            print(
                f"{str(row['feature']):<34}"
                f"{float(row['mean_strength']):>8.3f}"
                f"{float(row['std_strength']):>8.3f}"
                f"{float(row['worst_strength']):>8.3f}"
                f"{float(row['direction_stability']):>9.2f}"
                f"{float(row['mean_validation_balanced_accuracy']):>9.3f}"
                f"{str(row['direction']):>7}"
            )


def print_candidate_summary(
    candidates,
    class_names,
):

    print()
    print("=" * 100)
    print("STEP 4: VERIFIER EVIDENCE CANDIDATES")
    print("=" * 100)

    for class_name in class_names:

        rows = [
            r
            for r in candidates
            if r["class"]
            == class_name
        ]

        rows.sort(
            key=lambda r: (
                float(
                    r["final_score"]
                )
                if math.isfinite(
                    float(
                        r["final_score"]
                    )
                )
                else -1
            ),
            reverse=True,
        )

        print()
        print(
            f"--- {class_name} ---"
        )

        print(
            f"{'Feature':<34}"
            f"{'Level':>12}"
            f"{'Score':>9}"
            f"{'MeanS':>9}"
            f"{'Worst':>9}"
            f"{'Stable':>9}"
            f"{'PairS':>9}"
        )

        print("-" * 100)

        for row in rows:

            print(
                f"{str(row['feature']):<34}"
                f"{str(row['recommendation']):>12}"
                f"{float(row['final_score']):>9.3f}"
                f"{float(row['mean_strength']):>9.3f}"
                f"{float(row['worst_strength']):>9.3f}"
                f"{float(row['direction_stability']):>9.2f}"
                f"{float(row['pairwise_mean_strength']):>9.3f}"
            )


def print_pairwise_summary(
    pairwise_rows,
    class_names,
    top_k=5,
):

    print()
    print("=" * 100)
    print("STEP 3: PAIRWISE STABILITY")
    print("=" * 100)

    for i, class_a in enumerate(
        class_names
    ):

        for class_b in class_names[
            i + 1:
        ]:

            rows = [
                r
                for r in pairwise_rows
                if (
                    r["class_a"]
                    == class_a
                    and
                    r["class_b"]
                    == class_b
                )
            ]

            rows.sort(
                key=lambda r: (
                    float(
                        r[
                            "mean_strength"
                        ]
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

            for row in rows[:top_k]:

                print(
                    f"  "
                    f"{str(row['feature']):<34}"
                    f"strength="
                    f"{float(row['mean_strength']):.3f} "
                    f"worst="
                    f"{float(row['worst_strength']):.3f} "
                    f"stable="
                    f"{float(row['direction_stability']):.2f}"
                )


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Astronomy Verifier Feature Analyzer V2"
        )
    )

    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=DEFAULT_CLASSIFIER_DIR,
        help=(
            "Folder ที่มี "
            "astro_rule_classifier_new.py"
        ),
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
        help="Folder สำหรับ output",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_CLASS,
        help=(
            "จำนวนภาพสูงสุดต่อ class; "
            "ไม่ใส่ = ใช้ทั้งหมด"
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
        help=(
            "จำนวน feature สูงสุด "
            "สำหรับ pairwise console"
        ),
    )

    parser.add_argument(
        "--svg-top",
        type=int,
        default=12,
        help=(
            "จำนวน feature ต่อ class "
            "สำหรับ SVG"
        ),
    )

    args = parser.parse_args()

    classifier_dir = (
        args.classifier_dir
    )

    dataset_dir = (
        args.dataset
    )

    if args.output_dir is None:

        output_dir = (
            classifier_dir
            / "verifier_analysis_v2"
        )

    else:

        output_dir = (
            args.output_dir
        )

    if not classifier_dir.exists():

        raise SystemExit(
            "[ERROR] ไม่พบ classifier directory:\n"
            f"{classifier_dir}"
        )

    if not dataset_dir.exists():

        raise SystemExit(
            "[ERROR] ไม่พบ dataset directory:\n"
            f"{dataset_dir}"
        )

    if args.folds < 2:

        raise SystemExit(
            "[ERROR] folds ต้อง >= 2"
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
    print("=" * 100)
    print(
        "ASTRONOMY VERIFIER FEATURE ANALYZER V2"
    )
    print("=" * 100)

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
    # STEP 1
    # ------------------------------------------------------------------------

    (
        records,
        image_counts,
        error_counts,
    ) = collect_feature_records(
        dataset_dir=dataset_dir,
        class_names=class_names,
        read_gray_image=read_gray_image,
        extract_features=extract_features,
        max_images_per_class=args.max_images,
    )

    if not records:

        raise SystemExit(
            "[ERROR] ไม่พบ feature records"
        )

    all_features = get_all_features(
        records
    )

    print(
        f"\nจำนวน feature = "
        f"{len(all_features)}"
    )

    # ------------------------------------------------------------------------
    # Original-style distribution CSV
    # ------------------------------------------------------------------------

    distribution_rows = []

    for class_name in class_names:

        for feature_name in all_features:

            values = []

            for record in records:

                if (
                    record["class"]
                    != class_name
                ):

                    continue

                value = record[
                    "features"
                ].get(feature_name)

                if value is not None:

                    values.append(
                        value
                    )

            if not values:

                continue

            distribution_rows.append(
                {
                    "class":
                        class_name,

                    "feature":
                        feature_name,

                    **compute_distribution(
                        values
                    ),
                }
            )

    write_csv(
        output_dir
        / "feature_distribution.csv",
        distribution_rows,
    )

    # ------------------------------------------------------------------------
    # STEP 2: Full dataset ranking
    # ------------------------------------------------------------------------

    ranking_rows = []

    for class_name in class_names:

        for feature_name in all_features:

            class_values = (
                build_class_feature_values(
                    records,
                    class_names,
                    feature_name,
                )
            )

            row = (
                analyze_feature_for_class(
                    target_class=class_name,
                    feature_name=feature_name,
                    class_values=class_values,
                    class_names=class_names,
                )
            )

            ranking_rows.append(
                row
            )

    write_csv(
        output_dir
        / "verifier_feature_ranking.csv",
        ranking_rows,
    )

    # ------------------------------------------------------------------------
    # STEP 3: 5-fold stability
    # ------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "STEP 2: Running 5-fold stability..."
    )
    print("=" * 100)

    stability_rows = []

    for feature_index, feature_name in enumerate(
        all_features,
        start=1,
    ):

        print(
            f"\r  feature "
            f"{feature_index}/"
            f"{len(all_features)}: "
            f"{feature_name:<40}",
            end="",
        )

        rows = analyze_5fold_feature(
            records=records,
            class_names=class_names,
            feature_name=feature_name,
            n_folds=args.folds,
            seed=args.seed,
        )

        stability_rows.extend(
            rows
        )

    print()

    write_csv(
        output_dir
        / "feature_stability_5fold.csv",
        stability_rows,
    )

    # ------------------------------------------------------------------------
    # STEP 4: Pairwise
    # ------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "STEP 3: Running pairwise stability..."
    )
    print("=" * 100)

    pairwise_rows = []

    for feature_index, feature_name in enumerate(
        all_features,
        start=1,
    ):

        print(
            f"\r  feature "
            f"{feature_index}/"
            f"{len(all_features)}: "
            f"{feature_name:<40}",
            end="",
        )

        rows = pairwise_5fold_analysis(
            records=records,
            class_names=class_names,
            feature_name=feature_name,
            n_folds=args.folds,
            seed=args.seed,
        )

        pairwise_rows.extend(
            rows
        )

    print()

    write_csv(
        output_dir
        / "pairwise_stability_5fold.csv",
        pairwise_rows,
    )

    # ------------------------------------------------------------------------
    # STEP 5: Evidence candidates
    # ------------------------------------------------------------------------

    candidates = (
        generate_evidence_candidates(
            stability_rows=stability_rows,
            pairwise_rows=pairwise_rows,
            class_names=class_names,
        )
    )

    write_csv(
        output_dir
        / "verifier_evidence_candidates.csv",
        candidates,
    )

    # ------------------------------------------------------------------------
    # STEP 6: SVG
    # ------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "STEP 5: Generating SVG reports..."
    )
    print("=" * 100)

    create_feature_stability_svg(
        stability_rows=stability_rows,
        class_names=class_names,
        output_path=(
            output_dir
            / "feature_stability.svg"
        ),
        top_n=args.svg_top,
    )

    create_pairwise_svg(
        pairwise_rows=pairwise_rows,
        class_names=class_names,
        output_path=(
            output_dir
            / "pairwise_stability.svg"
        ),
        top_n=8,
    )

    create_evidence_matrix_svg(
        candidates=candidates,
        class_names=class_names,
        output_path=(
            output_dir
            / "evidence_candidate_matrix.svg"
        ),
    )

    # ------------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------------

    print_stability_summary(
        stability_rows,
        class_names,
        top_k=10,
    )

    print_pairwise_summary(
        pairwise_rows,
        class_names,
        top_k=args.pairwise_top,
    )

    print_candidate_summary(
        candidates,
        class_names,
    )

    # ------------------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("DATASET SUMMARY")
    print("=" * 100)

    print(
        f"{'Class':<25}"
        f"{'Images':>10}"
        f"{'Errors':>10}"
    )

    print("-" * 50)

    for class_name in class_names:

        print(
            f"{class_name:<25}"
            f"{image_counts.get(class_name, 0):>10}"
            f"{error_counts.get(class_name, 0):>10}"
        )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("V2 COMPLETE")
    print("=" * 100)

    print(
        """
Output สำคัญ:

1. feature_distribution.csv
   --------------------------------
   distribution ของ feature ต่อ class

2. verifier_feature_ranking.csv
   --------------------------------
   ranking แบบใช้ข้อมูลทั้งหมด

3. feature_stability_5fold.csv
   --------------------------------
   ตรวจว่า feature stable แค่ไหนใน 5 folds

4. pairwise_stability_5fold.csv
   --------------------------------
   ตรวจ feature สำหรับ class pair
   โดยเฉพาะคู่ที่สับสนกัน

5. verifier_evidence_candidates.csv
   --------------------------------
   *** ไฟล์สำคัญที่สุด ***

   CORE
       เหมาะเป็น evidence หลัก

   SECONDARY
       เหมาะเป็น evidence เสริม

   DROP
       หลักฐานไม่แข็งแรงหรือไม่ stable

6. feature_stability.svg
   --------------------------------
   ภาพรวม feature stability

7. pairwise_stability.svg
   --------------------------------
   feature ที่ใช้แยกแต่ละคู่

8. evidence_candidate_matrix.svg
   --------------------------------
   matrix สำหรับดูว่า class ไหน
   ควรใช้ feature อะไร


หมายเหตุสำคัญ:

V2 นี้ยังไม่สร้าง Rule Verifier จริง

มันทำหน้าที่ "วิเคราะห์หลักฐาน"
ก่อนที่เราจะเอาผลไปออกแบบ test_verifier.py

อย่าเพิ่งเอา CORE ไปเป็น rule โดยตรง
จนกว่าจะตรวจ pairwise และ distribution
ประกอบกันก่อน
"""
    )

    print()
    print(
        "[DONE]"
    )


if __name__ == "__main__":
    main()

