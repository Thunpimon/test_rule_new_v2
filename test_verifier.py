# """
# test_verifier.py

# Astronomy Evidence Verifier Tester
# ===================================

# Architecture:

#     Image
#       |
#       +----------------------+
#       |                      |
#       v                      v
#     CNN                    Feature Extraction
#     predict_onnx()         extract_features()
#       |                      |
#       |                      v
#       |               Distribution Evidence
#       |               Pairwise Evidence
#       |                      |
#       +----------+-----------+
#                  |
#                  v
#           Evidence Verifier
#                  |
#         +--------+---------+
#         |                  |
#         v                  v
#     CNN Prediction      Verifier Verdict
#     (authoritative)     SUPPORTED / WEAK /
#                         AMBIGUOUS / OUT_OF_SCOPE

# IMPORTANT
# ---------
# 1. CNN class remains authoritative.
# 2. Verifier NEVER replaces CNN class.
# 3. Verifier NEVER calls fuse_scores().
# 4. Verifier NEVER calls predict_with_rules().
# 5. Verifier uses actual distributions from feature_distribution.csv.
# 6. Verifier uses pairwise stability from pairwise_feature_stability.csv.
# 7. Verifier focuses on:
#        01_Good
#        02_Out_of_Focus
#        03_Tracking_Error

# Files expected from Phase 1:

#     verifier_analysis_v2/
#         feature_distribution.csv
#         verifier_feature_stability.csv
#         pairwise_feature_stability.csv
#         verifier_evidence_candidates.csv

# Outputs:

#     verifier_test_results.csv
#     verifier_confusion_summary.csv
#     verifier_evidence_summary.csv

# Usage
# -----

# Single image:

# python test_verifier.py --image "D:/images/test.jpg"

# Folder:

# python test_verifier.py --folder "D:/images/test"

# CSV:

# python test_verifier.py --csv "D:/images/test.csv"

# Custom paths:

# python test_verifier.py ^
#     --image "D:/images/test.jpg" ^
#     --classifier "D:/project/astro_rule_classifier_new.py" ^
#     --analysis-dir "D:/project/verifier_analysis_v2" ^
#     --onnx "D:/project/model.onnx"

# CSV format example:

# image,true_class
# D:/images/a.jpg,01_Good
# D:/images/b.jpg,02_Out_of_Focus
# D:/images/c.jpg,03_Tracking_Error

# or:

# image_path,label
# D:/images/a.jpg,01_Good
# ...

# """

# from __future__ import annotations

# import argparse
# import csv
# import importlib.util
# import math
# from dataclasses import asdict, is_dataclass
# from pathlib import Path
# from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
# import sys
# import importlib.util
# from pathlib import Path

# import numpy as np
# import pandas as pd


# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# DEFAULT_CLASSIFIER = Path(
#     r"D:\Internship\Test_Rule\test_rule_new_v2\astro_rule_classifier_new.py"
# )

# DEFAULT_ANALYSIS_DIR = Path(
#     r"D:\Internship\Test_Rule\test_rule_new_v2\verifier_analysis_v2"
# )

# DEFAULT_ONNX = Path(
#     r"D:\Internship\Test_Rule\test_rule_new_v2\eff_b0_kfold_add_focal_r2.onnx"
# )

# TARGET_CLASSES = [
#     "01_Good",
#     "02_Out_of_Focus",
#     "03_Tracking_Error",
# ]

# IMAGE_EXTENSIONS = {
#     ".png",
#     ".jpg",
#     ".jpeg",
#     ".bmp",
#     ".tif",
#     ".tiff",
#     ".fits",
#     ".fit",
#     ".fts",
# }

# # ---------------------------------------------------------------------------
# # Verifier aggregation weights
# #
# # These are NOT feature thresholds such as weak_at / strong_at.
# #
# # They only control how two evidence sources are combined:
# #
# #   class-specific evidence
# #   pairwise evidence
# #
# # Actual feature information still comes from the CSV distributions.
# # ---------------------------------------------------------------------------

# CLASS_SPECIFIC_WEIGHT = 0.50
# PAIRWISE_WEIGHT = 0.50

# # ---------------------------------------------------------------------------
# # Verdict policy
# #
# # These are decision-policy thresholds, not feature thresholds.
# # They can later be optimized in Phase 2.
# # ---------------------------------------------------------------------------

# SUPPORTED_MIN_EVIDENCE = 0.60
# SUPPORTED_MIN_MARGIN = 0.15

# WEAK_MIN_EVIDENCE = 0.40

# AMBIGUOUS_MARGIN = 0.08

# # ---------------------------------------------------------------------------
# # Minimum quality for evidence rows
# #
# # These are intentionally mild because candidate_score already comes from
# # the Phase 1 stability analysis.
# # ---------------------------------------------------------------------------

# MIN_CANDIDATE_SCORE = 0.0
# MIN_DIRECTION_CONSISTENCY = 0.0

# # ---------------------------------------------------------------------------
# # Number of strongest evidence rows printed per image
# # ---------------------------------------------------------------------------

# TOP_EVIDENCE_FEATURES = 8


# # ============================================================================
# # UTILITIES
# # ============================================================================

# def safe_float(value: Any) -> Optional[float]:
#     """Convert value to finite float or return None."""

#     try:
#         x = float(value)
#     except (TypeError, ValueError):
#         return None

#     if not math.isfinite(x):
#         return None

#     return x


# def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
#     """Clamp a numeric value."""

#     if not math.isfinite(value):
#         return low

#     return max(low, min(high, value))


# def sigmoid(x: float) -> float:
#     """Numerically stable sigmoid."""

#     if x >= 0:
#         z = math.exp(-x)
#         return 1.0 / (1.0 + z)

#     z = math.exp(x)
#     return z / (1.0 + z)


# def normalize_name(value: Any) -> str:
#     """Normalize class/feature names."""

#     if value is None:
#         return ""

#     return str(value).strip()


# def is_finite_number(value: Any) -> bool:
#     x = safe_float(value)
#     return x is not None


# def write_csv_rows(
#     path: Path,
#     rows: List[Dict[str, Any]],
# ) -> None:
#     """Write list of dictionaries to UTF-8 CSV."""

#     path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     if not rows:
#         print(f"[WARNING] ไม่มีข้อมูลสำหรับเขียน: {path}")
#         return

#     df = pd.DataFrame(rows)

#     df.to_csv(
#         path,
#         index=False,
#         encoding="utf-8-sig",
#     )

#     print(f"[SAVED] {path.resolve()}")


# # ============================================================================
# # IMPORT CLASSIFIER
# # ============================================================================

# # def import_classifier(
# #     classifier_path: Path,
# # ):
# #     """
# #     Dynamically import astro_rule_classifier_new.py.

# #     We deliberately import only the functions needed by this verifier.
# #     """

# #     classifier_path = classifier_path.resolve()

# #     if not classifier_path.exists():
# #         raise SystemExit(
# #             "\n[ERROR] ไม่พบ classifier:\n"
# #             f"{classifier_path}\n"
# #         )

# #     spec = importlib.util.spec_from_file_location(
# #         "astro_rule_classifier_new",
# #         str(classifier_path),
# #     )

# #     if spec is None or spec.loader is None:
# #         raise SystemExit(
# #             "[ERROR] ไม่สามารถสร้าง import spec สำหรับ classifier ได้"
# #         )

# #     module = importlib.util.module_from_spec(spec)

# #     try:
# #         spec.loader.exec_module(module)
# #     except Exception as exc:
# #         raise SystemExit(
# #             "\n[ERROR] import astro_rule_classifier_new.py ไม่สำเร็จ\n"
# #             f"Path : {classifier_path}\n"
# #             f"Error: {repr(exc)}\n"
# #         ) from exc

# #     required = [
# #         "CLASS_NAMES",
# #         "extract_features",
# #         "read_gray_image",
# #         "predict_onnx",
# #     ]

# #     missing = [
# #         name
# #         for name in required
# #         if not hasattr(module, name)
# #     ]

# #     if missing:
# #         raise SystemExit(
# #             "\n[ERROR] classifier ไม่มี function/variable ที่ต้องใช้:\n"
# #             + "\n".join(f"  - {x}" for x in missing)
# #         )

# #     return module
# # ============================================================================
# # IMPORT CLASSIFIER
# # ============================================================================
# def import_classifier(classifier_path: Path):
#     classifier_path = classifier_path.resolve()

#     if not classifier_path.exists():
#         raise SystemExit(
#             f"\n[ERROR] ไม่พบ classifier:\n{classifier_path}\n"
#         )

#     module_name = "astro_rule_classifier_new"

#     spec = importlib.util.spec_from_file_location(
#         module_name,
#         str(classifier_path)
#     )

#     if spec is None or spec.loader is None:
#         raise SystemExit(
#             "\n[ERROR] ไม่สามารถสร้าง import spec สำหรับ classifier ได้"
#         )

#     module = importlib.util.module_from_spec(spec)

#     # สำคัญมากสำหรับ Python 3.12 + dataclass
#     # ต้อง register module ก่อน exec_module()
#     sys.modules[module_name] = module

#     try:
#         spec.loader.exec_module(module)

#     except Exception as exc:
#         # ลบ module ออกจาก sys.modules หาก import ไม่สำเร็จ
#         sys.modules.pop(module_name, None)

#         raise SystemExit(
#             "\n[ERROR] import astro_rule_classifier_new.py ไม่สำเร็จ\n"
#             f"Path : {classifier_path}\n"
#             f"Error: {repr(exc)}\n"
#         ) from exc

#     required = [
#         "CLASS_NAMES",
#         "extract_features",
#         "read_gray_image",
#         "predict_onnx",
#     ]

#     missing = [
#         name
#         for name in required
#         if not hasattr(module, name)
#     ]

#     if missing:
#         raise SystemExit(
#             "\n[ERROR] classifier ไม่มี function/variable ที่ต้องใช้:\n"
#             + "\n".join(f"  - {x}" for x in missing)
#         )

#     print("[OK] import astro_rule_classifier_new.py สำเร็จ")

#     return module


# # ============================================================================
# # LOAD CSV
# # ============================================================================

# def read_csv_flexible(path: Path) -> pd.DataFrame:
#     """Read CSV with UTF-8 BOM support."""

#     if not path.exists():
#         raise FileNotFoundError(
#             f"ไม่พบไฟล์ CSV: {path}"
#         )

#     try:
#         df = pd.read_csv(
#             path,
#             encoding="utf-8-sig",
#         )
#     except UnicodeDecodeError:
#         df = pd.read_csv(
#             path,
#             encoding="utf-8",
#         )

#     # Normalize column names
#     df.columns = [
#         str(c).strip()
#         for c in df.columns
#     ]

#     return df


# def load_feature_distribution(
#     path: Path,
# ) -> Dict[Tuple[str, str], Dict[str, float]]:
#     """
#     Load:

#         feature_distribution.csv

#     Expected columns:

#         class
#         feature
#         n
#         min
#         p10
#         p25
#         p50
#         p75
#         p90
#         max
#         mean
#         std

#     These exact fields are generated by distribution_row() in the analyzer.
#     """

#     print()
#     print("=" * 90)
#     print("LOAD FEATURE DISTRIBUTION")
#     print("=" * 90)

#     df = read_csv_flexible(path)

#     required = {
#         "class",
#         "feature",
#         "p10",
#         "p25",
#         "p50",
#         "p75",
#         "p90",
#     }

#     missing = required - set(df.columns)

#     if missing:
#         raise SystemExit(
#             "\n[ERROR] feature_distribution.csv ขาด columns:\n"
#             + "\n".join(
#                 f"  - {x}"
#                 for x in sorted(missing)
#             )
#         )

#     distributions: Dict[
#         Tuple[str, str],
#         Dict[str, float]
#     ] = {}

#     for _, row in df.iterrows():

#         class_name = normalize_name(
#             row["class"]
#         )

#         feature_name = normalize_name(
#             row["feature"]
#         )

#         if not class_name or not feature_name:
#             continue

#         stats: Dict[str, float] = {}

#         for column in [
#             "n",
#             "min",
#             "p10",
#             "p25",
#             "p50",
#             "p75",
#             "p90",
#             "max",
#             "mean",
#             "std",
#         ]:
#             if column in df.columns:
#                 value = safe_float(
#                     row[column]
#                 )

#                 if value is not None:
#                     stats[column] = value

#         # Must have percentile information
#         required_percentiles = [
#             "p10",
#             "p25",
#             "p50",
#             "p75",
#             "p90",
#         ]

#         if not all(
#             x in stats
#             for x in required_percentiles
#         ):
#             continue

#         distributions[
#             (class_name, feature_name)
#         ] = stats

#     print(
#         f"[INFO] loaded distributions: "
#         f"{len(distributions)} rows"
#     )

#     return distributions


# def load_stability_csv(
#     path: Path,
# ) -> pd.DataFrame:
#     """
#     Load verifier_feature_stability.csv.

#     This is the one-vs-rest stability table.
#     """

#     print()
#     print("=" * 90)
#     print("LOAD VERIFIER FEATURE STABILITY")
#     print("=" * 90)

#     df = read_csv_flexible(path)

#     print(
#         f"[INFO] rows: {len(df)}"
#     )

#     return df


# def load_pairwise_stability(
#     path: Path,
# ) -> pd.DataFrame:
#     """
#     Load pairwise_feature_stability.csv.

#     Expected important columns:

#         class_a
#         class_b
#         feature
#         mean_strength
#         min_strength
#         std_strength
#         direction
#         direction_consistency
#         candidate_score
#     """

#     print()
#     print("=" * 90)
#     print("LOAD PAIRWISE FEATURE STABILITY")
#     print("=" * 90)

#     df = read_csv_flexible(path)

#     required = {
#         "class_a",
#         "class_b",
#         "feature",
#         "direction",
#     }

#     missing = required - set(df.columns)

#     if missing:
#         raise SystemExit(
#             "\n[ERROR] pairwise_feature_stability.csv "
#             "ขาด columns:\n"
#             + "\n".join(
#                 f"  - {x}"
#                 for x in sorted(missing)
#             )
#         )

#     print(
#         f"[INFO] rows: {len(df)}"
#     )

#     return df


# def load_candidates(
#     path: Path,
# ) -> Optional[pd.DataFrame]:
#     """
#     Load verifier_evidence_candidates.csv if available.

#     This file is optional.

#     Important:
#         candidate / candidate_score are screening information.
#         They are NOT weak_at / strong_at thresholds.
#     """

#     if not path.exists():
#         print(
#             f"[WARNING] ไม่พบ candidate file:\n"
#             f"          {path}\n"
#             f"          จะใช้ stability CSV โดยตรง"
#         )
#         return None

#     df = read_csv_flexible(path)

#     print()
#     print("=" * 90)
#     print("LOAD VERIFIER EVIDENCE CANDIDATES")
#     print("=" * 90)

#     print(
#         f"[INFO] rows: {len(df)}"
#     )

#     return df


# # ============================================================================
# # FEATURE EXTRACTION
# # ============================================================================

# def feature_dict_from_dataclass(
#     features: Any,
# ) -> Dict[str, Any]:
#     """Convert AstroFeatures dataclass to dictionary."""

#     if is_dataclass(features):
#         return asdict(features)

#     if hasattr(features, "__dict__"):
#         return dict(features.__dict__)

#     raise TypeError(
#         "extract_features() ไม่ได้คืน dataclass/object "
#         "ที่สามารถแปลงเป็น dictionary ได้"
#     )


# def extract_image_features(
#     image_path: Path,
#     classifier,
# ) -> Dict[str, Any]:
#     """
#     Extract actual feature values from an image.
#     """

#     gray = classifier.read_gray_image(
#         str(image_path)
#     )

#     features = classifier.extract_features(
#         gray
#     )

#     return feature_dict_from_dataclass(
#         features
#     )


# # ============================================================================
# # CNN
# # ============================================================================

# def run_cnn(
#     image_path: Path,
#     onnx_path: Path,
#     classifier,
# ) -> Dict[str, Any]:
#     """
#     Run CNN directly.

#     IMPORTANT:
#         We intentionally call predict_onnx()
#         and NOT predict_with_rules().
#     """

#     if not onnx_path.exists():
#         raise FileNotFoundError(
#             f"ไม่พบ ONNX model: {onnx_path}"
#         )

#     model_scores = classifier.predict_onnx(
#         str(image_path),
#         str(onnx_path),
#     )

#     if not isinstance(
#         model_scores,
#         dict,
#     ):
#         raise TypeError(
#             "predict_onnx() ต้องคืน Dict[str, float]"
#         )

#     cleaned_scores: Dict[str, float] = {}

#     for class_name, score in model_scores.items():

#         value = safe_float(score)

#         if value is None:
#             continue

#         cleaned_scores[
#             normalize_name(class_name)
#         ] = value

#     if not cleaned_scores:
#         raise RuntimeError(
#             "CNN ไม่คืน probability ที่ใช้งานได้"
#         )

#     cnn_class = max(
#         cleaned_scores,
#         key=cleaned_scores.get,
#     )

#     cnn_confidence = cleaned_scores[
#         cnn_class
#     ]

#     return {
#         "cnn_class": cnn_class,
#         "cnn_confidence": cnn_confidence,
#         "cnn_scores": cleaned_scores,
#     }


# # ============================================================================
# # QUANTILE / DISTRIBUTION FUNCTIONS
# # ============================================================================

# QUANTILE_POINTS = np.array(
#     [
#         0.10,
#         0.25,
#         0.50,
#         0.75,
#         0.90,
#     ],
#     dtype=float,
# )


# def clean_quantiles(
#     stats: Dict[str, float],
# ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
#     """
#     Return monotonic quantile points.

#     Because real-world feature distributions can occasionally have
#     repeated percentile values, we make the x values non-decreasing.
#     """

#     keys = [
#         "p10",
#         "p25",
#         "p50",
#         "p75",
#         "p90",
#     ]

#     values = []

#     for key in keys:

#         if key not in stats:
#             return None

#         value = safe_float(
#             stats[key]
#         )

#         if value is None:
#             return None

#         values.append(value)

#     values = np.asarray(
#         values,
#         dtype=float,
#     )

#     # Percentiles should be monotonic.
#     values = np.maximum.accumulate(
#         values
#     )

#     return (
#         QUANTILE_POINTS.copy(),
#         values,
#     )


# def percentile_position(
#     value: float,
#     stats: Dict[str, float],
# ) -> Optional[float]:
#     """
#     Estimate percentile position of a value
#     using p10/p25/p50/p75/p90.

#     Output:
#         0.0 -> very low
#         0.5 -> around median
#         1.0 -> very high

#     Values outside p10-p90 are extrapolated mildly
#     and then clipped to [0, 1].
#     """

#     value = safe_float(value)

#     if value is None:
#         return None

#     result = clean_quantiles(
#         stats
#     )

#     if result is None:
#         return None

#     qs, xs = result

#     # If all percentile values are identical,
#     # distribution provides no useful positional information.
#     if np.allclose(
#         xs,
#         xs[0],
#     ):
#         return 0.5

#     # np.interp clips outside the range.
#     # We deliberately keep the result within p10-p90
#     # for robust evidence.
#     q = float(
#         np.interp(
#             value,
#             xs,
#             qs,
#         )
#     )

#     return clamp(q)


# def directional_support(
#     q: float,
#     direction: str,
# ) -> float:
#     """
#     Convert percentile position to directional evidence.

#     HIGH:
#         high value supports class

#     LOW:
#         low value supports class

#     NONE / UNKNOWN:
#         neutral = 0.5
#     """

#     direction = normalize_name(
#         direction
#     ).upper()

#     q = clamp(q)

#     if direction == "HIGH":
#         return q

#     if direction == "LOW":
#         return 1.0 - q

#     return 0.5


# def typicality_score(
#     q: float,
# ) -> float:
#     """
#     Score how typical a value is around the class median.

#     q = 0.5 -> 1.0
#     q = 0.1/0.9 -> 0.2
#     q = 0.0/1.0 -> 0.0

#     This is NOT a learned threshold.
#     It is a normalized position inside the observed distribution.
#     """

#     q = clamp(q)

#     return clamp(
#         1.0 - 2.0 * abs(q - 0.5)
#     )


# def distribution_compatibility(
#     value: float,
#     stats: Dict[str, float],
#     direction: str,
# ) -> Optional[Dict[str, float]]:
#     """
#     Calculate evidence from one class distribution.
#     """

#     q = percentile_position(
#         value,
#         stats,
#     )

#     if q is None:
#         return None

#     directional = directional_support(
#         q,
#         direction,
#     )

#     typicality = typicality_score(
#         q
#     )

#     # Direction is more useful than mere typicality
#     # when direction is known.
#     direction_upper = normalize_name(
#         direction
#     ).upper()

#     if direction_upper in {
#         "HIGH",
#         "LOW",
#     }:
#         evidence = (
#             0.70 * directional
#             + 0.30 * typicality
#         )
#     else:
#         evidence = typicality

#     return {
#         "percentile": q,
#         "directional_support": directional,
#         "typicality": typicality,
#         "evidence": clamp(evidence),
#     }


# # ============================================================================
# # CANDIDATE WEIGHTS
# # ============================================================================

# def candidate_weight_from_row(
#     row: pd.Series,
# ) -> float:
#     """
#     Calculate evidence weight.

#     candidate_score is the preferred weight.

#     If candidate_score is unavailable,
#     reconstruct a conservative score from:

#         mean_strength
#         min_strength
#         direction_consistency
#         std_strength
#     """

#     candidate_score = (
#         safe_float(
#             row.get(
#                 "candidate_score"
#             )
#         )
#     )

#     if (
#         candidate_score is not None
#         and candidate_score >= MIN_CANDIDATE_SCORE
#     ):
#         return clamp(
#             candidate_score
#         )

#     mean_strength = safe_float(
#         row.get(
#             "mean_strength"
#         )
#     )

#     min_strength = safe_float(
#         row.get(
#             "min_strength"
#         )
#     )

#     direction_consistency = safe_float(
#         row.get(
#             "direction_consistency"
#         )
#     )

#     std_strength = safe_float(
#         row.get(
#             "std_strength"
#         )
#     )

#     if mean_strength is None:
#         mean_strength = 0.0

#     if min_strength is None:
#         min_strength = mean_strength

#     if direction_consistency is None:
#         direction_consistency = 0.0

#     if std_strength is None:
#         std_strength = 1.0

#     stability_bonus = max(
#         0.0,
#         1.0 - std_strength,
#     )

#     score = (
#         0.45 * mean_strength
#         + 0.30 * min_strength
#         + 0.15 * direction_consistency
#         + 0.10 * stability_bonus
#     )

#     return clamp(score)


# # ============================================================================
# # CLASS-SPECIFIC EVIDENCE
# # ============================================================================

# def build_class_specific_rows(
#     stability_df: pd.DataFrame,
#     candidates_df: Optional[pd.DataFrame],
# ) -> pd.DataFrame:
#     """
#     Prepare one-vs-rest evidence rows.

#     Preferred source:
#         verifier_evidence_candidates.csv

#     Fallback:
#         verifier_feature_stability.csv
#     """

#     if candidates_df is not None:

#         required = {
#             "candidate_type",
#             "class",
#             "feature",
#             "direction",
#         }

#         if required.issubset(
#             set(candidates_df.columns)
#         ):

#             df = candidates_df.copy()

#             df = df[
#                 df["candidate_type"]
#                 == "CLASS_SPECIFIC"
#             ].copy()

#             df = df[
#                 df["class"].isin(
#                     TARGET_CLASSES
#                 )
#             ].copy()

#             if not df.empty:
#                 return df

#     # Fallback to verifier_feature_stability.csv

#     df = stability_df.copy()

#     required = {
#         "class",
#         "feature",
#         "direction",
#     }

#     missing = required - set(
#         df.columns
#     )

#     if missing:
#         raise SystemExit(
#             "\n[ERROR] verifier_feature_stability.csv "
#             "ขาด columns:\n"
#             + "\n".join(
#                 f"  - {x}"
#                 for x in sorted(missing)
#             )
#         )

#     df = df[
#         df["class"].isin(
#             TARGET_CLASSES
#         )
#     ].copy()

#     return df


# def calculate_class_specific_evidence(
#     class_name: str,
#     feature_values: Dict[str, Any],
#     class_specific_df: pd.DataFrame,
#     distributions: Dict[
#         Tuple[str, str],
#         Dict[str, float]
#     ],
# ) -> Tuple[float, List[Dict[str, Any]]]:
#     """
#     Calculate one-vs-rest evidence for a target class.
#     """

#     rows: List[Dict[str, Any]] = []

#     subset = class_specific_df[
#         class_specific_df["class"]
#         == class_name
#     ]

#     weighted_sum = 0.0
#     weight_sum = 0.0

#     for _, row in subset.iterrows():

#         feature = normalize_name(
#             row.get("feature")
#         )

#         if feature not in feature_values:
#             continue

#         value = safe_float(
#             feature_values.get(feature)
#         )

#         if value is None:
#             continue

#         direction = normalize_name(
#             row.get("direction")
#         ).upper()

#         distribution = distributions.get(
#             (
#                 class_name,
#                 feature,
#             )
#         )

#         if distribution is None:
#             continue

#         result = distribution_compatibility(
#             value,
#             distribution,
#             direction,
#         )

#         if result is None:
#             continue

#         weight = candidate_weight_from_row(
#             row
#         )

#         if weight <= 0:
#             continue

#         weighted_sum += (
#             result["evidence"]
#             * weight
#         )

#         weight_sum += weight

#         rows.append({
#             "class": class_name,
#             "feature": feature,
#             "value": value,
#             "direction": direction,
#             "percentile": result[
#                 "percentile"
#             ],
#             "directional_support": result[
#                 "directional_support"
#             ],
#             "typicality": result[
#                 "typicality"
#             ],
#             "evidence": result[
#                 "evidence"
#             ],
#             "weight": weight,
#             "weighted_evidence": (
#                 result["evidence"]
#                 * weight
#             ),
#         })

#     if weight_sum <= 0:
#         return 0.5, rows

#     evidence = weighted_sum / weight_sum

#     return clamp(evidence), rows


# # ============================================================================
# # PAIRWISE EVIDENCE
# # ============================================================================

# def pairwise_feature_score(
#     value: float,
#     class_a: str,
#     class_b: str,
#     feature: str,
#     direction: str,
#     distributions: Dict[
#         Tuple[str, str],
#         Dict[str, float]
#     ],
# ) -> Optional[Dict[str, float]]:
#     """
#     Calculate pairwise evidence.

#     For each class:

#         q_a = percentile position within class A
#         q_b = percentile position within class B

#     Then compare q_a and q_b.

#     HIGH:
#         larger observed value should support class A.

#     LOW:
#         smaller observed value should support class A.

#     The difference is passed through sigmoid to obtain [0, 1].

#     0.5 = ambiguous
#     >0.5 = supports A
#     <0.5 = supports B
#     """

#     stats_a = distributions.get(
#         (
#             class_a,
#             feature,
#         )
#     )

#     stats_b = distributions.get(
#         (
#             class_b,
#             feature,
#         )
#     )

#     if (
#         stats_a is None
#         or stats_b is None
#     ):
#         return None

#     q_a = percentile_position(
#         value,
#         stats_a,
#     )

#     q_b = percentile_position(
#         value,
#         stats_b,
#     )

#     if (
#         q_a is None
#         or q_b is None
#     ):
#         return None

#     direction = normalize_name(
#         direction
#     ).upper()

#     # For HIGH, larger raw values should support A.
#     #
#     # For LOW, smaller raw values should support A.
#     #
#     # The percentile difference works as a normalized comparison.
#     if direction == "HIGH":
#         delta = q_a - q_b

#     elif direction == "LOW":
#         delta = q_b - q_a

#     else:
#         return None

#     score_a = sigmoid(
#         5.0 * delta
#     )

#     return {
#         "q_a": q_a,
#         "q_b": q_b,
#         "delta": delta,
#         "score_a": clamp(score_a),
#         "score_b": clamp(1.0 - score_a),
#     }


# def calculate_pairwise_evidence(
#     feature_values: Dict[str, Any],
#     pairwise_df: pd.DataFrame,
#     distributions: Dict[
#         Tuple[str, str],
#         Dict[str, float]
#     ],
# ) -> Tuple[
#     Dict[str, float],
#     List[Dict[str, Any]]
# ]:
#     """
#     Calculate pairwise evidence for all target classes.

#     Returns:

#         pairwise_class_evidence
#         detailed pairwise rows
#     """

#     class_scores: Dict[
#         str,
#         List[Tuple[float, float]]
#     ] = {
#         class_name: []
#         for class_name in TARGET_CLASSES
#     }

#     detail_rows: List[
#         Dict[str, Any]
#     ] = []

#     # Only pairs where both classes belong to verifier scope.
#     for _, row in pairwise_df.iterrows():

#         class_a = normalize_name(
#             row.get("class_a")
#         )

#         class_b = normalize_name(
#             row.get("class_b")
#         )

#         if (
#             class_a not in TARGET_CLASSES
#             or class_b not in TARGET_CLASSES
#         ):
#             continue

#         feature = normalize_name(
#             row.get("feature")
#         )

#         if feature not in feature_values:
#             continue

#         value = safe_float(
#             feature_values.get(feature)
#         )

#         if value is None:
#             continue

#         direction = normalize_name(
#             row.get("direction")
#         ).upper()

#         if direction not in {
#             "HIGH",
#             "LOW",
#         }:
#             continue

#         direction_consistency = safe_float(
#             row.get(
#                 "direction_consistency"
#             )
#         )

#         if (
#             direction_consistency is not None
#             and direction_consistency
#             < MIN_DIRECTION_CONSISTENCY
#         ):
#             continue

#         result = pairwise_feature_score(
#             value=value,
#             class_a=class_a,
#             class_b=class_b,
#             feature=feature,
#             direction=direction,
#             distributions=distributions,
#         )

#         if result is None:
#             continue

#         weight = candidate_weight_from_row(
#             row
#         )

#         if weight <= 0:
#             continue

#         score_a = result[
#             "score_a"
#         ]

#         score_b = result[
#             "score_b"
#         ]

#         class_scores[
#             class_a
#         ].append(
#             (
#                 score_a,
#                 weight,
#             )
#         )

#         class_scores[
#             class_b
#         ].append(
#             (
#                 score_b,
#                 weight,
#             )
#         )

#         detail_rows.append({
#             "class_a": class_a,
#             "class_b": class_b,
#             "feature": feature,
#             "value": value,
#             "direction": direction,
#             "q_a": result["q_a"],
#             "q_b": result["q_b"],
#             "delta": result["delta"],
#             "score_a": score_a,
#             "score_b": score_b,
#             "weight": weight,
#             "weighted_score_a": (
#                 score_a * weight
#             ),
#             "weighted_score_b": (
#                 score_b * weight
#             ),
#         })

#     final_scores: Dict[
#         str,
#         float
#     ] = {}

#     for class_name in TARGET_CLASSES:

#         values = class_scores[
#             class_name
#         ]

#         if not values:
#             final_scores[
#                 class_name
#             ] = 0.5

#             continue

#         numerator = sum(
#             score * weight
#             for score, weight
#             in values
#         )

#         denominator = sum(
#             weight
#             for _, weight
#             in values
#         )

#         if denominator <= 0:
#             final_scores[
#                 class_name
#             ] = 0.5

#         else:
#             final_scores[
#                 class_name
#             ] = clamp(
#                 numerator / denominator
#             )

#     return (
#         final_scores,
#         detail_rows,
#     )


# # ============================================================================
# # COMBINE EVIDENCE
# # ============================================================================

# def combine_evidence(
#     class_specific: Dict[str, float],
#     pairwise: Dict[str, float],
# ) -> Dict[str, float]:
#     """
#     Combine two evidence sources.

#     Missing evidence source = ignored rather than treated as zero.
#     """

#     final: Dict[str, float] = {}

#     for class_name in TARGET_CLASSES:

#         cs = class_specific.get(
#             class_name
#         )

#         pw = pairwise.get(
#             class_name
#         )

#         values = []
#         weights = []

#         if (
#             cs is not None
#             and math.isfinite(cs)
#         ):
#             values.append(cs)
#             weights.append(
#                 CLASS_SPECIFIC_WEIGHT
#             )

#         if (
#             pw is not None
#             and math.isfinite(pw)
#         ):
#             values.append(pw)
#             weights.append(
#                 PAIRWISE_WEIGHT
#             )

#         if not values:
#             final[
#                 class_name
#             ] = 0.5

#             continue

#         final[
#             class_name
#         ] = clamp(
#             sum(
#                 value * weight
#                 for value, weight
#                 in zip(values, weights)
#             )
#             / sum(weights)
#         )

#     return final


# # ============================================================================
# # RANKING
# # ============================================================================

# def rank_class_evidence(
#     evidence: Dict[str, float],
# ) -> List[Tuple[str, float]]:
#     """Return classes sorted by evidence descending."""

#     return sorted(
#         evidence.items(),
#         key=lambda x: x[1],
#         reverse=True,
#     )


# def calculate_margin(
#     cnn_class: str,
#     evidence: Dict[str, float],
# ) -> Tuple[
#     Optional[str],
#     float,
#     float,
# ]:
#     """
#     Calculate:

#         verifier evidence for CNN class
#         best rival
#         margin

#     Important:
#         CNN class is NOT changed.
#     """

#     if cnn_class not in TARGET_CLASSES:
#         return (
#             None,
#             float("nan"),
#             float("nan"),
#         )

#     cnn_evidence = evidence.get(
#         cnn_class,
#         0.5,
#     )

#     rivals = [
#         (
#             class_name,
#             score,
#         )
#         for class_name, score
#         in evidence.items()
#         if class_name != cnn_class
#     ]

#     if not rivals:
#         return (
#             None,
#             cnn_evidence,
#             float("nan"),
#         )

#     best_rival, rival_score = max(
#         rivals,
#         key=lambda x: x[1],
#     )

#     margin = (
#         cnn_evidence
#         - rival_score
#     )

#     return (
#         best_rival,
#         cnn_evidence,
#         margin,
#     )


# # ============================================================================
# # VERDICT
# # ============================================================================

# def verifier_verdict(
#     cnn_class: str,
#     evidence: Dict[str, float],
# ) -> Dict[str, Any]:
#     """
#     Determine verifier verdict.

#     CNN class is NEVER changed.

#     Possible verdicts:

#         SUPPORTED
#         WEAK
#         AMBIGUOUS
#         OUT_OF_SCOPE
#     """

#     if cnn_class not in TARGET_CLASSES:

#         return {
#             "verdict": "OUT_OF_SCOPE",
#             "cnn_evidence": float("nan"),
#             "best_rival": "",
#             "rival_evidence": float("nan"),
#             "margin": float("nan"),
#         }

#     best_rival, cnn_evidence, margin = (
#         calculate_margin(
#             cnn_class,
#             evidence,
#         )
#     )

#     if best_rival is None:

#         return {
#             "verdict": "AMBIGUOUS",
#             "cnn_evidence": cnn_evidence,
#             "best_rival": "",
#             "rival_evidence": float("nan"),
#             "margin": float("nan"),
#         }

#     rival_evidence = evidence[
#         best_rival
#     ]

#     # Strong support
#     if (
#         cnn_evidence
#         >= SUPPORTED_MIN_EVIDENCE
#         and margin
#         >= SUPPORTED_MIN_MARGIN
#     ):
#         verdict = "SUPPORTED"

#     # Very close contest
#     elif abs(margin) <= AMBIGUOUS_MARGIN:
#         verdict = "AMBIGUOUS"

#     # Evidence supports CNN class, but not strongly enough
#     elif (
#         cnn_evidence
#         >= WEAK_MIN_EVIDENCE
#         and margin > AMBIGUOUS_MARGIN
#     ):
#         verdict = "WEAK"

#     # Rival is stronger
#     else:
#         verdict = "AMBIGUOUS"

#     return {
#         "verdict": verdict,
#         "cnn_evidence": cnn_evidence,
#         "best_rival": best_rival,
#         "rival_evidence": rival_evidence,
#         "margin": margin,
#     }


# # ============================================================================
# # SINGLE IMAGE VERIFICATION
# # ============================================================================

# def verify_image(
#     image_path: Path,
#     classifier,
#     onnx_path: Path,
#     distributions: Dict[
#         Tuple[str, str],
#         Dict[str, float]
#     ],
#     stability_df: pd.DataFrame,
#     pairwise_df: pd.DataFrame,
#     candidates_df: Optional[pd.DataFrame],
#     true_class: Optional[str] = None,
# ) -> Tuple[
#     Dict[str, Any],
#     List[Dict[str, Any]],
#     List[Dict[str, Any]],
# ]:
#     """
#     Verify one image.

#     Returns:

#         result_row
#         class_specific_detail_rows
#         pairwise_detail_rows
#     """

#     image_path = image_path.resolve()

#     # ------------------------------------------------------------------------
#     # CNN
#     # ------------------------------------------------------------------------

#     cnn = run_cnn(
#         image_path=image_path,
#         onnx_path=onnx_path,
#         classifier=classifier,
#     )

#     cnn_class = cnn[
#         "cnn_class"
#     ]

#     cnn_confidence = cnn[
#         "cnn_confidence"
#     ]

#     # ------------------------------------------------------------------------
#     # Features
#     # ------------------------------------------------------------------------

#     feature_values = extract_image_features(
#         image_path=image_path,
#         classifier=classifier,
#     )

#     # ------------------------------------------------------------------------
#     # Candidate tables
#     # ------------------------------------------------------------------------

#     class_specific_df = (
#         build_class_specific_rows(
#             stability_df=stability_df,
#             candidates_df=candidates_df,
#         )
#     )

#     # ------------------------------------------------------------------------
#     # Class-specific evidence
#     # ------------------------------------------------------------------------

#     class_specific_scores: Dict[
#         str,
#         float
#     ] = {}

#     class_specific_details: List[
#         Dict[str, Any]
#     ] = []

#     for class_name in TARGET_CLASSES:

#         score, details = (
#             calculate_class_specific_evidence(
#                 class_name=class_name,
#                 feature_values=feature_values,
#                 class_specific_df=class_specific_df,
#                 distributions=distributions,
#             )
#         )

#         class_specific_scores[
#             class_name
#         ] = score

#         for detail in details:

#             detail[
#                 "image"
#             ] = str(image_path)

#             detail[
#                 "evidence_source"
#             ] = "CLASS_SPECIFIC"

#             class_specific_details.append(
#                 detail
#             )

#     # ------------------------------------------------------------------------
#     # Pairwise evidence
#     # ------------------------------------------------------------------------

#     pairwise_scores, pairwise_details = (
#         calculate_pairwise_evidence(
#             feature_values=feature_values,
#             pairwise_df=pairwise_df,
#             distributions=distributions,
#         )
#     )

#     for detail in pairwise_details:

#         detail[
#             "image"
#         ] = str(image_path)

#         detail[
#             "evidence_source"
#         ] = "PAIRWISE"

#     # ------------------------------------------------------------------------
#     # Combined evidence
#     # ------------------------------------------------------------------------

#     final_evidence = combine_evidence(
#         class_specific=class_specific_scores,
#         pairwise=pairwise_scores,
#     )

#     # ------------------------------------------------------------------------
#     # Verdict
#     # ------------------------------------------------------------------------

#     verdict_info = verifier_verdict(
#         cnn_class=cnn_class,
#         evidence=final_evidence,
#     )

#     # ------------------------------------------------------------------------
#     # Ranking
#     # ------------------------------------------------------------------------

#     ranking = rank_class_evidence(
#         final_evidence
#     )

#     evidence_rank = {
#         class_name: index + 1
#         for index, (
#             class_name,
#             _
#         ) in enumerate(ranking)
#     }

#     # ------------------------------------------------------------------------
#     # Best evidence features
#     # ------------------------------------------------------------------------

#     combined_feature_rows = []

#     for row in class_specific_details:
#         combined_feature_rows.append({
#             "source": "CLASS_SPECIFIC",
#             "feature": row["feature"],
#             "target_class": row["class"],
#             "evidence": row["evidence"],
#             "weight": row["weight"],
#         })

#     for row in pairwise_details:
#         combined_feature_rows.append({
#             "source": "PAIRWISE",
#             "feature": row["feature"],
#             "target_class": row["class_a"],
#             "against_class": row["class_b"],
#             "evidence": row["score_a"],
#             "weight": row["weight"],
#         })

#     combined_feature_rows.sort(
#         key=lambda x: (
#             x.get("evidence", 0.0)
#             * x.get("weight", 0.0)
#         ),
#         reverse=True,
#     )

#     top_features = (
#         combined_feature_rows[
#             :TOP_EVIDENCE_FEATURES
#         ]
#     )

#     top_feature_names = [
#         str(x["feature"])
#         for x in top_features
#     ]

#     # ------------------------------------------------------------------------
#     # Result row
#     # ------------------------------------------------------------------------

#     result: Dict[str, Any] = {

#         "image": str(
#             image_path
#         ),

#         "filename": image_path.name,

#         "true_class": (
#             true_class
#             if true_class
#             else ""
#         ),

#         # CNN
#         "cnn_class": cnn_class,
#         "cnn_confidence": cnn_confidence,

#         # Evidence
#         "good_evidence": final_evidence.get(
#             "01_Good",
#             0.5,
#         ),

#         "out_of_focus_evidence": final_evidence.get(
#             "02_Out_of_Focus",
#             0.5,
#         ),

#         "tracking_error_evidence": final_evidence.get(
#             "03_Tracking_Error",
#             0.5,
#         ),

#         # Source evidence
#         "good_class_specific": class_specific_scores.get(
#             "01_Good",
#             0.5,
#         ),

#         "out_of_focus_class_specific": class_specific_scores.get(
#             "02_Out_of_Focus",
#             0.5,
#         ),

#         "tracking_error_class_specific": class_specific_scores.get(
#             "03_Tracking_Error",
#             0.5,
#         ),

#         "good_pairwise": pairwise_scores.get(
#             "01_Good",
#             0.5,
#         ),

#         "out_of_focus_pairwise": pairwise_scores.get(
#             "02_Out_of_Focus",
#             0.5,
#         ),

#         "tracking_error_pairwise": pairwise_scores.get(
#             "03_Tracking_Error",
#             0.5,
#         ),

#         # Ranking
#         "evidence_rank_cnn_class": (
#             evidence_rank.get(
#                 cnn_class,
#                 "",
#             )
#         ),

#         # Best rival
#         "best_rival": verdict_info[
#             "best_rival"
#         ],

#         "cnn_class_evidence": verdict_info[
#             "cnn_evidence"
#         ],

#         "best_rival_evidence": verdict_info[
#             "rival_evidence"
#         ],

#         "pairwise_margin": verdict_info[
#             "margin"
#         ],

#         # Verdict
#         "verdict": verdict_info[
#             "verdict"
#         ],

#         # Evidence count
#         "class_specific_feature_count": len(
#             class_specific_details
#         ),

#         "pairwise_feature_count": len(
#             pairwise_details
#         ),

#         # Top evidence
#         "top_evidence_features": (
#             "; ".join(
#                 top_feature_names
#             )
#         ),
#     }

#     # ------------------------------------------------------------------------
#     # Add CNN probabilities
#     # ------------------------------------------------------------------------

#     for class_name in TARGET_CLASSES:

#         safe_name = (
#             class_name
#             .lower()
#             .replace(
#                 " ",
#                 "_",
#             )
#         )

#         result[
#             f"cnn_prob_{safe_name}"
#         ] = cnn[
#             "cnn_scores"
#         ].get(
#             class_name,
#             np.nan,
#         )

#     # ------------------------------------------------------------------------
#     # Add all extracted features
#     # ------------------------------------------------------------------------

#     for feature_name, value in feature_values.items():

#         if isinstance(
#             value,
#             (int, np.integer),
#         ):
#             result[
#                 f"feature_{feature_name}"
#             ] = int(value)

#         else:

#             numeric = safe_float(
#                 value
#             )

#             if numeric is not None:
#                 result[
#                     f"feature_{feature_name}"
#                 ] = numeric

#     return (
#         result,
#         class_specific_details,
#         pairwise_details,
#     )


# # ============================================================================
# # PRINT SINGLE IMAGE RESULT
# # ============================================================================

# def print_single_result(
#     result: Dict[str, Any],
# ) -> None:

#     print()
#     print("=" * 90)
#     print("VERIFIER RESULT")
#     print("=" * 90)

#     print(
#         f"Image             : "
#         f"{result['filename']}"
#     )

#     print(
#         f"CNN Prediction    : "
#         f"{result['cnn_class']}"
#     )

#     print(
#         f"CNN Confidence    : "
#         f"{result['cnn_confidence'] * 100:.2f}%"
#     )

#     print()

#     print(
#         f"Good Evidence     : "
#         f"{result['good_evidence']:.3f}"
#     )

#     print(
#         f"Out_of_Focus      : "
#         f"{result['out_of_focus_evidence']:.3f}"
#     )

#     print(
#         f"Tracking_Error    : "
#         f"{result['tracking_error_evidence']:.3f}"
#     )

#     print()

#     print(
#         f"Best Rival        : "
#         f"{result['best_rival'] or '-'}"
#     )

#     margin = result[
#         "pairwise_margin"
#     ]

#     if is_finite_number(margin):
#         print(
#             f"Margin            : "
#             f"{margin:+.3f}"
#         )
#     else:
#         print(
#             "Margin            : -"
#         )

#     print()

#     print(
#         f"Verifier Verdict  : "
#         f"{result['verdict']}"
#     )

#     print()

#     print("-" * 90)
#     print("EVIDENCE BY CLASS")
#     print("-" * 90)

#     print(
#         f"{'Class':<25}"
#         f"{'Class-Specific':>18}"
#         f"{'Pairwise':>15}"
#         f"{'Final':>15}"
#     )

#     print("-" * 90)

#     rows = [
#         (
#             "01_Good",
#             result[
#                 "good_class_specific"
#             ],
#             result[
#                 "good_pairwise"
#             ],
#             result[
#                 "good_evidence"
#             ],
#         ),
#         (
#             "02_Out_of_Focus",
#             result[
#                 "out_of_focus_class_specific"
#             ],
#             result[
#                 "out_of_focus_pairwise"
#             ],
#             result[
#                 "out_of_focus_evidence"
#             ],
#         ),
#         (
#             "03_Tracking_Error",
#             result[
#                 "tracking_error_class_specific"
#             ],
#             result[
#                 "tracking_error_pairwise"
#             ],
#             result[
#                 "tracking_error_evidence"
#             ],
#         ),
#     ]

#     for name, cs, pw, final in rows:

#         print(
#             f"{name:<25}"
#             f"{cs:>18.3f}"
#             f"{pw:>15.3f}"
#             f"{final:>15.3f}"
#         )

#     print("-" * 90)


# # ============================================================================
# # IMAGE DISCOVERY
# # ============================================================================

# def list_images(
#     folder: Path,
# ) -> List[Path]:
#     """Recursively discover supported image files."""

#     if not folder.exists():
#         raise FileNotFoundError(
#             f"ไม่พบ folder: {folder}"
#         )

#     images = [
#         p
#         for p in folder.rglob("*")
#         if (
#             p.is_file()
#             and p.suffix.lower()
#             in IMAGE_EXTENSIONS
#         )
#     ]

#     images.sort(
#         key=lambda p: str(p).lower()
#     )

#     return images


# # ============================================================================
# # TRUE LABEL INFERENCE
# # ============================================================================

# def infer_true_class_from_path(
#     image_path: Path,
# ) -> Optional[str]:
#     """
#     Try to infer true class from parent folders.

#     Example:

#         Dataset/
#             01_Good/
#                 img001.jpg

#     returns:

#         01_Good
#     """

#     parts = [
#         part
#         for part in image_path.parts
#     ]

#     for part in reversed(parts):

#         if part in TARGET_CLASSES:
#             return part

#     return None


# # ============================================================================
# # CSV INPUT
# # ============================================================================

# def find_image_column(
#     df: pd.DataFrame,
#     requested: Optional[str] = None,
# ) -> str:

#     if requested:

#         if requested not in df.columns:
#             raise SystemExit(
#                 f"[ERROR] ไม่พบ image column: "
#                 f"{requested}"
#             )

#         return requested

#     candidates = [
#         "image",
#         "image_path",
#         "filepath",
#         "file_path",
#         "path",
#         "filename",
#         "file",
#     ]

#     for column in candidates:

#         if column in df.columns:
#             return column

#     raise SystemExit(
#         "\n[ERROR] ไม่พบ column สำหรับ image path\n"
#         "ลองใช้:\n"
#         "  --image-column image\n"
#         "หรือให้ CSV มี column เช่น:\n"
#         "  image\n"
#         "  image_path\n"
#         "  path\n"
#     )


# def find_label_column(
#     df: pd.DataFrame,
#     requested: Optional[str] = None,
# ) -> Optional[str]:

#     if requested:

#         if requested not in df.columns:
#             raise SystemExit(
#                 f"[ERROR] ไม่พบ label column: "
#                 f"{requested}"
#             )

#         return requested

#     candidates = [
#         "true_class",
#         "true_label",
#         "label",
#         "class",
#         "target",
#     ]

#     for column in candidates:

#         if column in df.columns:
#             return column

#     return None


# def resolve_csv_image_path(
#     value: Any,
#     csv_path: Path,
# ) -> Path:

#     raw = str(value).strip()

#     path = Path(raw)

#     if path.is_absolute():
#         return path

#     # First try relative to CSV location
#     candidate = (
#         csv_path.parent
#         / path
#     )

#     if candidate.exists():
#         return candidate

#     # Then relative to current working directory
#     return path


# # ============================================================================
# # RUN ONE IMAGE
# # ============================================================================

# def run_one_image(
#     image_path: Path,
#     classifier,
#     onnx_path: Path,
#     distributions,
#     stability_df,
#     pairwise_df,
#     candidates_df,
#     true_class=None,
# ):
#     """
#     Run verification and catch image-specific errors.
#     """

#     try:

#         return verify_image(
#             image_path=image_path,
#             classifier=classifier,
#             onnx_path=onnx_path,
#             distributions=distributions,
#             stability_df=stability_df,
#             pairwise_df=pairwise_df,
#             candidates_df=candidates_df,
#             true_class=true_class,
#         )

#     except Exception as exc:

#         print()
#         print(
#             f"[ERROR] {image_path.name}: "
#             f"{repr(exc)}"
#         )

#         error_result = {
#             "image": str(
#                 image_path.resolve()
#             ),
#             "filename": image_path.name,
#             "true_class": (
#                 true_class
#                 if true_class
#                 else ""
#             ),
#             "cnn_class": "__ERROR__",
#             "cnn_confidence": np.nan,
#             "good_evidence": np.nan,
#             "out_of_focus_evidence": np.nan,
#             "tracking_error_evidence": np.nan,
#             "good_class_specific": np.nan,
#             "out_of_focus_class_specific": np.nan,
#             "tracking_error_class_specific": np.nan,
#             "good_pairwise": np.nan,
#             "out_of_focus_pairwise": np.nan,
#             "tracking_error_pairwise": np.nan,
#             "best_rival": "",
#             "cnn_class_evidence": np.nan,
#             "best_rival_evidence": np.nan,
#             "pairwise_margin": np.nan,
#             "verdict": "ERROR",
#             "error": repr(exc),
#         }

#         return (
#             error_result,
#             [],
#             [],
#         )


# # ============================================================================
# # FOLDER MODE
# # ============================================================================

# def run_folder_mode(
#     folder: Path,
#     classifier,
#     onnx_path: Path,
#     distributions,
#     stability_df,
#     pairwise_df,
#     candidates_df,
#     max_images: Optional[int] = None,
# ):
#     """Run verifier on all images in a folder."""

#     images = list_images(
#         folder
#     )

#     if max_images is not None:
#         images = images[
#             :max_images
#         ]

#     print()
#     print("=" * 90)
#     print("FOLDER MODE")
#     print("=" * 90)

#     print(
#         f"Folder : {folder.resolve()}"
#     )

#     print(
#         f"Images : {len(images)}"
#     )

#     results = []
#     evidence_rows = []

#     for index, image_path in enumerate(
#         images,
#         start=1,
#     ):

#         print(
#             f"\n[{index}/{len(images)}] "
#             f"{image_path.name}"
#         )

#         true_class = (
#             infer_true_class_from_path(
#                 image_path
#             )
#         )

#         result, class_details, pair_details = (
#             run_one_image(
#                 image_path=image_path,
#                 classifier=classifier,
#                 onnx_path=onnx_path,
#                 distributions=distributions,
#                 stability_df=stability_df,
#                 pairwise_df=pairwise_df,
#                 candidates_df=candidates_df,
#                 true_class=true_class,
#             )
#         )

#         results.append(
#             result
#         )

#         evidence_rows.extend(
#             class_details
#         )

#         evidence_rows.extend(
#             pair_details
#         )

#     return (
#         results,
#         evidence_rows,
#     )


# # ============================================================================
# # CSV MODE
# # ============================================================================

# def run_csv_mode(
#     csv_path: Path,
#     classifier,
#     onnx_path: Path,
#     distributions,
#     stability_df,
#     pairwise_df,
#     candidates_df,
#     image_column: Optional[str] = None,
#     label_column: Optional[str] = None,
#     max_images: Optional[int] = None,
# ):
#     """Run verifier on images listed in CSV."""

#     df = read_csv_flexible(
#         csv_path
#     )

#     image_column = find_image_column(
#         df,
#         image_column,
#     )

#     label_column = find_label_column(
#         df,
#         label_column,
#     )

#     print()
#     print("=" * 90)
#     print("CSV MODE")
#     print("=" * 90)

#     print(
#         f"CSV            : "
#         f"{csv_path.resolve()}"
#     )

#     print(
#         f"Image column    : "
#         f"{image_column}"
#     )

#     print(
#         f"Label column    : "
#         f"{label_column or '-'}"
#     )

#     if max_images is not None:
#         df = df.head(
#             max_images
#         )

#     results = []
#     evidence_rows = []

#     for index, row in df.iterrows():

#         image_path = resolve_csv_image_path(
#             row[
#                 image_column
#             ],
#             csv_path,
#         )

#         true_class = None

#         if label_column:
#             raw_label = row[
#                 label_column
#             ]

#             if pd.notna(raw_label):
#                 true_class = (
#                     normalize_name(
#                         raw_label
#                     )
#                 )

#         print()
#         print(
#             f"[{index + 1}/{len(df)}] "
#             f"{image_path}"
#         )

#         result, class_details, pair_details = (
#             run_one_image(
#                 image_path=image_path,
#                 classifier=classifier,
#                 onnx_path=onnx_path,
#                 distributions=distributions,
#                 stability_df=stability_df,
#                 pairwise_df=pairwise_df,
#                 candidates_df=candidates_df,
#                 true_class=true_class,
#             )
#         )

#         results.append(
#             result
#         )

#         evidence_rows.extend(
#             class_details
#         )

#         evidence_rows.extend(
#             pair_details
#         )

#     return (
#         results,
#         evidence_rows,
#     )


# # ============================================================================
# # CONFUSION SUMMARY
# # ============================================================================

# def build_confusion_summary(
#     results: List[Dict[str, Any]],
# ) -> pd.DataFrame:
#     """
#     Build summary when true_class is available.

#     This is NOT changing predictions.

#     CNN remains the prediction.
#     Verifier verdict is reported separately.
#     """

#     rows = []

#     for result in results:

#         true_class = normalize_name(
#             result.get(
#                 "true_class"
#             )
#         )

#         cnn_class = normalize_name(
#             result.get(
#                 "cnn_class"
#             )
#         )

#         verdict = normalize_name(
#             result.get(
#                 "verdict"
#             )
#         )

#         if not true_class:
#             continue

#         rows.append({
#             "true_class": true_class,
#             "cnn_class": cnn_class,
#             "verdict": verdict,
#         })

#     if not rows:
#         return pd.DataFrame()

#     df = pd.DataFrame(
#         rows
#     )

#     # ------------------------------------------------------------------------
#     # Main confusion table
#     # ------------------------------------------------------------------------

#     confusion = (
#         df.groupby(
#             [
#                 "true_class",
#                 "cnn_class",
#             ],
#             dropna=False,
#         )
#         .size()
#         .reset_index(
#             name="count"
#         )
#     )

#     total_errors = int(
#         (
#             df["true_class"]
#             != df["cnn_class"]
#         ).sum()
#     )

#     confusion[
#         "is_cnn_error"
#     ] = (
#         confusion[
#             "true_class"
#         ]
#         != confusion[
#             "cnn_class"
#         ]
#     )

#     if total_errors > 0:

#         confusion[
#             "percentage_of_cnn_errors"
#         ] = np.where(
#             confusion[
#                 "is_cnn_error"
#             ],
#             100.0
#             * confusion["count"]
#             / total_errors,
#             0.0,
#         )

#     else:

#         confusion[
#             "percentage_of_cnn_errors"
#         ] = 0.0

#     # ------------------------------------------------------------------------
#     # Verdict summary
#     # ------------------------------------------------------------------------

#     verdict_counts = (
#         df.groupby(
#             [
#                 "true_class",
#                 "cnn_class",
#                 "verdict",
#             ],
#             dropna=False,
#         )
#         .size()
#         .reset_index(
#             name="verdict_count"
#         )
#     )

#     summary = confusion.merge(
#         verdict_counts,
#         on=[
#             "true_class",
#             "cnn_class",
#         ],
#         how="left",
#     )

#     return summary.sort_values(
#         [
#             "is_cnn_error",
#             "count",
#         ],
#         ascending=[
#             False,
#             False,
#         ],
#     )


# # ============================================================================
# # GLOBAL METRICS
# # ============================================================================

# def build_global_summary(
#     results: List[Dict[str, Any]],
# ) -> Dict[str, Any]:

#     valid = [
#         r
#         for r in results
#         if r.get(
#             "cnn_class"
#         ) != "__ERROR__"
#     ]

#     if not valid:
#         return {}

#     total = len(
#         valid
#     )

#     cnn_known = [
#         r
#         for r in valid
#         if r.get(
#             "cnn_class"
#         )
#         in TARGET_CLASSES
#     ]

#     supported = sum(
#         r.get(
#             "verdict"
#         )
#         == "SUPPORTED"
#         for r in valid
#     )

#     weak = sum(
#         r.get(
#             "verdict"
#         )
#         == "WEAK"
#         for r in valid
#     )

#     ambiguous = sum(
#         r.get(
#             "verdict"
#         )
#         == "AMBIGUOUS"
#         for r in valid
#     )

#     out_of_scope = sum(
#         r.get(
#             "verdict"
#         )
#         == "OUT_OF_SCOPE"
#         for r in valid
#     )

#     errors = sum(
#         r.get(
#             "verdict"
#         )
#         == "ERROR"
#         for r in results
#     )

#     summary = {
#         "total_images": total,
#         "valid_images": len(valid),
#         "errors": errors,
#         "cnn_in_verifier_scope": len(
#             cnn_known
#         ),
#         "supported_count": supported,
#         "weak_count": weak,
#         "ambiguous_count": ambiguous,
#         "out_of_scope_count": out_of_scope,
#         "supported_rate": (
#             supported / total
#             if total
#             else 0.0
#         ),
#         "weak_rate": (
#             weak / total
#             if total
#             else 0.0
#         ),
#         "ambiguous_rate": (
#             ambiguous / total
#             if total
#             else 0.0
#         ),
#     }

#     # ------------------------------------------------------------------------
#     # If true labels exist
#     # ------------------------------------------------------------------------

#     labeled = [
#         r
#         for r in valid
#         if normalize_name(
#             r.get(
#                 "true_class"
#             )
#         )
#     ]

#     if labeled:

#         cnn_correct = sum(
#             r[
#                 "true_class"
#             ]
#             == r[
#                 "cnn_class"
#             ]
#             for r in labeled
#         )

#         summary[
#             "labeled_images"
#         ] = len(
#             labeled
#         )

#         summary[
#             "cnn_accuracy"
#         ] = (
#             cnn_correct
#             / len(labeled)
#         )

#     return summary


# def print_global_summary(
#     summary: Dict[str, Any],
# ) -> None:

#     print()
#     print("=" * 90)
#     print("GLOBAL VERIFIER SUMMARY")
#     print("=" * 90)

#     for key, value in summary.items():

#         if isinstance(
#             value,
#             float,
#         ):

#             if (
#                 "rate" in key
#                 or "accuracy" in key
#             ):

#                 print(
#                     f"{key:<30}: "
#                     f"{value:.4f}"
#                 )

#             else:

#                 print(
#                     f"{key:<30}: "
#                     f"{value:.4f}"
#                 )

#         else:

#             print(
#                 f"{key:<30}: "
#                 f"{value}"
#             )


# # ============================================================================
# # EVIDENCE SUMMARY CSV
# # ============================================================================

# def build_evidence_summary(
#     results: List[Dict[str, Any]],
# ) -> pd.DataFrame:
#     """
#     Create one row per image x verifier class.

#     This makes it easy to inspect:

#         image
#         CNN class
#         Good evidence
#         OOF evidence
#         TE evidence
#         best rival
#         margin
#         verdict
#     """

#     rows = []

#     for result in results:

#         image = result.get(
#             "image",
#             "",
#         )

#         cnn_class = result.get(
#             "cnn_class",
#             "",
#         )

#         rows.extend([
#             {
#                 "image": image,
#                 "filename": result.get(
#                     "filename",
#                     "",
#                 ),
#                 "true_class": result.get(
#                     "true_class",
#                     "",
#                 ),
#                 "cnn_class": cnn_class,
#                 "cnn_confidence": result.get(
#                     "cnn_confidence",
#                     np.nan,
#                 ),
#                 "verifier_class": "01_Good",
#                 "class_specific_evidence": result.get(
#                     "good_class_specific",
#                     np.nan,
#                 ),
#                 "pairwise_evidence": result.get(
#                     "good_pairwise",
#                     np.nan,
#                 ),
#                 "final_evidence": result.get(
#                     "good_evidence",
#                     np.nan,
#                 ),
#                 "best_rival": result.get(
#                     "best_rival",
#                     "",
#                 ),
#                 "margin": result.get(
#                     "pairwise_margin",
#                     np.nan,
#                 ),
#                 "verdict": result.get(
#                     "verdict",
#                     "",
#                 ),
#             },
#             {
#                 "image": image,
#                 "filename": result.get(
#                     "filename",
#                     "",
#                 ),
#                 "true_class": result.get(
#                     "true_class",
#                     "",
#                 ),
#                 "cnn_class": cnn_class,
#                 "cnn_confidence": result.get(
#                     "cnn_confidence",
#                     np.nan,
#                 ),
#                 "verifier_class": "02_Out_of_Focus",
#                 "class_specific_evidence": result.get(
#                     "out_of_focus_class_specific",
#                     np.nan,
#                 ),
#                 "pairwise_evidence": result.get(
#                     "out_of_focus_pairwise",
#                     np.nan,
#                 ),
#                 "final_evidence": result.get(
#                     "out_of_focus_evidence",
#                     np.nan,
#                 ),
#                 "best_rival": result.get(
#                     "best_rival",
#                     "",
#                 ),
#                 "margin": result.get(
#                     "pairwise_margin",
#                     np.nan,
#                 ),
#                 "verdict": result.get(
#                     "verdict",
#                     "",
#                 ),
#             },
#             {
#                 "image": image,
#                 "filename": result.get(
#                     "filename",
#                     "",
#                 ),
#                 "true_class": result.get(
#                     "true_class",
#                     "",
#                 ),
#                 "cnn_class": cnn_class,
#                 "cnn_confidence": result.get(
#                     "cnn_confidence",
#                     np.nan,
#                 ),
#                 "verifier_class": "03_Tracking_Error",
#                 "class_specific_evidence": result.get(
#                     "tracking_error_class_specific",
#                     np.nan,
#                 ),
#                 "pairwise_evidence": result.get(
#                     "tracking_error_pairwise",
#                     np.nan,
#                 ),
#                 "final_evidence": result.get(
#                     "tracking_error_evidence",
#                     np.nan,
#                 ),
#                 "best_rival": result.get(
#                     "best_rival",
#                     "",
#                 ),
#                 "margin": result.get(
#                     "pairwise_margin",
#                     np.nan,
#                 ),
#                 "verdict": result.get(
#                     "verdict",
#                     "",
#                 ),
#             },
#         ])

#     return pd.DataFrame(
#         rows
#     )


# # ============================================================================
# # MAIN
# # ============================================================================

# def main():

#     parser = argparse.ArgumentParser(
#         description=(
#             "Astronomy Evidence Verifier Tester"
#         )
#     )

#     # ------------------------------------------------------------------------
#     # Input mode
#     # ------------------------------------------------------------------------

#     mode = parser.add_mutually_exclusive_group(
#         required=True
#     )

#     mode.add_argument(
#         "--image",
#         type=Path,
#         help="ทดสอบภาพเดียว",
#     )

#     mode.add_argument(
#         "--folder",
#         type=Path,
#         help="ทดสอบทุกภาพใน folder",
#     )

#     mode.add_argument(
#         "--csv",
#         type=Path,
#         help="ทดสอบภาพจาก CSV",
#     )

#     # ------------------------------------------------------------------------
#     # Paths
#     # ------------------------------------------------------------------------

#     parser.add_argument(
#         "--classifier",
#         type=Path,
#         default=DEFAULT_CLASSIFIER,
#         help=(
#             "Path ของ astro_rule_classifier_new.py"
#         ),
#     )

#     parser.add_argument(
#         "--analysis-dir",
#         type=Path,
#         default=DEFAULT_ANALYSIS_DIR,
#         help=(
#             "Folder ที่มี feature_distribution.csv "
#             "และ pairwise_feature_stability.csv"
#         ),
#     )

#     parser.add_argument(
#         "--onnx",
#         type=Path,
#         default=DEFAULT_ONNX,
#         help=(
#             "Path ของ ONNX model"
#         ),
#     )

#     parser.add_argument(
#         "--output-dir",
#         type=Path,
#         default=None,
#         help=(
#             "Folder สำหรับ output CSV"
#         ),
#     )

#     # ------------------------------------------------------------------------
#     # CSV options
#     # ------------------------------------------------------------------------

#     parser.add_argument(
#         "--image-column",
#         type=str,
#         default=None,
#         help=(
#             "ชื่อ column ที่เก็บ image path"
#         ),
#     )

#     parser.add_argument(
#         "--label-column",
#         type=str,
#         default=None,
#         help=(
#             "ชื่อ column ที่เก็บ true class"
#         ),
#     )

#     # ------------------------------------------------------------------------
#     # Runtime
#     # ------------------------------------------------------------------------

#     parser.add_argument(
#         "--max-images",
#         type=int,
#         default=None,
#         help=(
#             "จำกัดจำนวนภาพ"
#         ),
#     )

#     args = parser.parse_args()

#     # =========================================================================
#     # Resolve paths
#     # =========================================================================

#     classifier_path = (
#         args.classifier
#         .resolve()
#     )

#     analysis_dir = (
#         args.analysis_dir
#         .resolve()
#     )

#     onnx_path = (
#         args.onnx
#         .resolve()
#     )

#     if args.output_dir:

#         output_dir = (
#             args.output_dir
#             .resolve()
#         )

#     else:

#         output_dir = (
#             analysis_dir
#             / "verifier_test"
#         )

#     output_dir.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     # =========================================================================
#     # Print configuration
#     # =========================================================================

#     print()
#     print("=" * 90)
#     print("ASTRONOMY EVIDENCE VERIFIER TEST")
#     print("=" * 90)

#     print(
#         f"Classifier : "
#         f"{classifier_path}"
#     )

#     print(
#         f"Analysis   : "
#         f"{analysis_dir}"
#     )

#     print(
#         f"ONNX       : "
#         f"{onnx_path}"
#     )

#     print(
#         f"Output     : "
#         f"{output_dir}"
#     )

#     print()

#     print(
#         "Verifier target classes:"
#     )

#     for class_name in TARGET_CLASSES:
#         print(
#             f"  - {class_name}"
#         )

#     # =========================================================================
#     # Check analysis files
#     # =========================================================================

#     distribution_path = (
#         analysis_dir
#         / "feature_distribution.csv"
#     )

#     stability_path = (
#         analysis_dir
#         / "verifier_feature_stability.csv"
#     )

#     pairwise_path = (
#         analysis_dir
#         / "pairwise_feature_stability.csv"
#     )

#     candidates_path = (
#         analysis_dir
#         / "verifier_evidence_candidates.csv"
#     )

#     required_files = [
#         distribution_path,
#         stability_path,
#         pairwise_path,
#     ]

#     missing_files = [
#         path
#         for path in required_files
#         if not path.exists()
#     ]

#     if missing_files:

#         raise SystemExit(
#             "\n[ERROR] ไม่พบไฟล์ analysis:\n"
#             + "\n".join(
#                 f"  - {path}"
#                 for path in missing_files
#             )
#             + "\n\n"
#             "ให้รัน analyze_rule_features.py "
#             "ก่อน"
#         )

#     # =========================================================================
#     # Import classifier
#     # =========================================================================

#     classifier = import_classifier(
#         classifier_path
#     )

#     class_names = list(
#         getattr(
#             classifier,
#             "CLASS_NAMES",
#             [],
#         )
#     )

#     print()
#     print(
#         f"[INFO] CNN classes: "
#         f"{class_names}"
#     )

#     missing_target_classes = [
#         class_name
#         for class_name in TARGET_CLASSES
#         if class_name not in class_names
#     ]

#     if missing_target_classes:

#         raise SystemExit(
#             "\n[ERROR] TARGET_CLASSES "
#             "ไม่มีอยู่ใน classifier:\n"
#             + "\n".join(
#                 f"  - {x}"
#                 for x in missing_target_classes
#             )
#         )

#     # =========================================================================
#     # Load analysis data
#     # =========================================================================

#     distributions = (
#         load_feature_distribution(
#             distribution_path
#         )
#     )

#     stability_df = (
#         load_stability_csv(
#             stability_path
#         )
#     )

#     pairwise_df = (
#         load_pairwise_stability(
#             pairwise_path
#         )
#     )

#     candidates_df = (
#         load_candidates(
#             candidates_path
#         )
#     )

#     # =========================================================================
#     # Run selected mode
#     # =========================================================================

#     results: List[
#         Dict[str, Any]
#     ] = []

#     evidence_rows: List[
#         Dict[str, Any]
#     ] = []

#     # -------------------------------------------------------------------------
#     # SINGLE IMAGE
#     # -------------------------------------------------------------------------

#     if args.image:

#         image_path = (
#             args.image
#             .resolve()
#         )

#         if not image_path.exists():

#             raise SystemExit(
#                 f"\n[ERROR] ไม่พบ image:\n"
#                 f"{image_path}"
#             )

#         (
#             result,
#             class_details,
#             pair_details,
#         ) = run_one_image(
#             image_path=image_path,
#             classifier=classifier,
#             onnx_path=onnx_path,
#             distributions=distributions,
#             stability_df=stability_df,
#             pairwise_df=pairwise_df,
#             candidates_df=candidates_df,
#             true_class=(
#                 infer_true_class_from_path(
#                     image_path
#                 )
#             ),
#         )

#         results.append(
#             result
#         )

#         evidence_rows.extend(
#             class_details
#         )

#         evidence_rows.extend(
#             pair_details
#         )

#         print_single_result(
#             result
#         )

#     # -------------------------------------------------------------------------
#     # FOLDER
#     # -------------------------------------------------------------------------

#     elif args.folder:

#         (
#             results,
#             evidence_rows,
#         ) = run_folder_mode(
#             folder=args.folder.resolve(),
#             classifier=classifier,
#             onnx_path=onnx_path,
#             distributions=distributions,
#             stability_df=stability_df,
#             pairwise_df=pairwise_df,
#             candidates_df=candidates_df,
#             max_images=args.max_images,
#         )

#     # -------------------------------------------------------------------------
#     # CSV
#     # -------------------------------------------------------------------------

#     elif args.csv:

#         (
#             results,
#             evidence_rows,
#         ) = run_csv_mode(
#             csv_path=args.csv.resolve(),
#             classifier=classifier,
#             onnx_path=onnx_path,
#             distributions=distributions,
#             stability_df=stability_df,
#             pairwise_df=pairwise_df,
#             candidates_df=candidates_df,
#             image_column=args.image_column,
#             label_column=args.label_column,
#             max_images=args.max_images,
#         )

#     # =========================================================================
#     # Save results
#     # =========================================================================

#     print()
#     print("=" * 90)
#     print("SAVE VERIFIER RESULTS")
#     print("=" * 90)

#     results_path = (
#         output_dir
#         / "verifier_test_results.csv"
#     )

#     write_csv_rows(
#         results_path,
#         results,
#     )

#     # =========================================================================
#     # Save detailed evidence
#     # =========================================================================

#     evidence_path = (
#         output_dir
#         / "verifier_evidence_summary.csv"
#     )

#     if results:

#         evidence_summary_df = (
#             build_evidence_summary(
#                 results
#             )
#         )

#         evidence_summary_df.to_csv(
#             evidence_path,
#             index=False,
#             encoding="utf-8-sig",
#         )

#         print(
#             f"[SAVED] "
#             f"{evidence_path.resolve()}"
#         )

#     # =========================================================================
#     # Save detailed feature evidence
#     # =========================================================================

#     detailed_evidence_path = (
#         output_dir
#         / "verifier_evidence_details.csv"
#     )

#     if evidence_rows:

#         detailed_evidence_df = (
#             pd.DataFrame(
#                 evidence_rows
#             )
#         )

#         detailed_evidence_df.to_csv(
#             detailed_evidence_path,
#             index=False,
#             encoding="utf-8-sig",
#         )

#         print(
#             f"[SAVED] "
#             f"{detailed_evidence_path.resolve()}"
#         )

#     # =========================================================================
#     # Confusion summary
#     # =========================================================================

#     confusion_df = (
#         build_confusion_summary(
#             results
#         )
#     )

#     confusion_path = (
#         output_dir
#         / "verifier_confusion_summary.csv"
#     )

#     if not confusion_df.empty:

#         confusion_df.to_csv(
#             confusion_path,
#             index=False,
#             encoding="utf-8-sig",
#         )

#         print(
#             f"[SAVED] "
#             f"{confusion_path.resolve()}"
#         )

#     else:

#         print(
#             "[INFO] ไม่มี true_class "
#             "จึงยังสร้าง confusion summary ไม่ได้"
#         )

#     # =========================================================================
#     # Global summary
#     # =========================================================================

#     global_summary = (
#         build_global_summary(
#             results
#         )
#     )

#     print_global_summary(
#         global_summary
#     )

#     # =========================================================================
#     # Print batch summary
#     # =========================================================================

#     if len(results) > 1:

#         print()
#         print("=" * 90)
#         print("VERDICT COUNTS")
#         print("=" * 90)

#         verdict_counts = {}

#         for result in results:

#             verdict = result.get(
#                 "verdict",
#                 "UNKNOWN",
#             )

#             verdict_counts[
#                 verdict
#             ] = (
#                 verdict_counts.get(
#                     verdict,
#                     0,
#                 )
#                 + 1
#             )

#         for verdict, count in sorted(
#             verdict_counts.items()
#         ):

#             print(
#                 f"{verdict:<20}: "
#                 f"{count}"
#             )

#     # =========================================================================
#     # Final
#     # =========================================================================

#     print()
#     print("=" * 90)
#     print("DONE")
#     print("=" * 90)

#     print(
#         f"Output folder:\n"
#         f"{output_dir.resolve()}"
#     )

#     print()
#     print(
#         "IMPORTANT:"
#     )

#     print(
#         "CNN class ยังคงเป็น prediction หลัก "
#         "และ Verifier เป็น evidence layer เท่านั้น"
#     )

#     print(
#         "Verifier ไม่ได้เปลี่ยน CNN class"
#     )

#     print(
#         "Verifier ไม่ได้ใช้ fuse_scores()"
#     )


# if __name__ == "__main__":
#     main()

"""
test_verifier.py

Astronomy Evidence Verifier Tester
===================================
"""
from __future__ import annotations
import argparse, csv, importlib.util, math, sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_CLASSIFIER = Path(r"D:\Internship\Test_Rule\test_rule_new_v2\astro_rule_classifier_new.py")
DEFAULT_ANALYSIS_DIR = Path(r"D:\Internship\Test_Rule\test_rule_new_v2\verifier_analysis_v2")
DEFAULT_ONNX = Path(r"D:\Internship\Test_Rule\test_rule_new_v2\eff_b0_kfold_add_focal_r2.onnx")
TARGET_CLASSES = ["01_Good", "02_Out_of_Focus", "03_Tracking_Error"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".fits", ".fit", ".fts"}

# CLASS_SPECIFIC_WEIGHT, PAIRWISE_WEIGHT = 0.50, 0.50
# SUPPORTED_MIN_EVIDENCE, SUPPORTED_MIN_MARGIN = 0.60, 0.15
# WEAK_MIN_EVIDENCE, AMBIGUOUS_MARGIN = 0.40, 0.08
# MIN_CANDIDATE_SCORE, MIN_DIRECTION_CONSISTENCY, TOP_EVIDENCE_FEATURES = 0.0, 0.0, 8

CLASS_SPECIFIC_WEIGHT, PAIRWISE_WEIGHT = 0.50, 0.50

# ============================================================================
# VERIFIER THRESHOLDS
# Optimized from threshold_class_analysis.py
# CNN logic และ CNN class prediction ไม่ถูกแก้ไข
# ============================================================================
SUPPORTED_MIN_EVIDENCE = 0.45
SUPPORTED_MIN_MARGIN = 0.04
WEAK_MIN_EVIDENCE = 0.30
AMBIGUOUS_MARGIN = 0.03

MIN_CANDIDATE_SCORE, MIN_DIRECTION_CONSISTENCY, TOP_EVIDENCE_FEATURES = 0.0, 0.0, 8

# ============================================================================
# UTILITIES
# ============================================================================
def safe_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if not math.isfinite(value) else max(low, min(high, value))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

def normalize_name(value: Any) -> str: return "" if value is None else str(value).strip()
def is_finite_number(value: Any) -> bool: return safe_float(value) is not None

def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[WARNING] ไม่มีข้อมูลสำหรับเขียน: {path}")
        return
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {path.resolve()}")

# ============================================================================
# IMPORT CLASSIFIER
# ============================================================================
def import_classifier(classifier_path: Path):
    classifier_path = classifier_path.resolve()
    if not classifier_path.exists(): raise SystemExit(f"\n[ERROR] ไม่พบ classifier:\n{classifier_path}\n")
    module_name = "astro_rule_classifier_new"
    spec = importlib.util.spec_from_file_location(module_name, str(classifier_path))
    if spec is None or spec.loader is None: raise SystemExit("\n[ERROR] ไม่สามารถสร้าง import spec สำหรับ classifier ได้")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try: spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise SystemExit(f"\n[ERROR] import astro_rule_classifier_new.py ไม่สำเร็จ\nPath : {classifier_path}\nError: {repr(exc)}\n") from exc
    missing = [name for name in ["CLASS_NAMES", "extract_features", "read_gray_image", "predict_onnx"] if not hasattr(module, name)]
    if missing: raise SystemExit("\n[ERROR] classifier ไม่มี function/variable ที่ต้องใช้:\n" + "\n".join(f"  - {x}" for x in missing))
    print("[OK] import astro_rule_classifier_new.py สำเร็จ")
    return module

# ============================================================================
# LOAD CSV
# ============================================================================
def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"ไม่พบไฟล์ CSV: {path}")
    try: df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError: df = pd.read_csv(path, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_feature_distribution(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    print("\n" + "=" * 90 + "\nLOAD FEATURE DISTRIBUTION\n" + "=" * 90)
    df = read_csv_flexible(path)
    missing = {"class", "feature", "p10", "p25", "p50", "p75", "p90"} - set(df.columns)
    if missing: raise SystemExit("\n[ERROR] feature_distribution.csv ขาด columns:\n" + "\n".join(f"  - {x}" for x in sorted(missing)))
    distributions: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, row in df.iterrows():
        c_name, f_name = normalize_name(row["class"]), normalize_name(row["feature"])
        if not c_name or not f_name: continue
        stats = {col: safe_float(row[col]) for col in ["n", "min", "p10", "p25", "p50", "p75", "p90", "max", "mean", "std"] if col in df.columns and safe_float(row[col]) is not None}
        if all(x in stats for x in ["p10", "p25", "p50", "p75", "p90"]): distributions[(c_name, f_name)] = stats
    print(f"[INFO] loaded distributions: {len(distributions)} rows")
    return distributions

def load_stability_csv(path: Path) -> pd.DataFrame:
    print("\n" + "=" * 90 + "\nLOAD VERIFIER FEATURE STABILITY\n" + "=" * 90)
    df = read_csv_flexible(path)
    print(f"[INFO] rows: {len(df)}")
    return df

def load_pairwise_stability(path: Path) -> pd.DataFrame:
    print("\n" + "=" * 90 + "\nLOAD PAIRWISE FEATURE STABILITY\n" + "=" * 90)
    df = read_csv_flexible(path)
    missing = {"class_a", "class_b", "feature", "direction"} - set(df.columns)
    if missing: raise SystemExit("\n[ERROR] pairwise_feature_stability.csv ขาด columns:\n" + "\n".join(f"  - {x}" for x in sorted(missing)))
    print(f"[INFO] rows: {len(df)}")
    return df

def load_candidates(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARNING] ไม่พบ candidate file:\n          {path}\n          จะใช้ stability CSV โดยตรง")
        return None
    print("\n" + "=" * 90 + "\nLOAD VERIFIER EVIDENCE CANDIDATES\n" + "=" * 90)
    df = read_csv_flexible(path)
    print(f"[INFO] rows: {len(df)}")
    return df

# ============================================================================
# FEATURE EXTRACTION & CNN
# ============================================================================
def feature_dict_from_dataclass(features: Any) -> Dict[str, Any]:
    if is_dataclass(features): return asdict(features)
    if hasattr(features, "__dict__"): return dict(features.__dict__)
    raise TypeError("extract_features() ไม่ได้คืน dataclass/object ที่สามารถแปลงเป็น dictionary ได้")

def extract_image_features(image_path: Path, classifier) -> Dict[str, Any]:
    return feature_dict_from_dataclass(classifier.extract_features(classifier.read_gray_image(str(image_path))))

def run_cnn(image_path: Path, onnx_path: Path, classifier) -> Dict[str, Any]:
    if not onnx_path.exists(): raise FileNotFoundError(f"ไม่พบ ONNX model: {onnx_path}")
    model_scores = classifier.predict_onnx(str(image_path), str(onnx_path))
    if not isinstance(model_scores, dict): raise TypeError("predict_onnx() ต้องคืน Dict[str, float]")
    cleaned_scores = {normalize_name(k): safe_float(v) for k, v in model_scores.items() if safe_float(v) is not None}
    if not cleaned_scores: raise RuntimeError("CNN ไม่คืน probability ที่ใช้งานได้")
    cnn_class = max(cleaned_scores, key=cleaned_scores.get)
    return {"cnn_class": cnn_class, "cnn_confidence": cleaned_scores[cnn_class], "cnn_scores": cleaned_scores}

# ============================================================================
# QUANTILE / DISTRIBUTION FUNCTIONS
# ============================================================================
QUANTILE_POINTS = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=float)

def clean_quantiles(stats: Dict[str, float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    keys = ["p10", "p25", "p50", "p75", "p90"]
    if not all(k in stats and safe_float(stats[k]) is not None for k in keys): return None
    values = np.maximum.accumulate(np.asarray([safe_float(stats[k]) for k in keys], dtype=float))
    return QUANTILE_POINTS.copy(), values

def percentile_position(value: float, stats: Dict[str, float]) -> Optional[float]:
    value = safe_float(value)
    if value is None: return None
    result = clean_quantiles(stats)
    if result is None: return None
    qs, xs = result
    if np.allclose(xs, xs[0]): return 0.5
    return clamp(float(np.interp(value, xs, qs)))

def directional_support(q: float, direction: str) -> float:
    direction, q = normalize_name(direction).upper(), clamp(q)
    return q if direction == "HIGH" else (1.0 - q if direction == "LOW" else 0.5)

def typicality_score(q: float) -> float: return clamp(1.0 - 2.0 * abs(clamp(q) - 0.5))

def distribution_compatibility(value: float, stats: Dict[str, float], direction: str) -> Optional[Dict[str, float]]:
    q = percentile_position(value, stats)
    if q is None: return None
    directional, typicality, direction_upper = directional_support(q, direction), typicality_score(q), normalize_name(direction).upper()
    evidence = (0.70 * directional + 0.30 * typicality) if direction_upper in {"HIGH", "LOW"} else typicality
    return {"percentile": q, "directional_support": directional, "typicality": typicality, "evidence": clamp(evidence)}

# ============================================================================
# CANDIDATE WEIGHTS & EVIDENCE CALCULATION
# ============================================================================
def candidate_weight_from_row(row: pd.Series) -> float:
    c_score = safe_float(row.get("candidate_score"))
    if c_score is not None and c_score >= MIN_CANDIDATE_SCORE: return clamp(c_score)
    m_str, min_str = safe_float(row.get("mean_strength")) or 0.0, safe_float(row.get("min_strength"))
    d_cons, s_str = safe_float(row.get("direction_consistency")) or 0.0, safe_float(row.get("std_strength")) or 1.0
    return clamp(0.45 * m_str + 0.30 * (min_str if min_str is not None else m_str) + 0.15 * d_cons + 0.10 * max(0.0, 1.0 - s_str))

def build_class_specific_rows(stability_df: pd.DataFrame, candidates_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if candidates_df is not None and {"candidate_type", "class", "feature", "direction"}.issubset(set(candidates_df.columns)):
        df = candidates_df[(candidates_df["candidate_type"] == "CLASS_SPECIFIC") & (candidates_df["class"].isin(TARGET_CLASSES))]
        if not df.empty: return df.copy()
    df = stability_df.copy()
    missing = {"class", "feature", "direction"} - set(df.columns)
    if missing: raise SystemExit("\n[ERROR] verifier_feature_stability.csv ขาด columns:\n" + "\n".join(f"  - {x}" for x in sorted(missing)))
    return df[df["class"].isin(TARGET_CLASSES)].copy()

def calculate_class_specific_evidence(class_name: str, feature_values: Dict[str, Any], class_specific_df: pd.DataFrame, distributions: Dict[Tuple[str, str], Dict[str, float]]) -> Tuple[float, List[Dict[str, Any]]]:
    rows, weighted_sum, weight_sum = [], 0.0, 0.0
    for _, row in class_specific_df[class_specific_df["class"] == class_name].iterrows():
        feature = normalize_name(row.get("feature"))
        if feature not in feature_values or safe_float(feature_values.get(feature)) is None: continue
        value, direction, dist = safe_float(feature_values.get(feature)), normalize_name(row.get("direction")).upper(), distributions.get((class_name, feature))
        if dist is None: continue
        result = distribution_compatibility(value, dist, direction)
        weight = candidate_weight_from_row(row)
        if result is None or weight <= 0: continue
        weighted_sum += result["evidence"] * weight
        weight_sum += weight
        rows.append({"class": class_name, "feature": feature, "value": value, "direction": direction, "percentile": result["percentile"], "directional_support": result["directional_support"], "typicality": result["typicality"], "evidence": result["evidence"], "weight": weight, "weighted_evidence": result["evidence"] * weight})
    return (clamp(weighted_sum / weight_sum) if weight_sum > 0 else 0.5), rows

def pairwise_feature_score(value: float, class_a: str, class_b: str, feature: str, direction: str, distributions: Dict[Tuple[str, str], Dict[str, float]]) -> Optional[Dict[str, float]]:
    stats_a, stats_b = distributions.get((class_a, feature)), distributions.get((class_b, feature))
    if not stats_a or not stats_b: return None
    q_a, q_b = percentile_position(value, stats_a), percentile_position(value, stats_b)
    if q_a is None or q_b is None: return None
    direction = normalize_name(direction).upper()
    delta = (q_a - q_b) if direction == "HIGH" else ((q_b - q_a) if direction == "LOW" else None)
    if delta is None: return None
    score_a = sigmoid(5.0 * delta)
    return {"q_a": q_a, "q_b": q_b, "delta": delta, "score_a": clamp(score_a), "score_b": clamp(1.0 - score_a)}

def calculate_pairwise_evidence(feature_values: Dict[str, Any], pairwise_df: pd.DataFrame, distributions: Dict[Tuple[str, str], Dict[str, float]]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    class_scores, detail_rows = {c: [] for c in TARGET_CLASSES}, []
    for _, row in pairwise_df.iterrows():
        c_a, c_b = normalize_name(row.get("class_a")), normalize_name(row.get("class_b"))
        if c_a not in TARGET_CLASSES or c_b not in TARGET_CLASSES: continue
        feature = normalize_name(row.get("feature"))
        if feature not in feature_values or safe_float(feature_values.get(feature)) is None: continue
        val, dir_up = safe_float(feature_values.get(feature)), normalize_name(row.get("direction")).upper()
        dir_cons = safe_float(row.get("direction_consistency"))
        if dir_up not in {"HIGH", "LOW"} or (dir_cons is not None and dir_cons < MIN_DIRECTION_CONSISTENCY): continue
        result, weight = pairwise_feature_score(val, c_a, c_b, feature, dir_up, distributions), candidate_weight_from_row(row)
        if result is None or weight <= 0: continue
        class_scores[c_a].append((result["score_a"], weight))
        class_scores[c_b].append((result["score_b"], weight))
        detail_rows.append({"class_a": c_a, "class_b": c_b, "feature": feature, "value": val, "direction": dir_up, "q_a": result["q_a"], "q_b": result["q_b"], "delta": result["delta"], "score_a": result["score_a"], "score_b": result["score_b"], "weight": weight, "weighted_score_a": result["score_a"] * weight, "weighted_score_b": result["score_b"] * weight})
    return {c: clamp(sum(s * w for s, w in class_scores[c]) / sum(w for _, w in class_scores[c])) if sum(w for _, w in class_scores[c]) > 0 else 0.5 for c in TARGET_CLASSES}, detail_rows

def combine_evidence(class_specific: Dict[str, float], pairwise: Dict[str, float]) -> Dict[str, float]:
    final = {}
    for c in TARGET_CLASSES:
        cs, pw, vals, wts = class_specific.get(c), pairwise.get(c), [], []
        if cs is not None and math.isfinite(cs): vals.append(cs); wts.append(CLASS_SPECIFIC_WEIGHT)
        if pw is not None and math.isfinite(pw): vals.append(pw); wts.append(PAIRWISE_WEIGHT)
        final[c] = clamp(sum(v * w for v, w in zip(vals, wts)) / sum(wts)) if vals else 0.5
    return final

# ============================================================================
# RANKING & VERDICT
# ============================================================================
def rank_class_evidence(evidence: Dict[str, float]) -> List[Tuple[str, float]]: return sorted(evidence.items(), key=lambda x: x[1], reverse=True)

def calculate_margin(cnn_class: str, evidence: Dict[str, float]) -> Tuple[Optional[str], float, float]:
    if cnn_class not in TARGET_CLASSES: return None, float("nan"), float("nan")
    cnn_ev = evidence.get(cnn_class, 0.5)
    rivals = [(c, s) for c, s in evidence.items() if c != cnn_class]
    if not rivals: return None, cnn_ev, float("nan")
    best_rival, rival_score = max(rivals, key=lambda x: x[1])
    return best_rival, cnn_ev, cnn_ev - rival_score

def verifier_verdict(cnn_class: str, evidence: Dict[str, float]) -> Dict[str, Any]:
    if cnn_class not in TARGET_CLASSES: return {"verdict": "OUT_OF_SCOPE", "cnn_evidence": float("nan"), "best_rival": "", "rival_evidence": float("nan"), "margin": float("nan")}
    best_rival, cnn_ev, margin = calculate_margin(cnn_class, evidence)
    if best_rival is None: return {"verdict": "AMBIGUOUS", "cnn_evidence": cnn_ev, "best_rival": "", "rival_evidence": float("nan"), "margin": float("nan")}
    rival_ev = evidence[best_rival]
    verdict = "SUPPORTED" if (cnn_ev >= SUPPORTED_MIN_EVIDENCE and margin >= SUPPORTED_MIN_MARGIN) else ("AMBIGUOUS" if abs(margin) <= AMBIGUOUS_MARGIN else ("WEAK" if (cnn_ev >= WEAK_MIN_EVIDENCE and margin > AMBIGUOUS_MARGIN) else "AMBIGUOUS"))
    return {"verdict": verdict, "cnn_evidence": cnn_ev, "best_rival": best_rival, "rival_evidence": rival_ev, "margin": margin}

def verify_image(image_path: Path, classifier, onnx_path: Path, distributions: Dict[Tuple[str, str], Dict[str, float]], stability_df: pd.DataFrame, pairwise_df: pd.DataFrame, candidates_df: Optional[pd.DataFrame], true_class: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    image_path = image_path.resolve()
    cnn = run_cnn(image_path, onnx_path, classifier)
    feature_values = extract_image_features(image_path, classifier)
    cs_df = build_class_specific_rows(stability_df, candidates_df)
    
    cs_scores, cs_details = {}, []
    for c in TARGET_CLASSES:
        score, details = calculate_class_specific_evidence(c, feature_values, cs_df, distributions)
        cs_scores[c] = score
        for d in details: d.update({"image": str(image_path), "evidence_source": "CLASS_SPECIFIC"}); cs_details.append(d)
        
    pw_scores, pw_details = calculate_pairwise_evidence(feature_values, pairwise_df, distributions)
    for d in pw_details: d.update({"image": str(image_path), "evidence_source": "PAIRWISE"})

    final_ev = combine_evidence(cs_scores, pw_scores)
    verdict_info = verifier_verdict(cnn["cnn_class"], final_ev)
    evidence_rank = {c: idx + 1 for idx, (c, _) in enumerate(rank_class_evidence(final_ev))}
    
    comb_feats = [{"source": "CLASS_SPECIFIC", "feature": r["feature"], "target_class": r["class"], "evidence": r["evidence"], "weight": r["weight"]} for r in cs_details] + [{"source": "PAIRWISE", "feature": r["feature"], "target_class": r["class_a"], "against_class": r["class_b"], "evidence": r["score_a"], "weight": r["weight"]} for r in pw_details]
    top_feats = [str(x["feature"]) for x in sorted(comb_feats, key=lambda x: x.get("evidence", 0.0) * x.get("weight", 0.0), reverse=True)[:TOP_EVIDENCE_FEATURES]]

    res = {
        "image": str(image_path), "filename": image_path.name, "true_class": true_class or "", "cnn_class": cnn["cnn_class"], "cnn_confidence": cnn["cnn_confidence"],
        "good_evidence": final_ev.get("01_Good", 0.5), "out_of_focus_evidence": final_ev.get("02_Out_of_Focus", 0.5), "tracking_error_evidence": final_ev.get("03_Tracking_Error", 0.5),
        "good_class_specific": cs_scores.get("01_Good", 0.5), "out_of_focus_class_specific": cs_scores.get("02_Out_of_Focus", 0.5), "tracking_error_class_specific": cs_scores.get("03_Tracking_Error", 0.5),
        "good_pairwise": pw_scores.get("01_Good", 0.5), "out_of_focus_pairwise": pw_scores.get("02_Out_of_Focus", 0.5), "tracking_error_pairwise": pw_scores.get("03_Tracking_Error", 0.5),
        "evidence_rank_cnn_class": evidence_rank.get(cnn["cnn_class"], ""), "best_rival": verdict_info["best_rival"], "cnn_class_evidence": verdict_info["cnn_evidence"], "best_rival_evidence": verdict_info["rival_evidence"], "pairwise_margin": verdict_info["margin"], "verdict": verdict_info["verdict"],
        "class_specific_feature_count": len(cs_details), "pairwise_feature_count": len(pw_details), "top_evidence_features": "; ".join(top_feats)
    }
    for c in TARGET_CLASSES: res[f"cnn_prob_{c.lower().replace(' ', '_')}"] = cnn["cnn_scores"].get(c, np.nan)
    for f, v in feature_values.items():
        if isinstance(v, (int, np.integer)): res[f"feature_{f}"] = int(v)
        elif safe_float(v) is not None: res[f"feature_{f}"] = safe_float(v)
    return res, cs_details, pw_details

def print_single_result(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 90 + "\nVERIFIER RESULT\n" + "=" * 90)
    print(f"Image             : {res['filename']}\nCNN Prediction    : {res['cnn_class']}\nCNN Confidence    : {res['cnn_confidence'] * 100:.2f}%\n")
    print(f"Good Evidence     : {res['good_evidence']:.3f}\nOut_of_Focus      : {res['out_of_focus_evidence']:.3f}\nTracking_Error    : {res['tracking_error_evidence']:.3f}\n")
    print(f"Best Rival        : {res['best_rival'] or '-'}\nMargin            : {res['pairwise_margin']:+.3f}" if is_finite_number(res['pairwise_margin']) else "Margin            : -")
    print(f"\nVerifier Verdict  : {res['verdict']}\n\n" + "-" * 90 + "\nEVIDENCE BY CLASS\n" + "-" * 90)
    print(f"{'Class':<25}{'Class-Specific':>18}{'Pairwise':>15}{'Final':>15}\n" + "-" * 90)
    for c, c_pfx in [("01_Good", "good"), ("02_Out_of_Focus", "out_of_focus"), ("03_Tracking_Error", "tracking_error")]:
        print(f"{c:<25}{res[f'{c_pfx}_class_specific']:>18.3f}{res[f'{c_pfx}_pairwise']:>15.3f}{res[f'{c_pfx}_evidence']:>15.3f}")
    print("-" * 90)

# ============================================================================
# HELPER RUNNERS & SUMMARIES
# ============================================================================
def list_images(folder: Path) -> List[Path]:
    if not folder.exists(): raise FileNotFoundError(f"ไม่พบ folder: {folder}")
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS], key=lambda p: str(p).lower())

def infer_true_class_from_path(image_path: Path) -> Optional[str]:
    for part in reversed(image_path.parts):
        if part in TARGET_CLASSES: return part
    return None

def find_image_column(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested:
        if requested not in df.columns: raise SystemExit(f"[ERROR] ไม่พบ image column: {requested}")
        return requested
    for col in ["image", "image_path", "filepath", "file_path", "path", "filename", "file"]:
        if col in df.columns: return col
    raise SystemExit("\n[ERROR] ไม่พบ column สำหรับ image path\nลองใช้:\n  --image-column image")

def find_label_column(df: pd.DataFrame, requested: Optional[str] = None) -> Optional[str]:
    if requested:
        if requested not in df.columns: raise SystemExit(f"[ERROR] ไม่พบ label column: {requested}")
        return requested
    for col in ["true_class", "true_label", "label", "class", "target"]:
        if col in df.columns: return col
    return None

def resolve_csv_image_path(value: Any, csv_path: Path) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else (csv_path.parent / path if (csv_path.parent / path).exists() else path)

def run_one_image(img_path: Path, clf, onnx, dists, stab, pw, cands, true_c=None):
    try: return verify_image(img_path, clf, onnx, dists, stab, pw, cands, true_c)
    except Exception as exc:
        print(f"\n[ERROR] {img_path.name}: {repr(exc)}")
        err_res = {"image": str(img_path.resolve()), "filename": img_path.name, "true_class": true_c or "", "cnn_class": "__ERROR__", "cnn_confidence": np.nan, "good_evidence": np.nan, "out_of_focus_evidence": np.nan, "tracking_error_evidence": np.nan, "good_class_specific": np.nan, "out_of_focus_class_specific": np.nan, "tracking_error_class_specific": np.nan, "good_pairwise": np.nan, "out_of_focus_pairwise": np.nan, "tracking_error_pairwise": np.nan, "best_rival": "", "cnn_class_evidence": np.nan, "best_rival_evidence": np.nan, "pairwise_margin": np.nan, "verdict": "ERROR", "error": repr(exc)}
        return err_res, [], []

def run_folder_mode(folder: Path, clf, onnx, dists, stab, pw, cands, max_imgs: Optional[int] = None):
    images = list_images(folder)[:max_imgs] if max_imgs else list_images(folder)
    print("\n" + "=" * 90 + "\nFOLDER MODE\n" + "=" * 90 + f"\nFolder : {folder.resolve()}\nImages : {len(images)}")
    results, ev_rows = [], []
    for idx, img in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] {img.name}")
        res, cs_det, pw_det = run_one_image(img, clf, onnx, dists, stab, pw, cands, infer_true_class_from_path(img))
        results.append(res); ev_rows.extend(cs_det + pw_det)
    return results, ev_rows

def run_csv_mode(csv_path: Path, clf, onnx, dists, stab, pw, cands, img_col: Optional[str] = None, lbl_col: Optional[str] = None, max_imgs: Optional[int] = None):
    df = read_csv_flexible(csv_path)
    img_col, lbl_col = find_image_column(df, img_col), find_label_column(df, lbl_col)
    print("\n" + "=" * 90 + "\nCSV MODE\n" + "=" * 90 + f"\nCSV            : {csv_path.resolve()}\nImage column    : {img_col}\nLabel column    : {lbl_col or '-'}")
    if max_imgs: df = df.head(max_imgs)
    results, ev_rows = [], []
    for idx, row in df.iterrows():
        img_path = resolve_csv_image_path(row[img_col], csv_path)
        true_c = normalize_name(row[lbl_col]) if lbl_col and pd.notna(row[lbl_col]) else None
        print(f"\n[{idx + 1}/{len(df)}] {img_path}")
        res, cs_det, pw_det = run_one_image(img_path, clf, onnx, dists, stab, pw, cands, true_c)
        results.append(res); ev_rows.extend(cs_det + pw_det)
    return results, ev_rows

def build_confusion_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [{"true_class": normalize_name(r.get("true_class")), "cnn_class": normalize_name(r.get("cnn_class")), "verdict": normalize_name(r.get("verdict"))} for r in results if normalize_name(r.get("true_class"))]
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    conf = df.groupby(["true_class", "cnn_class"], dropna=False).size().reset_index(name="count")
    total_errors = int((df["true_class"] != df["cnn_class"]).sum())
    conf["is_cnn_error"] = conf["true_class"] != conf["cnn_class"]
    conf["percentage_of_cnn_errors"] = np.where(conf["is_cnn_error"], 100.0 * conf["count"] / total_errors, 0.0) if total_errors > 0 else 0.0
    vd_counts = df.groupby(["true_class", "cnn_class", "verdict"], dropna=False).size().reset_index(name="verdict_count")
    return conf.merge(vd_counts, on=["true_class", "cnn_class"], how="left").sort_values(["is_cnn_error", "count"], ascending=[False, False])

def build_global_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if r.get("cnn_class") != "__ERROR__"]
    if not valid: return {}
    total, labeled = len(valid), [r for r in valid if normalize_name(r.get("true_class"))]
    sm = {"total_images": total, "valid_images": len(valid), "errors": sum(r.get("verdict") == "ERROR" for r in results), "cnn_in_verifier_scope": sum(r.get("cnn_class") in TARGET_CLASSES for r in valid)}
    for v in ["SUPPORTED", "WEAK", "AMBIGUOUS", "OUT_OF_SCOPE"]: sm[f"{v.lower()}_count"] = sum(r.get("verdict") == v for r in valid)
    for v in ["supported", "weak", "ambiguous"]: sm[f"{v}_rate"] = sm[f"{v}_count"] / total if total else 0.0
    if labeled: sm.update({"labeled_images": len(labeled), "cnn_accuracy": sum(r["true_class"] == r["cnn_class"] for r in labeled) / len(labeled)})
    return sm

def print_global_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 90 + "\nGLOBAL VERIFIER SUMMARY\n" + "=" * 90)
    for k, v in summary.items(): print(f"{k:<30}: {v:.4f}" if isinstance(v, float) else f"{k:<30}: {v}")

def build_evidence_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        for c, pfx in [("01_Good", "good"), ("02_Out_of_Focus", "out_of_focus"), ("03_Tracking_Error", "tracking_error")]:
            rows.append({"image": r.get("image", ""), "filename": r.get("filename", ""), "true_class": r.get("true_class", ""), "cnn_class": r.get("cnn_class", ""), "cnn_confidence": r.get("cnn_confidence", np.nan), "verifier_class": c, "class_specific_evidence": r.get(f"{pfx}_class_specific", np.nan), "pairwise_evidence": r.get(f"{pfx}_pairwise", np.nan), "final_evidence": r.get(f"{pfx}_evidence", np.nan), "best_rival": r.get("best_rival", ""), "margin": r.get("pairwise_margin", np.nan), "verdict": r.get("verdict", "")})
    return pd.DataFrame(rows)

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Astronomy Evidence Verifier Tester")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="ทดสอบภาพเดียว")
    mode.add_argument("--folder", type=Path, help="ทดสอบทุกภาพใน folder")
    mode.add_argument("--csv", type=Path, help="ทดสอบภาพจาก CSV")
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER, help="Path ของ astro_rule_classifier_new.py")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR, help="Folder ที่มี CSV analysis")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Path ของ ONNX model")
    parser.add_argument("--output-dir", type=Path, default=None, help="Folder สำหรับ output CSV")
    parser.add_argument("--image-column", type=str, default=None, help="ชื่อ column ที่เก็บ image path")
    parser.add_argument("--label-column", type=str, default=None, help="ชื่อ column ที่เก็บ true class")
    parser.add_argument("--max-images", type=int, default=None, help="จำกัดจำนวนภาพ")
    args = parser.parse_args()

    clf_path, an_dir, onnx_path = args.classifier.resolve(), args.analysis_dir.resolve(), args.onnx.resolve()
    out_dir = args.output_dir.resolve() if args.output_dir else (an_dir / "verifier_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90 + "\nASTRONOMY EVIDENCE VERIFIER TEST\n" + "=" * 90)
    print(f"Classifier : {clf_path}\nAnalysis   : {an_dir}\nONNX       : {onnx_path}\nOutput     : {out_dir}\n\nVerifier target classes:\n" + "\n".join(f"  - {c}" for c in TARGET_CLASSES))

    req_files = [an_dir / f for f in ["feature_distribution.csv", "verifier_feature_stability.csv", "pairwise_feature_stability.csv"]]
    missing_f = [p for p in req_files if not p.exists()]
    if missing_f: raise SystemExit("\n[ERROR] ไม่พบไฟล์ analysis:\n" + "\n".join(f"  - {p}" for p in missing_f) + "\n\nให้รัน analyze_rule_features.py ก่อน")

    classifier = import_classifier(clf_path)
    cls_names = list(getattr(classifier, "CLASS_NAMES", []))
    print(f"\n[INFO] CNN classes: {cls_names}")
    missing_tc = [c for c in TARGET_CLASSES if c not in cls_names]
    if missing_tc: raise SystemExit("\n[ERROR] TARGET_CLASSES ไม่มีอยู่ใน classifier:\n" + "\n".join(f"  - {x}" for x in missing_tc))

    dists, stab = load_feature_distribution(an_dir / "feature_distribution.csv"), load_stability_csv(an_dir / "verifier_feature_stability.csv")
    pw, cands = load_pairwise_stability(an_dir / "pairwise_feature_stability.csv"), load_candidates(an_dir / "verifier_evidence_candidates.csv")

    results, ev_rows = [], []
    if args.image:
        img_path = args.image.resolve()
        if not img_path.exists(): raise SystemExit(f"\n[ERROR] ไม่พบ image:\n{img_path}")
        res, cs_det, pw_det = run_one_image(img_path, classifier, onnx_path, dists, stab, pw, cands, infer_true_class_from_path(img_path))
        results.append(res); ev_rows.extend(cs_det + pw_det); print_single_result(res)
    elif args.folder: results, ev_rows = run_folder_mode(args.folder.resolve(), classifier, onnx_path, dists, stab, pw, cands, args.max_images)
    elif args.csv: results, ev_rows = run_csv_mode(args.csv.resolve(), classifier, onnx_path, dists, stab, pw, cands, args.image_column, args.label_column, args.max_images)

    print("\n" + "=" * 90 + "\nSAVE VERIFIER RESULTS\n" + "=" * 90)
    write_csv_rows(out_dir / "verifier_test_results.csv", results)
    if results:
        pd.DataFrame(build_evidence_summary(results)).to_csv(out_dir / "verifier_evidence_summary.csv", index=False, encoding="utf-8-sig")
        print(f"[SAVED] {(out_dir / 'verifier_evidence_summary.csv').resolve()}")
    if ev_rows:
        pd.DataFrame(ev_rows).to_csv(out_dir / "verifier_evidence_details.csv", index=False, encoding="utf-8-sig")
        print(f"[SAVED] {(out_dir / 'verifier_evidence_details.csv').resolve()}")
    
    conf_df = build_confusion_summary(results)
    if not conf_df.empty:
        conf_df.to_csv(out_dir / "verifier_confusion_summary.csv", index=False, encoding="utf-8-sig")
        print(f"[SAVED] {(out_dir / 'verifier_confusion_summary.csv').resolve()}")
    else: print("[INFO] ไม่มี true_class จึงยังสร้าง confusion summary ไม่ได้")

    print_global_summary(build_global_summary(results))

    if len(results) > 1:
        print("\n" + "=" * 90 + "\nVERDICT COUNTS\n" + "=" * 90)
        vd_counts = {}
        for r in results:
            vd = r.get("verdict", "UNKNOWN")
            vd_counts[vd] = vd_counts.get(vd, 0) + 1
        for vd, cnt in sorted(vd_counts.items()): print(f"{vd:<20}: {cnt}")

    print("\n" + "=" * 90 + "\nDONE\n" + "=" * 90 + f"\nOutput folder:\n{out_dir.resolve()}\n\nIMPORTANT:\nCNN class ยังคงเป็น prediction หลัก และ Verifier เป็น evidence layer เท่านั้น\nVerifier ไม่ได้เปลี่ยน CNN class\nVerifier ไม่ได้ใช้ fuse_scores()")

if __name__ == "__main__": main()