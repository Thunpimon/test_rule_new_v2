# # ทดสอบการทำ เพิ่ม FWHM-baseline + Eccentricity evidence ทดสอบเฉพาะคู่ 02_Out_of_Focus ↔ 03_Tracking_Error
# # วัด Accuracy, Balanced Accuracy, Macro-F1, confusion matrix

# from __future__ import annotations

# import argparse
# import json
# import math
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import cv2
# import numpy as np
# import csv

# try:
#     from sklearn.metrics import (
#         accuracy_score,
#         balanced_accuracy_score,
#         classification_report,
#         confusion_matrix,
#         f1_score,
#     )
#     from sklearn.model_selection import StratifiedKFold
# except ImportError as exc:
#     raise SystemExit(
#         "ต้องติดตั้ง scikit-learn ก่อน:\n"
#         "pip install scikit-learn"
#     ) from exc


# # ============================================================
# # Configuration
# # ============================================================

# CLASS_OOF = "02_Out_of_Focus"
# CLASS_TE = "03_Tracking_Error"

# VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# # ============================================================
# # Data structures
# # ============================================================

# @dataclass
# class StarEvidence:
#     image_path: str
#     true_class: str

#     star_count: int
#     valid_profile_count: int

#     median_fwhm: float
#     median_eccentricity: float
#     median_fwhm_ratio: float

#     elongated_star_ratio: float
#     mean_aspect_ratio: float
#     elongated_angle_consistency: float
#     elongated_radial_alignment: float

#     # New evidence
#     fwhm_relative: float
#     fwhm_evidence: float
#     eccentricity_evidence: float

#     # Combined evidence
#     tracking_fwhm_ecc_evidence: float
#     oof_fwhm_ecc_evidence: float


# # ============================================================
# # Utility functions
# # ============================================================

# def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
#     return float(max(minimum, min(maximum, value)))


# def score_high(
#     value: float,
#     weak_at: float,
#     strong_at: float,
# ) -> float:
#     if strong_at == weak_at:
#         return 1.0 if value >= strong_at else 0.0

#     return clamp(
#         (value - weak_at) / (strong_at - weak_at)
#     )


# def score_low(
#     value: float,
#     strong_at: float,
#     weak_at: float,
# ) -> float:
#     return 1.0 - score_high(
#         value,
#         strong_at,
#         weak_at,
#     )


# def weighted_mean(
#     parts: List[Tuple[float, float]]
# ) -> float:
#     total_weight = 0.0
#     total_score = 0.0

#     for score, weight in parts:
#         total_score += clamp(score) * weight
#         total_weight += weight

#     if total_weight <= 0:
#         return 0.0

#     return total_score / total_weight


# # ============================================================
# # Image loading
# # ============================================================

# def read_gray_image(
#     image_path: str | Path,
# ) -> np.ndarray:

#     image = cv2.imread(
#         str(image_path),
#         cv2.IMREAD_UNCHANGED,
#     )

#     if image is None:
#         raise ValueError(
#             f"Cannot read image: {image_path}"
#         )

#     if image.ndim == 3:
#         image = cv2.cvtColor(
#             image,
#             cv2.COLOR_BGR2GRAY,
#         )

#     if image.dtype == np.uint8:
#         return image

#     image_f = image.astype(np.float32)

#     min_val = float(np.min(image_f))
#     max_val = float(np.max(image_f))

#     if max_val <= min_val:
#         return np.zeros(
#             image.shape,
#             dtype=np.uint8,
#         )

#     return cv2.normalize(
#         image_f,
#         None,
#         0,
#         255,
#         cv2.NORM_MINMAX,
#     ).astype(np.uint8)


# # ============================================================
# # Background / star detection
# # ============================================================

# def flatten_background(
#     gray: np.ndarray,
#     blur_size: int = 51,
# ) -> np.ndarray:

#     kernel = (
#         blur_size
#         if blur_size % 2 == 1
#         else blur_size + 1
#     )

#     kernel = max(kernel, 3)

#     background = cv2.GaussianBlur(
#         gray,
#         (kernel, kernel),
#         0,
#     )

#     flattened = cv2.subtract(
#         gray,
#         background,
#     )

#     return cv2.normalize(
#         flattened,
#         None,
#         0,
#         255,
#         cv2.NORM_MINMAX,
#     )


# def detect_star_contours(
#     gray: np.ndarray,
# ) -> List[np.ndarray]:

#     flat = flatten_background(gray)

#     mean_val, std_val = cv2.meanStdDev(flat)

#     threshold = max(
#         float(mean_val[0][0] + 2.0 * std_val[0][0]),
#         35.0,
#     )

#     _, mask = cv2.threshold(
#         flat,
#         threshold,
#         255,
#         cv2.THRESH_BINARY,
#     )

#     kernel = np.ones(
#         (3, 3),
#         np.uint8,
#     )

#     mask = cv2.morphologyEx(
#         mask,
#         cv2.MORPH_OPEN,
#         kernel,
#     )

#     contours, _ = cv2.findContours(
#         mask,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE,
#     )

#     return list(contours)


# # ============================================================
# # Star profile
# # ============================================================

# def measure_star_profile(
#     gray: np.ndarray,
#     contour: np.ndarray,
# ) -> Dict[str, float] | None:

#     x, y, w, h = cv2.boundingRect(contour)

#     pad = int(
#         max(
#             8,
#             min(
#                 32,
#                 max(w, h) * 2,
#             ),
#         )
#     )

#     y0 = max(0, y - pad)
#     y1 = min(
#         gray.shape[0],
#         y + h + pad,
#     )

#     x0 = max(0, x - pad)
#     x1 = min(
#         gray.shape[1],
#         x + w + pad,
#     )

#     patch = gray[
#         y0:y1,
#         x0:x1,
#     ].astype(np.float64)

#     if patch.size < 25:
#         return None

#     border = np.concatenate(
#         [
#             patch[0, :],
#             patch[-1, :],
#             patch[:, 0],
#             patch[:, -1],
#         ]
#     )

#     background = float(
#         np.median(border)
#     )

#     signal = np.maximum(
#         patch - background,
#         0.0,
#     )

#     total_flux = float(
#         np.sum(signal)
#     )

#     peak_flux = float(
#         np.max(signal)
#     )

#     if total_flux <= 1e-6:
#         return None

#     if peak_flux < 3.0:
#         return None

#     yy, xx = np.indices(
#         patch.shape,
#         dtype=np.float64,
#     )

#     cx = float(
#         np.sum(xx * signal)
#         / total_flux
#     )

#     cy = float(
#         np.sum(yy * signal)
#         / total_flux
#     )

#     dx = xx - cx
#     dy = yy - cy

#     var_x = float(
#         np.sum(signal * dx * dx)
#         / total_flux
#     )

#     var_y = float(
#         np.sum(signal * dy * dy)
#         / total_flux
#     )

#     cov_xy = float(
#         np.sum(signal * dx * dy)
#         / total_flux
#     )

#     cov = np.array(
#         [
#             [var_x, cov_xy],
#             [cov_xy, var_y],
#         ],
#         dtype=np.float64,
#     )

#     eigvals = np.linalg.eigvalsh(cov)

#     sigma_minor = math.sqrt(
#         max(
#             float(eigvals[0]),
#             1e-6,
#         )
#     )

#     sigma_major = math.sqrt(
#         max(
#             float(eigvals[1]),
#             1e-6,
#         )
#     )

#     fwhm_major = (
#         2.3548 * sigma_major
#     )

#     fwhm_minor = (
#         2.3548 * sigma_minor
#     )

#     fwhm = (
#         fwhm_major + fwhm_minor
#     ) / 2.0

#     fwhm_ratio = (
#         fwhm_major
#         / max(fwhm_minor, 1e-6)
#     )

#     eccentricity = math.sqrt(
#         max(
#             0.0,
#             1.0
#             - (
#                 sigma_minor
#                 * sigma_minor
#             )
#             / max(
#                 sigma_major
#                 * sigma_major,
#                 1e-6,
#             ),
#         )
#     )

#     return {
#         "fwhm": float(fwhm),
#         "fwhm_ratio": float(
#             fwhm_ratio
#         ),
#         "eccentricity": float(
#             eccentricity
#         ),
#     }


# # ============================================================
# # Extract features needed for experiment
# # ============================================================

# def extract_features(
#     gray: np.ndarray,
# ) -> Dict[str, float]:

#     h, w = gray.shape

#     contours = detect_star_contours(
#         gray
#     )

#     total_pixels = h * w

#     max_star_area = max(
#         5000.0,
#         total_pixels * 0.02,
#     )

#     star_count = 0

#     fwhms = []
#     eccentricities = []
#     fwhm_ratios = []

#     elongated_count = 0
#     aspect_ratios = []
#     elongated_angles = []
#     radial_alignments = []

#     img_cx = w / 2.0
#     img_cy = h / 2.0

#     for contour in contours:

#         area = float(
#             cv2.contourArea(contour)
#         )

#         if area < 3.0:
#             continue

#         if area > max_star_area:
#             continue

#         perimeter = float(
#             cv2.arcLength(
#                 contour,
#                 True,
#             )
#         )

#         if perimeter <= 0:
#             continue

#         x, y, bw, bh = cv2.boundingRect(
#             contour
#         )

#         aspect_ratio = (
#             max(bw, bh)
#             / max(
#                 min(bw, bh),
#                 1,
#             )
#         )

#         star_count += 1

#         aspect_ratios.append(
#             float(aspect_ratio)
#         )

#         profile = measure_star_profile(
#             gray,
#             contour,
#         )

#         if profile is not None:
#             fwhms.append(
#                 profile["fwhm"]
#             )

#             eccentricities.append(
#                 profile["eccentricity"]
#             )

#             fwhm_ratios.append(
#                 profile["fwhm_ratio"]
#             )

#         if (
#             aspect_ratio > 1.45
#             or (
#                 area > 80.0
#                 and aspect_ratio > 1.25
#             )
#         ):

#             elongated_count += 1

#             rect = cv2.minAreaRect(
#                 contour
#             )

#             (_, _), (rw, rh), angle = rect

#             if rw < rh:
#                 angle += 90.0

#             angle %= 180.0

#             elongated_angles.append(
#                 angle
#             )

#             star_cx = (
#                 x + bw / 2.0
#             )

#             star_cy = (
#                 y + bh / 2.0
#             )

#             dx = star_cx - img_cx
#             dy = star_cy - img_cy

#             if abs(dx) < 1e-6 and abs(dy) < 1e-6:
#                 radial_angle = 0.0
#             else:
#                 radial_angle = (
#                     math.degrees(
#                         math.atan2(
#                             dy,
#                             dx,
#                         )
#                     )
#                     % 180.0
#                 )

#             diff = abs(
#                 angle - radial_angle
#             ) % 180.0

#             if diff > 90.0:
#                 diff = 180.0 - diff

#             alignment = (
#                 1.0 - diff / 90.0
#             )

#             radial_alignments.append(
#                 alignment
#             )

#     return {
#         "star_count": star_count,

#         "valid_profile_count": len(
#             fwhms
#         ),

#         "median_fwhm": (
#             float(np.median(fwhms))
#             if fwhms
#             else 0.0
#         ),

#         "median_eccentricity": (
#             float(
#                 np.median(
#                     eccentricities
#                 )
#             )
#             if eccentricities
#             else 0.0
#         ),

#         "median_fwhm_ratio": (
#             float(
#                 np.median(
#                     fwhm_ratios
#                 )
#             )
#             if fwhm_ratios
#             else 0.0
#         ),

#         "elongated_star_ratio": (
#             elongated_count
#             / star_count
#             if star_count
#             else 0.0
#         ),

#         "mean_aspect_ratio": (
#             float(
#                 np.mean(
#                     aspect_ratios
#                 )
#             )
#             if aspect_ratios
#             else 0.0
#         ),

#         "elongated_angle_consistency": (
#             axial_angle_consistency(
#                 elongated_angles
#             )
#         ),

#         "elongated_radial_alignment": (
#             float(
#                 np.mean(
#                     radial_alignments
#                 )
#             )
#             if radial_alignments
#             else 0.0
#         ),
#     }


# # ============================================================
# # Circular statistics
# # ============================================================

# def axial_angle_consistency(
#     angles: List[float],
# ) -> float:

#     if len(angles) < 2:
#         return 0.0

#     doubled = np.deg2rad(
#         np.asarray(
#             angles,
#             dtype=np.float64,
#         )
#         * 2.0
#     )

#     mean_cos = float(
#         np.mean(np.cos(doubled))
#     )

#     mean_sin = float(
#         np.mean(np.sin(doubled))
#     )

#     resultant = math.hypot(
#         mean_cos,
#         mean_sin,
#     )

#     return clamp(resultant)


# # ============================================================
# # NEW:
# # FWHM baseline + eccentricity evidence
# # ============================================================

# def compute_fwhm_ecc_evidence(
#     median_fwhm: float,
#     median_eccentricity: float,
#     fwhm_base: float,
#     ecc_base: float,
# ) -> Tuple[float, float, float, float]:

#     if fwhm_base <= 0:
#         raise ValueError(
#             "fwhm_base must be > 0"
#         )

#     # --------------------------------------------------------
#     # Relative FWHM
#     #
#     # Example:
#     # FWHM = 26
#     # baseline = 20
#     #
#     # relative = 1.30
#     # --------------------------------------------------------

#     fwhm_relative = (
#         median_fwhm
#         / fwhm_base
#         if median_fwhm > 0
#         else 0.0
#     )

#     # --------------------------------------------------------
#     # FWHM evidence for Tracking Error
#     #
#     # 1.10 = starts becoming suspicious
#     # 1.30 = strong evidence
#     # --------------------------------------------------------

#     fwhm_evidence = score_high(
#         fwhm_relative,
#         1.10,
#         1.30,
#     )

#     # --------------------------------------------------------
#     # Eccentricity evidence
#     #
#     # ecc <= baseline  -> round
#     # ecc > baseline   -> elongated
#     #
#     # 0.45 -> baseline
#     # 0.65 -> strong elongation
#     # --------------------------------------------------------

#     eccentricity_evidence = score_high(
#         median_eccentricity,
#         ecc_base,
#         0.65,
#     )

#     # --------------------------------------------------------
#     # Tracking Error:
#     #
#     #       FWHM high
#     #           AND
#     #       eccentricity high
#     #
#     # Multiplication intentionally requires both.
#     # --------------------------------------------------------

#     tracking_evidence = (
#         fwhm_evidence
#         * eccentricity_evidence
#     )

#     # --------------------------------------------------------
#     # Out of Focus:
#     #
#     #       FWHM high
#     #           AND
#     #       eccentricity LOW
#     # --------------------------------------------------------

#     roundness_evidence = score_low(
#         median_eccentricity,
#         ecc_base,
#         0.65,
#     )

#     oof_evidence = (
#         fwhm_evidence
#         * roundness_evidence
#     )

#     return (
#         float(fwhm_relative),
#         float(fwhm_evidence),
#         float(eccentricity_evidence),
#         float(tracking_evidence),
#         float(oof_evidence),
#     )


# # ============================================================
# # Existing-style scores
# # ============================================================

# def baseline_oof_score(
#     f: Dict[str, float],
# ) -> float:

#     return weighted_mean(
#         [
#             (
#                 score_high(
#                     f["median_fwhm"],
#                     18.0,
#                     30.0,
#                 ),
#                 1.1,
#             ),

#             (
#                 score_low(
#                     f["median_eccentricity"],
#                     0.72,
#                     0.82,
#                 ),
#                 0.8,
#             ),

#             (
#                 score_high(
#                     f["elongated_star_ratio"],
#                     0.0,
#                     0.10,
#                 ),
#                 0.4,
#             ),
#         ]
#     )


# def baseline_te_score(
#     f: Dict[str, float],
# ) -> float:

#     return weighted_mean(
#         [
#             (
#                 score_high(
#                     f["elongated_star_ratio"],
#                     0.10,
#                     0.25,
#                 ),
#                 1.3,
#             ),

#             (
#                 score_high(
#                     f["mean_aspect_ratio"],
#                     1.12,
#                     1.30,
#                 ),
#                 1.0,
#             ),

#             (
#                 score_high(
#                     f["median_eccentricity"],
#                     0.35,
#                     0.65,
#                 ),
#                 1.1,
#             ),

#             (
#                 score_high(
#                     f["median_fwhm_ratio"],
#                     1.06,
#                     1.25,
#                 ),
#                 1.0,
#             ),

#             (
#                 score_high(
#                     f["elongated_angle_consistency"],
#                     0.35,
#                     0.75,
#                 ),
#                 0.25,
#             ),
#         ]
#     )


# # ============================================================
# # Enhanced scores
# # ============================================================

# def enhanced_oof_score(
#     f: Dict[str, float],
#     oof_evidence: float,
# ) -> float:

#     base = baseline_oof_score(f)

#     # Add new FWHM + roundness evidence
#     return weighted_mean(
#         [
#             (base, 3.0),
#             (oof_evidence, 1.5),
#         ]
#     )


# def enhanced_te_score(
#     f: Dict[str, float],
#     tracking_evidence: float,
# ) -> float:

#     base = baseline_te_score(f)

#     # Add new FWHM + eccentricity evidence
#     return weighted_mean(
#         [
#             (base, 3.0),
#             (tracking_evidence, 1.5),
#         ]
#     )


# # ============================================================
# # Dataset loading
# # ============================================================

# def collect_dataset(
#     root: Path,
#     fwhm_base: float,
#     ecc_base: float,
# ) -> List[StarEvidence]:

#     classes = {
#         CLASS_OOF: root / CLASS_OOF,
#         CLASS_TE: root / CLASS_TE,
#     }

#     records: List[StarEvidence] = []

#     for class_name, class_dir in classes.items():

#         if not class_dir.exists():
#             raise FileNotFoundError(
#                 f"ไม่พบ folder: {class_dir}"
#             )

#         image_paths = sorted(
#             p
#             for p in class_dir.rglob("*")
#             if p.suffix.lower()
#             in VALID_EXTENSIONS
#         )

#         print(
#             f"{class_name}: "
#             f"{len(image_paths)} images"
#         )

#         for index, image_path in enumerate(
#             image_paths,
#             start=1,
#         ):

#             try:
#                 gray = read_gray_image(
#                     image_path
#                 )

#                 f = extract_features(
#                     gray
#                 )

#                 (
#                     fwhm_relative,
#                     fwhm_evidence,
#                     eccentricity_evidence,
#                     tracking_evidence,
#                     oof_evidence,
#                 ) = compute_fwhm_ecc_evidence(
#                     f["median_fwhm"],
#                     f["median_eccentricity"],
#                     fwhm_base,
#                     ecc_base,
#                 )

#                 records.append(
#                     StarEvidence(
#                         image_path=str(
#                             image_path
#                         ),
#                         true_class=class_name,

#                         star_count=int(
#                             f["star_count"]
#                         ),

#                         valid_profile_count=int(
#                             f[
#                                 "valid_profile_count"
#                             ]
#                         ),

#                         median_fwhm=float(
#                             f["median_fwhm"]
#                         ),

#                         median_eccentricity=float(
#                             f[
#                                 "median_eccentricity"
#                             ]
#                         ),

#                         median_fwhm_ratio=float(
#                             f[
#                                 "median_fwhm_ratio"
#                             ]
#                         ),

#                         elongated_star_ratio=float(
#                             f[
#                                 "elongated_star_ratio"
#                             ]
#                         ),

#                         mean_aspect_ratio=float(
#                             f[
#                                 "mean_aspect_ratio"
#                             ]
#                         ),

#                         elongated_angle_consistency=float(
#                             f[
#                                 "elongated_angle_consistency"
#                             ]
#                         ),

#                         elongated_radial_alignment=float(
#                             f[
#                                 "elongated_radial_alignment"
#                             ]
#                         ),

#                         fwhm_relative=fwhm_relative,
#                         fwhm_evidence=fwhm_evidence,
#                         eccentricity_evidence=eccentricity_evidence,

#                         tracking_fwhm_ecc_evidence=tracking_evidence,
#                         oof_fwhm_ecc_evidence=oof_evidence,
#                     )
#                 )

#             except Exception as exc:
#                 print(
#                     f"[WARN] {image_path}: "
#                     f"{exc}"
#                 )

#             if index % 50 == 0:
#                 print(
#                     f"  processed "
#                     f"{index}/{len(image_paths)}"
#                 )

#     return records


# # ============================================================
# # Prediction
# # ============================================================

# def predict_baseline(
#     record: StarEvidence,
# ) -> str:

#     f = asdict(record)

#     oof = baseline_oof_score(f)
#     te = baseline_te_score(f)

#     return (
#         CLASS_OOF
#         if oof >= te
#         else CLASS_TE
#     )


# def predict_enhanced(
#     record: StarEvidence,
# ) -> str:

#     f = asdict(record)

#     oof = enhanced_oof_score(
#         f,
#         record.oof_fwhm_ecc_evidence,
#     )

#     te = enhanced_te_score(
#         f,
#         record.tracking_fwhm_ecc_evidence,
#     )

#     return (
#         CLASS_OOF
#         if oof >= te
#         else CLASS_TE
#     )


# # ============================================================
# # Evaluation
# # ============================================================

# def evaluate_predictions(
#     y_true: List[str],
#     y_pred: List[str],
# ) -> Dict[str, object]:

#     labels = [
#         CLASS_OOF,
#         CLASS_TE,
#     ]

#     cm = confusion_matrix(
#         y_true,
#         y_pred,
#         labels=labels,
#     )

#     return {
#         "accuracy": float(
#             accuracy_score(
#                 y_true,
#                 y_pred,
#             )
#         ),

#         "balanced_accuracy": float(
#             balanced_accuracy_score(
#                 y_true,
#                 y_pred,
#             )
#         ),

#         "macro_f1": float(
#             f1_score(
#                 y_true,
#                 y_pred,
#                 labels=labels,
#                 average="macro",
#                 zero_division=0,
#             )
#         ),

#         "confusion_matrix": cm.tolist(),

#         "classification_report": classification_report(
#             y_true,
#             y_pred,
#             labels=labels,
#             zero_division=0,
#             output_dict=True,
#         ),
#     }


# # ============================================================
# # 5-fold experiment
# # ============================================================

# def run_5fold(
#     records: List[StarEvidence],
#     random_state: int = 42,
# ) -> Dict[str, object]:

#     y = np.asarray(
#         [
#             r.true_class
#             for r in records
#         ]
#     )

#     indices = np.arange(
#         len(records)
#     )

#     skf = StratifiedKFold(
#         n_splits=5,
#         shuffle=True,
#         random_state=random_state,
#     )

#     baseline_scores = []
#     enhanced_scores = []

#     fold_results = []

#     for fold, (
#         train_idx,
#         test_idx,
#     ) in enumerate(
#         skf.split(indices, y),
#         start=1,
#     ):

#         test_records = [
#             records[i]
#             for i in test_idx
#         ]

#         y_true = [
#             r.true_class
#             for r in test_records
#         ]

#         y_base = [
#             predict_baseline(r)
#             for r in test_records
#         ]

#         y_enhanced = [
#             predict_enhanced(r)
#             for r in test_records
#         ]

#         base_metrics = evaluate_predictions(
#             y_true,
#             y_base,
#         )

#         enhanced_metrics = evaluate_predictions(
#             y_true,
#             y_enhanced,
#         )

#         baseline_scores.append(
#             base_metrics
#         )

#         enhanced_scores.append(
#             enhanced_metrics
#         )

#         fold_results.append(
#             {
#                 "fold": fold,
#                 "baseline": base_metrics,
#                 "enhanced": enhanced_metrics,
#             }
#         )

#         print()
#         print(
#             f"========== Fold {fold} =========="
#         )

#         print(
#             "Baseline:"
#             f" accuracy={base_metrics['accuracy']:.4f}"
#             f" balanced_acc={base_metrics['balanced_accuracy']:.4f}"
#             f" macro_f1={base_metrics['macro_f1']:.4f}"
#         )

#         print(
#             "Enhanced:"
#             f" accuracy={enhanced_metrics['accuracy']:.4f}"
#             f" balanced_acc={enhanced_metrics['balanced_accuracy']:.4f}"
#             f" macro_f1={enhanced_metrics['macro_f1']:.4f}"
#         )

#     def summarize(
#         results: List[Dict[str, object]]
#     ) -> Dict[str, float]:

#         return {
#             metric: float(
#                 np.mean(
#                     [
#                         r[metric]
#                         for r in results
#                     ]
#                 )
#             )
#             for metric in [
#                 "accuracy",
#                 "balanced_accuracy",
#                 "macro_f1",
#             ]
#         }

#     base_summary = summarize(
#         baseline_scores
#     )

#     enhanced_summary = summarize(
#         enhanced_scores
#     )

#     improvement = {
#         metric: (
#             enhanced_summary[metric]
#             - base_summary[metric]
#         )
#         for metric in base_summary
#     }

#     return {
#         "baseline_mean": base_summary,
#         "enhanced_mean": enhanced_summary,
#         "improvement": improvement,
#         "folds": fold_results,
#     }


# # ============================================================
# # Parameter sweep
# # ============================================================

# def parameter_sweep(
#     records: List[StarEvidence],
# ) -> List[Dict[str, float]]:

#     fwhm_bases = [
#         15.0,
#         18.0,
#         20.0,
#         22.0,
#         25.0,
#         28.0,
#         30.0,
#     ]

#     ecc_bases = [
#         0.40,
#         0.42,
#         0.45,
#         0.48,
#         0.50,
#     ]

#     results = []

#     for fwhm_base in fwhm_bases:

#         for ecc_base in ecc_bases:

#             updated_records = []

#             for record in records:

#                 (
#                     fwhm_relative,
#                     fwhm_evidence,
#                     eccentricity_evidence,
#                     tracking_evidence,
#                     oof_evidence,
#                 ) = compute_fwhm_ecc_evidence(
#                     record.median_fwhm,
#                     record.median_eccentricity,
#                     fwhm_base,
#                     ecc_base,
#                 )

#                 updated = StarEvidence(
#                     **{
#                         **asdict(record),
#                         "fwhm_relative": fwhm_relative,
#                         "fwhm_evidence": fwhm_evidence,
#                         "eccentricity_evidence": eccentricity_evidence,
#                         "tracking_fwhm_ecc_evidence": tracking_evidence,
#                         "oof_fwhm_ecc_evidence": oof_evidence,
#                     }
#                 )

#                 updated_records.append(
#                     updated
#                 )

#             y_true = [
#                 r.true_class
#                 for r in updated_records
#             ]

#             y_pred = [
#                 predict_enhanced(r)
#                 for r in updated_records
#             ]

#             metrics = evaluate_predictions(
#                 y_true,
#                 y_pred,
#             )

#             results.append(
#                 {
#                     "fwhm_base": fwhm_base,
#                     "ecc_base": ecc_base,
#                     "accuracy": metrics[
#                         "accuracy"
#                     ],
#                     "balanced_accuracy": metrics[
#                         "balanced_accuracy"
#                     ],
#                     "macro_f1": metrics[
#                         "macro_f1"
#                     ],
#                 }
#             )

#     results.sort(
#         key=lambda x: (
#             x["balanced_accuracy"],
#             x["macro_f1"],
#         ),
#         reverse=True,
#     )

#     return results


# # ============================================================
# # Evidence CSV
# # ============================================================

# def save_evidence_csv(
#     records: List[StarEvidence],
#     output_path: Path,
# ) -> None:

#     output_path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     fieldnames = list(
#         asdict(records[0]).keys()
#     )

#     with output_path.open(
#         "w",
#         newline="",
#         encoding="utf-8-sig",
#     ) as f:

#         writer = csv.DictWriter(
#             f,
#             fieldnames=fieldnames,
#         )

#         writer.writeheader()

#         for record in records:
#             writer.writerow(
#                 asdict(record)
#             )


# # ============================================================
# # Main
# # ============================================================

# def main() -> None:

#     parser = argparse.ArgumentParser(
#         description=(
#             "Test FWHM-baseline + eccentricity "
#             "evidence for OOF vs Tracking Error"
#         )
#     )

#     parser.add_argument(
#         "--dataset",
#         required=True,
#         help=(
#             "Dataset root containing "
#             "02_Out_of_Focus/ and "
#             "03_Tracking_Error/"
#         ),
#     )

#     parser.add_argument(
#         "--fwhm-base",
#         type=float,
#         default=20.0,
#         help="FWHM baseline",
#     )

#     parser.add_argument(
#         "--ecc-base",
#         type=float,
#         default=0.45,
#         help="Eccentricity baseline",
#     )

#     parser.add_argument(
#         "--output",
#         default="fwhm_ecc_experiment",
#         help="Output directory",
#     )

#     parser.add_argument(
#         "--sweep",
#         action="store_true",
#         help="Run FWHM/Eccentricity parameter sweep",
#     )

#     args = parser.parse_args()

#     dataset = Path(
#         args.dataset
#     )

#     output = Path(
#         args.output
#     )

#     output.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     print()
#     print("=" * 70)
#     print(
#         "FWHM BASELINE + ECCENTRICITY "
#         "EVIDENCE EXPERIMENT"
#     )
#     print("=" * 70)

#     print(
#         f"Dataset       : {dataset}"
#     )

#     print(
#         f"FWHM baseline : {args.fwhm_base}"
#     )

#     print(
#         f"Ecc baseline  : {args.ecc_base}"
#     )

#     # --------------------------------------------------------
#     # Extract
#     # --------------------------------------------------------

#     records = collect_dataset(
#         dataset,
#         args.fwhm_base,
#         args.ecc_base,
#     )

#     if not records:
#         raise SystemExit(
#             "ไม่พบข้อมูล"
#         )

#     save_evidence_csv(
#         records,
#         output / "evidence.csv",
#     )

#     print()
#     print(
#         f"Saved: {output / 'evidence.csv'}"
#     )

#     # --------------------------------------------------------
#     # Show feature distribution
#     # --------------------------------------------------------

#     print()
#     print("=" * 70)
#     print("FEATURE DISTRIBUTION")
#     print("=" * 70)

#     for class_name in [
#         CLASS_OOF,
#         CLASS_TE,
#     ]:

#         subset = [
#             r
#             for r in records
#             if r.true_class == class_name
#         ]

#         print()
#         print(class_name)

#         print(
#             f"  median FWHM          : "
#             f"{np.median([r.median_fwhm for r in subset]):.4f}"
#         )

#         print(
#             f"  median eccentricity  : "
#             f"{np.median([r.median_eccentricity for r in subset]):.4f}"
#         )

#         print(
#             f"  median FWHM ratio    : "
#             f"{np.median([r.median_fwhm_ratio for r in subset]):.4f}"
#         )

#         print(
#             f"  median FWHM relative : "
#             f"{np.median([r.fwhm_relative for r in subset]):.4f}"
#         )

#         print(
#             f"  median TE evidence   : "
#             f"{np.median([r.tracking_fwhm_ecc_evidence for r in subset]):.4f}"
#         )

#         print(
#             f"  median OOF evidence  : "
#             f"{np.median([r.oof_fwhm_ecc_evidence for r in subset]):.4f}"
#         )

#     # --------------------------------------------------------
#     # 5-fold
#     # --------------------------------------------------------

#     result = run_5fold(
#         records
#     )

#     print()
#     print("=" * 70)
#     print("5-FOLD RESULT")
#     print("=" * 70)

#     print()
#     print("BASELINE")

#     for key, value in result[
#         "baseline_mean"
#     ].items():

#         print(
#             f"  {key:20s}: "
#             f"{value:.4f}"
#         )

#     print()
#     print("ENHANCED")

#     for key, value in result[
#         "enhanced_mean"
#     ].items():

#         print(
#             f"  {key:20s}: "
#             f"{value:.4f}"
#         )

#     print()
#     print("IMPROVEMENT")

#     for key, value in result[
#         "improvement"
#     ].items():

#         sign = "+" if value >= 0 else ""

#         print(
#             f"  {key:20s}: "
#             f"{sign}{value:.4f}"
#         )

#     # --------------------------------------------------------
#     # Parameter sweep
#     # --------------------------------------------------------

#     if args.sweep:

#         print()
#         print("=" * 70)
#         print(
#             "PARAMETER SWEEP"
#         )
#         print("=" * 70)

#         sweep = parameter_sweep(
#             records
#         )

#         sweep_path = (
#             output
#             / "parameter_sweep.csv"
#         )

#         with sweep_path.open(
#             "w",
#             newline="",
#             encoding="utf-8-sig",
#         ) as f:

#             writer = csv.DictWriter(
#                 f,
#                 fieldnames=[
#                     "fwhm_base",
#                     "ecc_base",
#                     "accuracy",
#                     "balanced_accuracy",
#                     "macro_f1",
#                 ],
#             )

#             writer.writeheader()

#             writer.writerows(
#                 sweep
#             )

#         print()

#         print(
#             "Top 10 parameter combinations:"
#         )

#         for row in sweep[:10]:

#             print(
#                 f"  FWHM={row['fwhm_base']:5.1f} "
#                 f"ECC={row['ecc_base']:.2f} "
#                 f"BAL_ACC={row['balanced_accuracy']:.4f} "
#                 f"F1={row['macro_f1']:.4f}"
#             )

#         print()
#         print(
#             f"Saved: {sweep_path}"
#         )

#     # --------------------------------------------------------
#     # JSON result
#     # --------------------------------------------------------

#     result_path = (
#         output
#         / "result.json"
#     )

#     with result_path.open(
#         "w",
#         encoding="utf-8",
#     ) as f:

#         json.dump(
#             result,
#             f,
#             ensure_ascii=False,
#             indent=2,
#         )

#     print()
#     print(
#         f"Saved: {result_path}"
#     )

#     print()
#     print("=" * 70)
#     print("DONE")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()



"""
analyze_verifier_features_v2.py

Evidence discovery pipeline สำหรับ Rule-Based Verifier
ของระบบวิเคราะห์คุณภาพภาพดาราศาสตร์

หลักการ:
1. ใช้ extract_features() จาก classifier จริง
2. รวม train + val สำหรับ discovery
3. วิเคราะห์ทุก numeric feature
4. วิเคราะห์ pairwise เฉพาะคู่ class ที่สนใจ
5. ใช้ 5-fold Stratified CV
6. sweep threshold จากข้อมูลจริง
7. ทดสอบทั้ง HIGH และ LOW direction
8. วัด:
   - ROC-AUC
   - balanced accuracy
   - F1
   - precision
   - recall
   - effect size
   - fold stability
9. สร้าง verifier_evidence_candidates.csv
10. test set ไม่ถูกใช้หา threshold
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# OPTIONAL SCIKIT-LEARN
# ============================================================

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold
except ImportError as exc:
    raise SystemExit(
        "\nไม่พบ scikit-learn\n"
        "ติดตั้งด้วย:\n"
        "pip install scikit-learn pandas numpy\n"
    ) from exc


# ============================================================
# CONFIG
# ============================================================

DEFAULT_CLASSIFIER = "./astro_rule_classifier_new.py"
DEFAULT_DATASET = "./Dataset_For_Rule_Base"
DEFAULT_OUTPUT = "./verifier_feature_analysis_v2"

DEFAULT_FOLDS = 5

CLASS_NAMES = [
    "01_Good",
    "02_Out_of_Focus",
    "03_Tracking_Error",
    "04_Over_Saturated",
    "05_No_Star",
    "06_Satellite",
]

PAIR_NAMES = [
    ("02_Out_of_Focus", "03_Tracking_Error"),
]

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".fits",
    ".fit",
    ".fts",
}


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value: Any) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float("nan")


def feature_dict_from_object(features: Any) -> Dict[str, Any]:
    """
    รองรับ:
    - dataclass
    - object.__dict__
    - dictionary
    """
    if isinstance(features, dict):
        return dict(features)

    if is_dataclass(features):
        return dict(asdict(features))

    if hasattr(features, "__dict__"):
        return dict(vars(features))

    raise TypeError(
        "extract_features() ไม่ได้คืน object ที่แปลงเป็น dict ได้"
    )


def import_module_from_file(path: Path):
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบ classifier: {path}"
        )

    module_name = f"astro_classifier_{abs(hash(str(path)))}"

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"ไม่สามารถ import classifier: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []

    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_split_dirs(dataset_root: Path) -> Dict[str, Path]:
    """
    รองรับทั้ง:

    Dataset/
        train/
        val/
        test/

    และกรณี Dataset root มี class folder โดยตรง:

    Dataset/
        01_Good/
        02_Out_of_Focus/
        ...
    """

    result = {}

    for split in ["train", "val", "test"]:
        p = dataset_root / split

        if p.exists() and p.is_dir():
            result[split] = p

    if result:
        return result

    # flat dataset
    class_dirs = [
        dataset_root / c
        for c in CLASS_NAMES
        if (dataset_root / c).exists()
    ]

    if class_dirs:
        result["all"] = dataset_root

    return result


def discover_class_dirs(split_dir: Path) -> Dict[str, Path]:
    result = {}

    for class_name in CLASS_NAMES:
        p = split_dir / class_name

        if p.exists() and p.is_dir():
            result[class_name] = p

    return result


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_dataset(
    classifier: Any,
    split_name: str,
    split_dir: Path,
) -> pd.DataFrame:

    print()
    print("=" * 80)
    print(f"FEATURE EXTRACTION: {split_name}")
    print(f"PATH: {split_dir.resolve()}")
    print("=" * 80)

    class_dirs = discover_class_dirs(split_dir)

    if not class_dirs:
        print("ไม่พบ class folder")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for class_name, class_dir in class_dirs.items():

        images = list_images(class_dir)

        print(
            f"\n{class_name}: {len(images)} images"
        )

        for i, image_path in enumerate(images, 1):

            try:

                gray = classifier.read_gray_image(
                    str(image_path)
                )

                features = classifier.extract_features(
                    gray
                )

                data = feature_dict_from_object(
                    features
                )

                row: Dict[str, Any] = {
                    "split": split_name,
                    "class": class_name,
                    "filename": image_path.name,
                    "relative_path": str(
                        image_path.relative_to(split_dir)
                    ),
                }

                for key, value in data.items():

                    if isinstance(
                        value,
                        (bool, np.bool_),
                    ):
                        row[key] = int(value)

                    elif isinstance(
                        value,
                        (int, np.integer),
                    ):
                        row[key] = int(value)

                    elif isinstance(
                        value,
                        (float, np.floating),
                    ):
                        row[key] = float(value)

                    else:
                        converted = safe_float(value)

                        if math.isfinite(converted):
                            row[key] = converted

                rows.append(row)

            except Exception as exc:

                print(
                    f"\n[ERROR] {image_path.name}: {exc}"
                )

            if i % 50 == 0 or i == len(images):
                print(
                    f"  processed {i}/{len(images)}",
                    end="\r",
                )

        print()

    return pd.DataFrame(rows)


# ============================================================
# FEATURE SELECTION
# ============================================================

def numeric_features(df: pd.DataFrame) -> List[str]:

    excluded = {
        "split",
        "class",
        "filename",
        "relative_path",
    }

    result = []

    for col in df.columns:

        if col in excluded:
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        valid = values.notna().sum()

        if valid < 10:
            continue

        result.append(col)

    return result


# ============================================================
# EFFECT SIZE
# ============================================================

def cohens_d(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return float("nan")

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled = math.sqrt(
        (
            (len(x) - 1) * vx
            + (len(y) - 1) * vy
        )
        / (
            len(x) + len(y) - 2
        )
    )

    if pooled <= 0:
        return 0.0

    return float(
        (np.mean(x) - np.mean(y))
        / pooled
    )


# ============================================================
# THRESHOLD GENERATION
# ============================================================

def make_thresholds(
    values: np.ndarray,
    n: int = 31,
) -> np.ndarray:

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.array([])

    q = np.linspace(
        0.02,
        0.98,
        n,
    )

    thresholds = np.quantile(
        values,
        q,
    )

    return np.unique(thresholds)


# ============================================================
# SINGLE FEATURE CLASSIFIER
# ============================================================

def predict_threshold(
    values: np.ndarray,
    threshold: float,
    direction: str,
) -> np.ndarray:

    if direction == "high":
        return (
            values >= threshold
        ).astype(int)

    return (
        values <= threshold
    ).astype(int)


# ============================================================
# FOLD EVALUATION
# ============================================================

def evaluate_candidate_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature: str,
    threshold: float,
    direction: str,
    positive_class: str,
) -> Dict[str, float]:

    train_values = pd.to_numeric(
        train_df[feature],
        errors="coerce",
    ).to_numpy(dtype=float)

    val_values = pd.to_numeric(
        val_df[feature],
        errors="coerce",
    ).to_numpy(dtype=float)

    train_mask = np.isfinite(train_values)
    val_mask = np.isfinite(val_values)

    if train_mask.sum() < 2:
        return {}

    if val_mask.sum() < 2:
        return {}

    y_train = (
        train_df["class"].to_numpy()
        == positive_class
    ).astype(int)

    y_val = (
        val_df["class"].to_numpy()
        == positive_class
    ).astype(int)

    thresholds = make_thresholds(
        train_values[train_mask]
    )

    if len(thresholds) == 0:
        return {}

    best = None

    for threshold_candidate in thresholds:

        pred = predict_threshold(
            val_values[val_mask],
            float(threshold_candidate),
            direction,
        )

        true = y_val[val_mask]

        if len(np.unique(true)) < 2:
            continue

        bal = balanced_accuracy_score(
            true,
            pred,
        )

        f1 = f1_score(
            true,
            pred,
            zero_division=0,
        )

        score = (
            0.65 * bal
            + 0.35 * f1
        )

        candidate = {
            "threshold": float(
                threshold_candidate
            ),
            "balanced_accuracy": float(bal),
            "f1": float(f1),
            "accuracy": float(
                accuracy_score(
                    true,
                    pred,
                )
            ),
            "precision": float(
                precision_score(
                    true,
                    pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    true,
                    pred,
                    zero_division=0,
                )
            ),
            "score": float(score),
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best or {}


# ============================================================
# GLOBAL PAIRWISE ANALYSIS
# ============================================================

def pairwise_feature_analysis(
    df: pd.DataFrame,
    class_a: str,
    class_b: str,
    features: Sequence[str],
) -> pd.DataFrame:

    pair = df[
        df["class"].isin(
            [class_a, class_b]
        )
    ].copy()

    records = []

    for feature in features:

        a = pd.to_numeric(
            pair[
                pair["class"] == class_a
            ][feature],
            errors="coerce",
        ).dropna().to_numpy()

        b = pd.to_numeric(
            pair[
                pair["class"] == class_b
            ][feature],
            errors="coerce",
        ).dropna().to_numpy()

        if len(a) < 5 or len(b) < 5:
            continue

        values = np.concatenate(
            [a, b]
        )

        labels = np.concatenate(
            [
                np.zeros(len(a)),
                np.ones(len(b)),
            ]
        )

        try:
            auc_raw = roc_auc_score(
                labels,
                values,
            )

            auc_low = auc_raw
            auc_high = 1.0 - auc_raw

        except Exception:
            auc_low = float("nan")
            auc_high = float("nan")

        records.append({
            "class_a": class_a,
            "class_b": class_b,
            "feature": feature,

            "n_a": len(a),
            "n_b": len(b),

            "a_mean": float(np.mean(a)),
            "b_mean": float(np.mean(b)),

            "a_median": float(np.median(a)),
            "b_median": float(np.median(b)),

            "a_std": float(np.std(a)),
            "b_std": float(np.std(b)),

            "effect_size_a_minus_b": cohens_d(
                a,
                b,
            ),

            "auc_low": float(auc_low),
            "auc_high": float(auc_high),

            "best_auc": float(
                max(
                    auc_low,
                    auc_high,
                )
            ),
        })

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.sort_values(
            "best_auc",
            ascending=False,
        )

    return result


# ============================================================
# 5-FOLD STABILITY
# ============================================================

def five_fold_stability(
    df: pd.DataFrame,
    feature: str,
    positive_class: str,
    negative_class: str,
    folds: int = 5,
) -> Dict[str, Any]:

    pair = df[
        df["class"].isin(
            [
                positive_class,
                negative_class,
            ]
        )
    ].copy()

    values = pd.to_numeric(
        pair[feature],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = np.isfinite(values)

    pair = pair.loc[
        valid
    ].reset_index(drop=True)

    values = values[valid]

    y = (
        pair["class"].to_numpy()
        == positive_class
    ).astype(int)

    if len(pair) < folds * 2:
        return {}

    if len(np.unique(y)) != 2:
        return {}

    skf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    fold_results = {
        "high": [],
        "low": [],
    }

    for train_idx, val_idx in skf.split(
        values,
        y,
    ):

        train_values = values[train_idx]
        val_values = values[val_idx]

        train_y = y[train_idx]
        val_y = y[val_idx]

        thresholds = make_thresholds(
            train_values
        )

        for direction in ["high", "low"]:

            best = None

            for threshold in thresholds:

                pred = predict_threshold(
                    val_values,
                    threshold,
                    direction,
                )

                bal = balanced_accuracy_score(
                    val_y,
                    pred,
                )

                f1 = f1_score(
                    val_y,
                    pred,
                    zero_division=0,
                )

                score = (
                    0.65 * bal
                    + 0.35 * f1
                )

                result = {
                    "threshold": float(
                        threshold
                    ),
                    "balanced_accuracy": float(
                        bal
                    ),
                    "f1": float(f1),
                    "score": float(score),
                }

                if (
                    best is None
                    or result["score"]
                    > best["score"]
                ):
                    best = result

            if best is not None:
                fold_results[
                    direction
                ].append(best)

    final_records = []

    for direction in [
        "high",
        "low",
    ]:

        rows = fold_results[direction]

        if not rows:
            continue

        bal = np.array(
            [
                x["balanced_accuracy"]
                for x in rows
            ]
        )

        f1 = np.array(
            [
                x["f1"]
                for x in rows
            ]
        )

        thresholds = np.array(
            [
                x["threshold"]
                for x in rows
            ]
        )

        scores = np.array(
            [
                x["score"]
                for x in rows
            ]
        )

        final_records.append({
            "direction": direction,

            "mean_balanced_accuracy":
                float(np.mean(bal)),

            "std_balanced_accuracy":
                float(np.std(bal)),

            "mean_f1":
                float(np.mean(f1)),

            "std_f1":
                float(np.std(f1)),

            "mean_score":
                float(np.mean(scores)),

            "std_score":
                float(np.std(scores)),

            "mean_threshold":
                float(np.mean(thresholds)),

            "std_threshold":
                float(np.std(thresholds)),

            "threshold_min":
                float(np.min(thresholds)),

            "threshold_max":
                float(np.max(thresholds)),

            "folds":
                len(rows),
        })

    if not final_records:
        return {}

    final_records.sort(
        key=lambda x: x[
            "mean_score"
        ],
        reverse=True,
    )

    return final_records[0]


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def build_evidence_candidates(
    df: pd.DataFrame,
    features: Sequence[str],
    pair: Tuple[str, str],
    folds: int,
) -> pd.DataFrame:

    class_a, class_b = pair

    pair_df = pairwise_feature_analysis(
        df,
        class_a,
        class_b,
        features,
    )

    records = []

    print()
    print("=" * 80)
    print(
        f"PAIRWISE ANALYSIS: "
        f"{class_a} <-> {class_b}"
    )
    print("=" * 80)

    for _, row in pair_df.iterrows():

        feature = row["feature"]

        for positive_class in [
            class_a,
            class_b,
        ]:

            negative_class = (
                class_b
                if positive_class == class_a
                else class_a
            )

            stability = five_fold_stability(
                df,
                feature,
                positive_class,
                negative_class,
                folds,
            )

            if not stability:
                continue

            # absolute effect size
            effect = abs(
                float(
                    row[
                        "effect_size_a_minus_b"
                    ]
                )
            )

            auc = float(
                row["best_auc"]
            )

            mean_bal = float(
                stability[
                    "mean_balanced_accuracy"
                ]
            )

            std_bal = float(
                stability[
                    "std_balanced_accuracy"
                ]
            )

            mean_f1 = float(
                stability["mean_f1"]
            )

            threshold = float(
                stability[
                    "mean_threshold"
                ]
            )

            # ------------------------------------------------
            # Stability score
            # ------------------------------------------------

            stability_score = max(
                0.0,
                1.0 - std_bal,
            )

            # ------------------------------------------------
            # Overall evidence score
            # ------------------------------------------------

            evidence_score = (
                0.35 * mean_bal
                + 0.20 * mean_f1
                + 0.20 * auc
                + 0.15 * min(
                    effect / 2.0,
                    1.0,
                )
                + 0.10 * stability_score
            )

            records.append({

                "pair":
                    f"{class_a} <-> {class_b}",

                "positive_class":
                    positive_class,

                "negative_class":
                    negative_class,

                "feature":
                    feature,

                "direction":
                    stability["direction"],

                "threshold":
                    threshold,

                "threshold_std":
                    float(
                        stability[
                            "std_threshold"
                        ]
                    ),

                "threshold_min":
                    float(
                        stability[
                            "threshold_min"
                        ]
                    ),

                "threshold_max":
                    float(
                        stability[
                            "threshold_max"
                        ]
                    ),

                "roc_auc":
                    auc,

                "effect_size":
                    effect,

                "mean_balanced_accuracy":
                    mean_bal,

                "std_balanced_accuracy":
                    std_bal,

                "mean_f1":
                    mean_f1,

                "std_f1":
                    float(
                        stability["std_f1"]
                    ),

                "mean_score":
                    float(
                        stability["mean_score"]
                    ),

                "stability_score":
                    stability_score,

                "evidence_score":
                    evidence_score,

                "n_total":
                    int(
                        row["n_a"]
                        + row["n_b"]
                    ),
            })

    result = pd.DataFrame(records)

    if result.empty:
        return result

    result = result.sort_values(
        [
            "evidence_score",
            "mean_balanced_accuracy",
            "roc_auc",
        ],
        ascending=False,
    ).reset_index(drop=True)

    result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result


# ============================================================
# FEATURE STATISTICS
# ============================================================

def save_feature_statistics(
    df: pd.DataFrame,
    features: Sequence[str],
    output_dir: Path,
) -> None:

    records = []

    for feature in features:

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        records.append({
            "feature": feature,
            "n": len(values),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "q25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "q75": float(values.quantile(0.75)),
            "max": float(values.max()),
        })

    pd.DataFrame(records).to_csv(
        output_dir
        / "feature_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# PRINT TOP CANDIDATES
# ============================================================

def print_top_candidates(
    candidates: pd.DataFrame,
    n: int = 20,
) -> None:

    if candidates.empty:
        print("\nไม่พบ evidence candidate")
        return

    print()
    print("=" * 100)
    print("TOP EVIDENCE CANDIDATES")
    print("=" * 100)

    cols = [
        "rank",
        "positive_class",
        "feature",
        "direction",
        "threshold",
        "roc_auc",
        "effect_size",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "mean_f1",
        "evidence_score",
    ]

    display = candidates[
        [
            c
            for c in cols
            if c in candidates.columns
        ]
    ].head(n)

    print(
        display.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze verifier features v2 "
            "with 5-fold pairwise stability."
        )
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--classifier",
        default=DEFAULT_CLASSIFIER,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
    )

    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help=(
            "วิเคราะห์ทุกคู่ class "
            "แทนเฉพาะ OOF <-> Tracking"
        ),
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
    )

    args = parser.parse_args()

    dataset_root = Path(
        args.dataset
    ).resolve()

    classifier_path = Path(
        args.classifier
    ).resolve()

    output_dir = Path(
        args.output
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("VERIFIER FEATURE ANALYSIS V2")
    print("=" * 80)

    print(
        f"Dataset    : {dataset_root}"
    )

    print(
        f"Classifier : {classifier_path}"
    )

    print(
        f"Output     : {output_dir}"
    )

    print(
        f"Folds      : {args.folds}"
    )

    # --------------------------------------------------------
    # Import classifier
    # --------------------------------------------------------

    classifier = import_module_from_file(
        classifier_path
    )

    # --------------------------------------------------------
    # Discover splits
    # --------------------------------------------------------

    split_dirs = discover_split_dirs(
        dataset_root
    )

    if not split_dirs:

        raise SystemExit(
            "\nไม่พบ dataset\n"
            f"{dataset_root}"
        )

    print("\nSplits:")

    for name, path in split_dirs.items():
        print(
            f"  {name}: {path}"
        )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    split_frames = {}

    for split, split_dir in split_dirs.items():

        df = extract_dataset(
            classifier,
            split,
            split_dir,
        )

        if not df.empty:
            split_frames[split] = df

            df.to_csv(
                output_dir
                / f"features_{split}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    if not split_frames:
        raise SystemExit(
            "ไม่สามารถ extract features ได้"
        )

    # --------------------------------------------------------
    # Discovery dataset
    # --------------------------------------------------------

    discovery_frames = []

    for split in [
        "train",
        "val",
    ]:

        if split in split_frames:
            discovery_frames.append(
                split_frames[split]
            )

    if discovery_frames:

        discovery_df = pd.concat(
            discovery_frames,
            ignore_index=True,
        )

    elif "all" in split_frames:

        discovery_df = split_frames[
            "all"
        ].copy()

    else:

        discovery_df = next(
            iter(
                split_frames.values()
            )
        ).copy()

    print()
    print(
        f"Discovery samples: "
        f"{len(discovery_df)}"
    )

    # --------------------------------------------------------
    # Feature list
    # --------------------------------------------------------

    features = numeric_features(
        discovery_df
    )

    print(
        f"Numeric features: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    save_feature_statistics(
        discovery_df,
        features,
        output_dir,
    )

    # --------------------------------------------------------
    # Pair selection
    # --------------------------------------------------------

    if args.all_pairs:

        available_classes = [
            c
            for c in CLASS_NAMES
            if c in set(
                discovery_df["class"]
            )
        ]

        pairs = []

        for i in range(
            len(available_classes)
        ):

            for j in range(
                i + 1,
                len(available_classes),
            ):

                pairs.append(
                    (
                        available_classes[i],
                        available_classes[j],
                    )
                )

    else:

        pairs = [
            pair
            for pair in PAIR_NAMES
            if (
                pair[0]
                in set(discovery_df["class"])
                and
                pair[1]
                in set(discovery_df["class"])
            )
        ]

    # --------------------------------------------------------
    # Evidence discovery
    # --------------------------------------------------------

    all_candidates = []

    for pair in pairs:

        candidates = build_evidence_candidates(
            discovery_df,
            features,
            pair,
            args.folds,
        )

        if not candidates.empty:
            all_candidates.append(
                candidates
            )

    if all_candidates:

        evidence_candidates = pd.concat(
            all_candidates,
            ignore_index=True,
        )

        evidence_candidates = (
            evidence_candidates
            .sort_values(
                [
                    "evidence_score",
                    "mean_balanced_accuracy",
                    "roc_auc",
                ],
                ascending=False,
            )
            .reset_index(drop=True)
        )

        evidence_candidates[
            "rank"
        ] = np.arange(
            1,
            len(evidence_candidates) + 1,
        )

    else:

        evidence_candidates = (
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # Save candidates
    # --------------------------------------------------------

    candidate_path = (
        output_dir
        / "verifier_evidence_candidates.csv"
    )

    evidence_candidates.to_csv(
        candidate_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"[SAVED] {candidate_path}"
    )

    print_top_candidates(
        evidence_candidates
    )

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    summary = {
        "dataset": str(
            dataset_root
        ),
        "classifier": str(
            classifier_path
        ),
        "folds": args.folds,
        "discovery_samples": int(
            len(discovery_df)
        ),
        "numeric_feature_count": len(
            features
        ),
        "pairs": [
            list(x)
            for x in pairs
        ],
        "candidate_count": int(
            len(evidence_candidates)
        ),
    }

    import json

    with open(
        output_dir / "analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Test set note
    # --------------------------------------------------------

    if "test" in split_frames:

        test_count = len(
            split_frames["test"]
        )

        print()
        print(
            f"Test samples available: "
            f"{test_count}"
        )

        print(
            "หมายเหตุ: test set "
            "ไม่ได้ใช้หา threshold/evidence"
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()