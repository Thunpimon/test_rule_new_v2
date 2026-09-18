"""
astro_pipeline_v2.py
====================
Enhanced Astronomy Image Quality Pipeline (V2 - Star Selection & MaxIm DL Logic)

Key Improvements over V1:
  1. Star Selection Filtering:
     - Margin exclusion (ignores edge stars prone to lens coma / astigmatism)
     - Hot pixel & noise exclusion (area >= 5 px)
     - Massive saturated blob exclusion
  2. Physical Signature Decoupling (MaxIm DL Principle):
     - Removes global intensity / background from star shape rules.
     - Out of Focus requires: High FWHM/HFR AND High Roundness (Low Eccentricity).
     - Tracking Error requires: High Eccentricity AND High Angle Consistency (elongated in the same direction).
     - Good requires: Low FWHM, Low Eccentricity, No Saturation/Streaks.
     - Over Saturated: High saturated pixels, flat-top blooming, spikes.
     - No Star: Star count near zero, low valid profiles.
     - Satellite: Long linear streaks across image.

Usage:
    # Single image:
    python astro_pipeline_v2.py path/to/image.png

    # Batch test on folder:
    python astro_pipeline_v2.py --batch Dataset_For_Rule_Base/01_Good --output good_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Reuse base utilities from existing code without modifying it
from astro_rule_classifier_new import (
    CLASS_NAMES,
    clamp,
    contour_hollowness,
    detect_bright_streaks,
    detect_lines,
    detect_star_contours,
    elongated_axis_angle,
    measure_star_profile,
    predict_onnx,
    read_gray_image,
    robust_mad,
    score_high,
    score_low,
    weighted_mean,
)

DEFAULT_ONNX_PATH = Path(__file__).parent / "eff_b0_kfold_add_focal_r2.onnx"

# ============================================================================
# CAMERA & TELESCOPE OPTICAL SPECIFICATIONS
# ============================================================================
# Telescope: PlaneWave CDK700 (Focal Length = 4,540 mm)
# Camera: Andor iKon-M 934 (Pixel Size = 13.0 um, 1024 x 1024)
TELESCOPE_FOCAL_LENGTH_MM: float = 4540.0
CAMERA_PIXEL_SIZE_UM: float = 13.0
DEFAULT_BINNING: float = 1.0

# Pixel Scale Formula: (Pixel Size mm / Focal Length mm) * 206265 * Binning
# (0.013 / 4540) * 206265 ≈ 0.589975 arcsec/pixel
PIXEL_SCALE_ARCSEC: float = (
    (CAMERA_PIXEL_SIZE_UM / 1000.0) / TELESCOPE_FOCAL_LENGTH_MM
) * 206264.806 * DEFAULT_BINNING

# Physical limits of stars on the sky:
# Stars are point sources smeared by atmospheric seeing (1.5 - 5.0 arcsec; severe blur <= 15-20 arcsec).
# Any single object spanning > 20 arcsec (> 34 px diameter, area > 900 px) or broad diffuse emission
# is an extended astronomical object (Galaxy / Nebula), NOT an individual star.
MAX_STAR_DIAMETER_ARCSEC: float = 20.0
MAX_STAR_AREA_CAP_PHYSICS: float = math.pi * ((MAX_STAR_DIAMETER_ARCSEC / PIXEL_SCALE_ARCSEC) / 2.0) ** 2

# Configurable per-class pass/fail thresholds
DEFAULT_THRESHOLDS_V2: Dict[str, float] = {
    "01_Good": 0.50,
    "02_Out_of_Focus": 0.50,
    "03_Tracking_Error": 0.50,
    "04_Over_Saturated": 0.50,
    "05_No_Star": 0.50,
    "06_Satellite": 0.50,
}


@dataclass
class AstroFeaturesV2:
    width: int
    height: int
    mean_intensity: float
    std_intensity: float
    background_median: float
    background_mad: float
    sharpness: float
    saturated_ratio: float
    bright_area_ratio: float

    # Filtered star metrics (central clean stars only)
    total_raw_contours: int
    star_count: int
    mean_star_area: float
    mean_circularity: float
    median_eccentricity: float
    mean_aspect_ratio: float
    max_aspect_ratio: float
    mean_hollowness: float
    max_hollowness: float
    valid_profile_count: int
    median_fwhm: float
    mean_fwhm: float
    median_hfr: float

    # Elongation & Angle Consistency
    elongated_star_count: int
    elongated_star_ratio: float
    elongated_angle_consistency: float
    elongated_angle_std: float

    # Streaks & Lines
    long_line_count: int
    longest_line_ratio: float
    streak_count: int
    max_streak_length_ratio: float
    max_streak_aspect_ratio: float
    faint_streak_evidence: float = 0.0
    max_val: float = 255.0
    max_projection_diff: float = 0.0
    median_fwhm_arcsec: float = 0.0
    mean_fwhm_arcsec: float = 0.0
    median_hfr_arcsec: float = 0.0


def compute_axial_consistency(angles: List[float]) -> Tuple[float, float]:
    """Calculate axial angular consistency (0.0 to 1.0) and standard deviation in degrees."""
    if len(angles) < 2:
        return 0.0, 90.0
    doubled = np.deg2rad(np.asarray(angles, dtype=np.float64) * 2.0)
    mean_cos = float(np.mean(np.cos(doubled)))
    mean_sin = float(np.mean(np.sin(doubled)))
    resultant = clamp(math.hypot(mean_cos, mean_sin))
    if resultant <= 1e-6:
        return 0.0, 90.0
    std_rad = math.sqrt(max(0.0, -2.0 * math.log(resultant))) / 2.0
    return float(resultant), float(np.rad2deg(std_rad))

def detect_satellite_streaks_v2(gray: np.ndarray) -> Tuple[int, float, float, float]:
    """
    Fast, robust Satellite Trail detection without star subtraction:
      - Relative adaptive threshold above local background: bg + max(2.5 * mad, 12.0), capped at 250.0
      - Small morphological CLOSE (3x3) to bridge minor pixel gaps along the trail
      - Detects long, slender linear contours (length >= 120 px, aspect >= 4.5)
      - Returns: (streak_count, max_streak_length_ratio, max_streak_aspect_ratio, max_streak_length_px)
    """
    h, w = gray.shape
    diag = math.hypot(w, h)
    gray_f = gray.astype(np.float64)
    bg = float(np.median(gray_f))
    mad = robust_mad(gray_f)

    thresh = min(bg + max(2.5 * mad, 12.0), 250.0)
    _, mask = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    streak_count = 0
    max_len = 0.0
    max_aspect = 0.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 30.0:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        l = max(float(rw), float(rh))
        s = max(min(float(rw), float(rh)), 1.0)
        aspect = l / s

        # A genuine satellite streak is physically long (>= 120 px or >= 0.12 of frame) and slender (aspect >= 4.5)
        if l >= 120.0 and aspect >= 4.5:
            streak_count += 1
            if l > max_len:
                max_len = l
                max_aspect = aspect

    return streak_count, float(max_len / diag), float(max_aspect), float(max_len)


def detect_faint_streak_evidence_v2(gray: np.ndarray) -> float:
    """Detect parallel drift trails in sparse or noisy starless frames (e.g. 2606061WXB_1_2)."""
    h, w = gray.shape
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    image_max = float(np.max(gray))
    threshold = max(float(np.percentile(gray, 95.0)), mean + 2.5 * std, 35.0)
    threshold = min(threshold, max(image_max - 1.0, 0.0))
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15.0:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        long_side = max(rw, rh)
        short_side = max(min(rw, rh), 1.0)
        aspect = long_side / short_side
        ang = rect[2]
        if rw < rh:
            ang = (ang + 90) % 180
        if aspect >= 2.2 and long_side >= 15.0:
            angles.append(ang)
    if len(angles) >= 4:
        cons, _ = compute_axial_consistency(angles)
        if cons >= 0.60:
            return 0.65
        elif cons >= 0.40:
            return 0.52
    return 0.0


def detect_star_contours_v2(gray: np.ndarray) -> Tuple[List[np.ndarray], float, float]:
    """
    Multi-Scale Star Detection that preserves both compact in-focus stars
    and large swollen / donut out-of-focus stars (40-80 px) by using a 151px background estimate.
    """
    bg = cv2.GaussianBlur(gray, (151, 151), 0)
    diff = cv2.subtract(gray, bg)
    mean_val, std_val = cv2.meanStdDev(diff)
    thresh_val = max(float(mean_val[0][0] + 1.8 * std_val[0][0]), 25.0)
    _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_pixels = float(gray.shape[0] * gray.shape[1])
    bright_pixels = float(np.count_nonzero(opened))
    bright_area_ratio = bright_pixels / total_pixels
    saturated_ratio = float(np.count_nonzero(gray >= 250)) / total_pixels

    return contours, bright_area_ratio, saturated_ratio


def extract_features_v2(
    gray: np.ndarray,
    border_margin_pct: float = 0.05,
    min_star_area: float = 5.0,
    max_star_area_cap: Optional[float] = None,
) -> AstroFeaturesV2:
    """
    Extract robust astronomical features with Star Selection Filtering:
      - Multi-Scale Star Detection (retains both sharp point sources and swollen/donut stars)
      - Physical Scale Gating: Exclude non-star extended bodies (Galaxies / Nebulae)
      - Ignores border region (where lens optical aberrations like Coma dominate)
      - Filters out hot pixels / cosmic rays (< 5 px)
    """
    h, w = gray.shape
    total_pixels = float(h * w)
    gray_f = gray.astype(np.float64)

    if max_star_area_cap is None:
        max_star_area_cap = MAX_STAR_AREA_CAP_PHYSICS

    margin_x = int(w * border_margin_pct)
    margin_y = int(h * border_margin_pct)

    # 1. Detect large extended emissions (e.g. Galaxy or diffuse Nebula)
    # If the diffuse body occupies a notable portion of the field, we exclude knots/gas inside it
    # so we measure true celestial reference stars in the field.
    bg_large = cv2.GaussianBlur(gray, (101, 101), 0)
    bg_med = float(np.median(bg_large))
    raw_extended_mask = (bg_large > (bg_med + 25.0)).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_extended_mask)
    has_large_extended_object = False
    extended_emission_mask = raw_extended_mask > 0
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = int(np.argmax(areas) + 1)
        max_component_ratio = float(stats[max_idx, cv2.CC_STAT_AREA]) / total_pixels
        # A true galaxy/diffuse nebula has a single large contiguous core (>= 6.0% of frame)
        if max_component_ratio >= 0.025:
            has_large_extended_object = True

    contours, bright_area_ratio, saturated_ratio = detect_star_contours_v2(gray)

    star_count = 0
    elongated_count = 0
    areas: List[float] = []
    circularities: List[float] = []
    aspect_ratios: List[float] = []
    elongated_angles: List[float] = []
    hollownesses: List[float] = []
    fwhms: List[float] = []
    hfrs: List[float] = []
    eccentricities: List[float] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_star_area or area > max_star_area_cap:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        # Star Selection: Exclude stars located in the border margin
        if (
            x < margin_x
            or (x + bw) > (w - margin_x)
            or y < margin_y
            or (y + bh) > (h - margin_y)
        ):
            continue

        cx, cy = int(x + bw / 2), int(y + bh / 2)
        # Exclude emission knots inside major extended galaxy/nebula bodies
        if has_large_extended_object and extended_emission_mask[cy, cx]:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        # Rotation-invariant aspect ratio & angle using fitEllipse
        aspect = 1.0
        angle = 0.0
        if len(contour) >= 5:
            try:
                box = cv2.fitEllipse(contour)
                d1, d2 = box[1][0], box[1][1]
                aspect = max(d1, d2) / max(min(d1, d2), 1e-3)
                angle = box[2]  # clockwise from vertical [0, 180)
            except Exception:
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                angle = 0.0 if bh > bw else 90.0
        else:
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            angle = 0.0 if bh > bw else 90.0

        star_count += 1
        areas.append(area)
        aspect_ratios.append(float(aspect))
        circularities.append(float(circularity))

        # Hollowness (central dip) is physically valid for optical secondary-mirror donuts (area >= 15 px)
        if area >= 15.0:
            hollownesses.append(contour_hollowness(gray, contour))
        else:
            hollownesses.append(0.0)

        profile = measure_star_profile(gray, contour)
        if profile is not None:
            fwhms.append(profile["fwhm"])
            hfrs.append(profile["hfr"])
            eccentricities.append(profile["eccentricity"])

        # Check for elongated star:
        # Require area >= 10.0 to filter out single-pixel noise clusters, aspect >= 1.30
        if area >= 10.0 and aspect >= 1.30:
            elongated_count += 1
            elongated_angles.append(angle)

    # Line & Streak detection
    long_line_count, longest_line_ratio, _, _, _ = detect_lines(gray)
    streak_count, max_streak_length_ratio, max_streak_aspect_ratio, _ = detect_satellite_streaks_v2(gray)

    # Angle consistency
    consistency, angle_std = compute_axial_consistency(elongated_angles)
    max_hollowness = float(np.max(hollownesses)) if hollownesses else 0.0

    # Faint streak evidence (only computed on sparse / low-star frames to save computation)
    faint_streak_evidence = detect_faint_streak_evidence_v2(gray) if star_count < 5 else 0.0

    # Projection difference for horizontal/vertical CCD blooming bars
    max_val = float(np.max(gray))
    rm = np.mean(gray_f, axis=1)
    cm = np.mean(gray_f, axis=0)
    r_diff = float(np.max(rm) - np.median(rm))
    c_diff = float(np.max(cm) - np.median(cm))
    max_projection_diff = max(r_diff, c_diff)

    # Resolution Normalization: Scale physical pixel metrics relative to standard 1024x1024 frame
    # (e.g., in a 512x512 image, 1 pixel represents 2x angular scale, so scale=2.0)
    scale = 1024.0 / max(float(w), float(h))
    scale_sq = scale * scale

    raw_mean_star_area = float(np.mean(areas)) if areas else 0.0
    raw_median_fwhm = float(np.median(fwhms)) if fwhms else 0.0
    raw_mean_fwhm = float(np.mean(fwhms)) if fwhms else 0.0
    raw_median_hfr = float(np.median(hfrs)) if hfrs else 0.0
    raw_sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return AstroFeaturesV2(
        width=w,
        height=h,
        mean_intensity=float(np.mean(gray_f)),
        std_intensity=float(np.std(gray_f)),
        background_median=float(np.median(gray_f)),
        background_mad=robust_mad(gray_f),
        sharpness=raw_sharpness * scale_sq,
        saturated_ratio=saturated_ratio,
        bright_area_ratio=bright_area_ratio,
        total_raw_contours=len(contours),
        star_count=star_count,
        mean_star_area=raw_mean_star_area * scale_sq,
        mean_circularity=float(np.mean(circularities)) if circularities else 0.0,
        median_eccentricity=float(np.median(eccentricities)) if eccentricities else 0.0,
        mean_aspect_ratio=float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
        max_aspect_ratio=float(np.max(aspect_ratios)) if aspect_ratios else 0.0,
        mean_hollowness=float(np.mean(hollownesses)) if hollownesses else 0.0,
        max_hollowness=max_hollowness,
        valid_profile_count=len(fwhms),
        median_fwhm=raw_median_fwhm * scale,
        mean_fwhm=raw_mean_fwhm * scale,
        median_hfr=raw_median_hfr * scale,
        elongated_star_count=elongated_count,
        elongated_star_ratio=float(elongated_count / star_count) if star_count else 0.0,
        elongated_angle_consistency=consistency,
        elongated_angle_std=angle_std,
        long_line_count=long_line_count,
        longest_line_ratio=longest_line_ratio,
        streak_count=streak_count,
        max_streak_length_ratio=max_streak_length_ratio,
        max_streak_aspect_ratio=max_streak_aspect_ratio,
        faint_streak_evidence=faint_streak_evidence,
        max_val=max_val,
        max_projection_diff=max_projection_diff,
        median_fwhm_arcsec=float(raw_median_fwhm * scale * PIXEL_SCALE_ARCSEC),
        mean_fwhm_arcsec=float(raw_mean_fwhm * scale * PIXEL_SCALE_ARCSEC),
        median_hfr_arcsec=float(raw_median_hfr * scale * PIXEL_SCALE_ARCSEC),
    )


# ============================================================================
# PHYSICS-BASED RULE SCORING FUNCTIONS (V2 - MaxIm DL Logic)
# ============================================================================

def score_good_v2(f: AstroFeaturesV2) -> float:
    """
    01_Good:
      - Small to moderate FWHM (accommodates atmospheric seeing 16-28 px)
      - Natural star roundness: Low eccentricity (< 0.55) & High circularity (> 0.60)
      - Sufficient stars present (>= 2)
      - Clean: Low saturation, no hollowness, no long satellite streaks
      - Gated: Penalize if stars are clearly elongated (TE) or swollen/hollow (OOF)
    """
    # A genuine Good frame requires sufficient stars (>= 3). Isolated 1-2 points are hot pixels/cosmic rays.
    if f.star_count < 2:
        return 0.05

    # Gate: Reject swollen OOF stars, hollow donut stars, and blurry defocused frames immediately from Good
    if (f.median_fwhm > 28.0 and f.sharpness < 15000.0) or f.mean_star_area > 115.0 or f.mean_hollowness > 0.05:
        return 0.10

    base = weighted_mean([
        (score_low(f.median_fwhm, 17.0, 32.0), 1.8),
        (score_low(f.median_eccentricity, 0.38, 0.58), 1.6),
        (score_high(f.mean_circularity, 0.60, 0.85), 1.2),
        (score_high(f.sharpness, 6000.0, 35000.0), 1.5),
        (score_high(f.star_count, 3, 25), 0.8),
        (score_low(f.saturated_ratio, 0.010, 0.040), 1.0),
        (score_low(f.mean_hollowness, 0.008, 0.04), 1.2),
    ])

    if f.star_count == 2:
        base *= 0.35

    # Tracking gate: If stars are elongated AND highly angle-consistent -> strong penalty
    if f.elongated_star_count >= 5 and f.elongated_star_ratio > 0.35 and f.elongated_angle_consistency > 0.45:
        te_penalty = 0.25
    elif f.elongated_star_count >= 20 and f.elongated_star_ratio >= 0.30 and f.elongated_angle_consistency >= 0.30 and f.mean_aspect_ratio >= 1.32:
        # Dense star field with >= 20 elongated stars in parallel (e.g. 260512N494_1_1)
        te_penalty = 0.25
    elif f.elongated_star_count >= 5 and f.mean_aspect_ratio > 1.48:
        te_penalty = 0.30
    else:
        te_penalty = 1.0

    # Satellite gate: Long linear streaks across the frame are not Good
    sat_penalty = score_low(f.max_streak_length_ratio, 0.20, 0.55)
    if f.max_streak_length_ratio >= 0.18 and f.max_streak_aspect_ratio >= 5.0:
        sat_penalty = min(sat_penalty, 0.15)
    elif f.max_streak_length_ratio >= 0.12 and f.max_streak_aspect_ratio >= 10.0:
        sat_penalty = min(sat_penalty, 0.15)

    # Over-saturation gate: Blooming columns or normalized saturated frames are not Good
    if f.max_projection_diff >= 50.0 or (f.max_val <= 165.0 and f.background_median <= 25.0 and f.star_count >= 10):
        base *= 0.15

    return clamp(base * te_penalty * sat_penalty)


def score_out_of_focus_v2(f: AstroFeaturesV2) -> float:
    """
    02_Out_of_Focus (Two-Track Architecture with Evidence Gating):
      - Track 1: Donut Signature (Field-wide hollowness from secondary mirror obstruction)
      - Track 2: Swollen Star Signature (Defocus blur: large FWHM/Area, round, not saturated)
    """
    if f.star_count < 2:
        return 0.02

    # Track 1: Donut Signature (requires genuine star sample, not 1-2 tiny 6px noise specks)
    donut_evidence = 0.0
    if f.star_count >= 4 and f.mean_star_area >= 20.0:
        donut_evidence = score_high(f.mean_hollowness, 0.04, 0.12)

    donut_score = weighted_mean([
        (donut_evidence, 2.5),
        (score_high(f.median_fwhm, 18.0, 35.0), 1.2),
        (score_low(f.median_eccentricity, 0.35, 0.60), 1.0),
    ]) if donut_evidence > 0.0 else 0.0

    # Track 2: Swollen Star Signature (Defocus blur must not be long elongated trails)
    swollen_evidence = max(
        score_high(f.mean_star_area, 80.0, 200.0),
        score_high(f.median_fwhm, 26.0, 38.0)
    ) if f.mean_aspect_ratio < 1.50 else 0.0

    swollen_score = weighted_mean([
        (swollen_evidence, 2.5),
        (score_high(f.median_fwhm, 26.0, 42.0), 2.0),
        (score_low(f.sharpness, 3000.0, 15000.0), 1.8),
        (score_high(f.median_hfr, 12.0, 22.0), 1.5),
        (score_low(f.median_eccentricity, 0.40, 0.75), 1.2),
        (score_low(f.saturated_ratio, 0.005, 0.030), 1.2),
    ]) if swollen_evidence > 0.0 else 0.0

    # Anti-Tracking gate: Only penalize swollen score; DO NOT penalize true optical donuts!
    if f.mean_aspect_ratio > 1.40 and f.elongated_angle_consistency > 0.65:
        swollen_score *= 0.20
        if donut_evidence < 0.50:
            donut_score *= 0.20

    final_score = max(donut_score, swollen_score)
    return clamp(final_score)


def score_tracking_error_v2(f: AstroFeaturesV2) -> float:
    """
    03_Tracking_Error (Pure Geometry & Vector Consensus):
      - Rotation-invariant high aspect ratio & high eccentricity
      - Parallel drift consensus across multiple stars
      - Exempts long trails from high consistency requirement
      - Rejects true OOF (swollen discs and central obstruction donuts)
      - Detects parallel trails in faint / starless frames
    """
    if f.star_count < 3 or f.elongated_star_count < 2:
        if f.faint_streak_evidence >= 0.40:
            return clamp(f.faint_streak_evidence)
        return 0.05

    # Anti-OOF guard: Blurry non-parallel stars (FWHM > 28, Sharpness < 15000, Aspect < 1.40, Consistency < 0.65) are OOF, not TE!
    if f.median_fwhm > 28.0 and f.sharpness < 15000.0 and f.mean_aspect_ratio < 1.40 and f.elongated_angle_consistency < 0.65:
        return 0.10

    # Donut / OOF rejection:
    # A true OOF image has swollen round stars or hollow round donuts.
    # In TE, stars with large area (>115) only occur if they are trails (aspect >= 1.45).
    if f.mean_aspect_ratio < 1.45:
        if f.mean_hollowness > 0.04:
            return 0.05
        if f.mean_star_area > 115.0:
            return 0.08

    # Hollow donut gate: A star with central obstruction (hollowness > 0.08 or area > 200)
    # can only be Tracking Error if stars form extreme parallel trails (consistency >= 0.70).
    # Area check only guards if stars are NOT elongated trails (aspect < 1.60),
    # preventing images like 260511AAJR_1_1 (aspect 3.77) from being falsely blocked.
    if (f.mean_hollowness > 0.08 or (f.mean_star_area > 200.0 and f.mean_aspect_ratio < 1.60)) and f.elongated_angle_consistency < 0.70:
        return 0.05

    # Core shape evidence
    shape_score = weighted_mean([
        (score_high(f.median_eccentricity, 0.42, 0.75), 1.8),
        (score_high(f.mean_aspect_ratio, 1.22, 1.55), 2.2),
        (score_high(f.elongated_star_ratio, 0.22, 0.55), 2.0),
        (score_high(f.max_aspect_ratio, 1.60, 4.0), 1.0),
    ])

    # Angle consistency evidence
    cons_score = score_high(f.elongated_angle_consistency, 0.22, 0.58)

    # Dense star field consensus booster
    if f.elongated_star_count >= 20 and f.elongated_star_ratio >= 0.30 and f.elongated_angle_consistency >= 0.30 and f.mean_aspect_ratio >= 1.32:
        cons_score = max(cons_score, 0.65)
        shape_score = max(shape_score, 0.60)

    # Long trail exemption: If trails are long, consistency requirement is relaxed
    if f.mean_aspect_ratio > 1.45 and f.elongated_star_ratio > 0.30:
        cons_score = max(cons_score, 0.75)
        shape_score = max(shape_score, 0.70)

    # Combined score (weighted mean, NOT destructive multiplication)
    score = weighted_mean([
        (shape_score, 2.2),
        (cons_score, 2.0),
    ])

    # Star count gate (needs at least 3 elongated stars to form consensus)
    count_gate = score_high(f.elongated_star_count, 2, 6)

    # Hard rejection if angles are completely random (< 0.20) and trails are not long (< 1.35)
    if f.elongated_angle_consistency < 0.20 and f.mean_aspect_ratio < 1.35:
        score *= 0.20

    # Satellite gate: An isolated long streak across the frame is Satellite, not Tracking Error
    is_field_drift = (f.elongated_star_ratio >= 0.40 and f.elongated_angle_consistency >= 0.75)
    if not is_field_drift and f.max_streak_aspect_ratio >= 14.0 and f.max_streak_length_ratio >= 0.14:
        score *= 0.30
    elif f.max_streak_length_ratio >= 0.28 and f.max_streak_aspect_ratio >= 14.0 and f.elongated_star_count < 10:
        score *= 0.15

    return clamp(score * count_gate)


def score_over_saturated_v2(f: AstroFeaturesV2) -> float:
    """
    04_Over_Saturated (Three-Track Architecture):
      - Track 1: Massive CCD blooming bars / column bleeds (max_projection_diff >= 50.0).
                 Guarded against parallel tracking error trails.
      - Track 2: Normalized / Stretched saturation (max_val <= 165.0, star_count >= 10, bg_med <= 25.0).
      - Track 3: High contrast dark frames with saturated cores/spikes (bg_med <= 25.0, max_val >= 250.0, saturated_ratio >= 0.003, fwhm < 25.0).
      - Anti-OOF guard: Hollow donuts or huge swollen discs are OOF, not Over_Saturated.
    """
    # Track 1: Massive Blooming Bar / Column (projection diff >= 50)
    track1 = 0.0
    if f.max_projection_diff >= 50.0:
        # Anti-Satellite guard: if a long slender satellite streak exists,
        # it is a horizontal/vertical satellite trail, NOT CCD blooming!
        if f.max_streak_length_ratio >= 0.25 and f.max_streak_aspect_ratio >= 10.0:
            track1 = 0.10
        # Anti-TE guard: if parallel trails across many stars exist, it's Tracking Error, not a blooming bar
        elif f.elongated_star_count >= 5 and f.elongated_angle_consistency >= 0.45 and f.mean_aspect_ratio >= 1.50:
            track1 = 0.10
        # Anti-No_Star / Flare guard: Blooming bars originate from saturated stars.
        # An empty frame with no stars and low bright area is a dome/sky gradient flare, NOT blooming!
        elif f.star_count <= 2 and f.bright_area_ratio < 0.001:
            track1 = 0.05
        else:
            track1 = max(score_high(f.max_projection_diff, 50.0, 75.0), 0.90)

    # Anti-OOF guard for Tracks 2 & 3: Hollow donuts, large swollen discs, or blurry non-blooming stars
    if f.mean_hollowness > 0.04 or (f.mean_star_area > 120.0 and f.median_fwhm > 30.0) or (f.median_fwhm > 28.0 and f.sharpness < 15000.0 and f.max_projection_diff < 45.0):
        return clamp(track1 if track1 > 0.0 else 0.05)

    # Track 2: Type 2 Normalized Saturation (max <= 165, star_count >= 10, bg <= 25)
    track2 = 0.0
    if f.max_val <= 165.0 and f.star_count >= 10 and f.background_median <= 25.0:
        track2 = max(score_high(f.star_count, 10, 50) * score_low(f.max_val, 130.0, 165.0), 0.85)

    # Track 3: High contrast dark frame with saturated spikes (FILE_CURRENT_FOCUS)
    track3 = 0.0
    if f.background_median <= 25.0 and f.max_val >= 250.0 and f.saturated_ratio >= 0.003 and f.median_fwhm < 25.0:
        track3 = max(score_high(f.saturated_ratio, 0.003, 0.020), 0.85)

    base = weighted_mean([
        (score_high(f.saturated_ratio, 0.015, 0.060), 2.2),
        (score_high(f.bright_area_ratio, 0.015, 0.050), 1.8),
        (score_low(f.sharpness, 500.0, 8000.0), 0.8),
        (score_high(f.max_aspect_ratio, 4.0, 10.0), 0.7),
    ])
    if f.star_count <= 2 and f.bright_area_ratio < 0.001:
        base *= 0.10
    return clamp(max(track1, track2, track3, base))


def score_no_star_v2(f: AstroFeaturesV2) -> float:
    """
    05_No_Star:
      - Star count near 0
      - Low valid profiles
      - Low bright area
      - Gate: If there is at least 1 real sharp star (FWHM < 28 px and sharpness > 6000),
        it is not empty sky!
    """
    base = weighted_mean([
        (score_low(f.star_count, 0, 4), 2.5),
        (score_low(f.valid_profile_count, 0, 3), 1.8),
        (score_low(f.bright_area_ratio, 0.001, 0.008), 1.0),
        (score_low(f.std_intensity, 5.0, 18.0), 0.8),
    ])

    # Sharp star gate: If real sharp star exists, strongly penalize No_Star
    # Require at least 3 stars with at least 2 valid profiles (isolated single/double points are hot pixels/noise)
    if f.star_count >= 3 and f.valid_profile_count >= 2 and 5.0 < f.median_fwhm < 28.0 and f.sharpness > 6000.0:
        base *= 0.35

    return clamp(base)


def score_satellite_v2(f: AstroFeaturesV2) -> float:
    """
    06_Satellite:
      - Distinct long linear streak across frame (length >= 120 px, aspect >= 4.5)
      - High confidence for clear long streaks (length_ratio >= 0.25, aspect >= 10.0)
      - Anti-blooming gate: CCD blooming column/bar is Over_Saturated, not Satellite
      - Anti-TE gate: Field-wide parallel star trails (multiple stars drifting together) is TE, not Satellite
    """
    if f.streak_count < 1 or f.max_streak_length_ratio < 0.10:
        return 0.02

    # High confidence for distinct long streaks OR slender streaks
    if (f.max_streak_length_ratio >= 0.25 and f.max_streak_aspect_ratio >= 10.0) or \
       (f.max_streak_length_ratio >= 0.12 and f.max_streak_aspect_ratio >= 12.0):
        base = max(score_high(f.max_streak_length_ratio, 0.12, 0.50), 0.88)
    else:
        base = weighted_mean([
            (score_high(f.streak_count, 0.8, 2.0), 1.5),
            (score_high(f.max_streak_length_ratio, 0.12, 0.50), 2.5),
            (score_high(f.max_streak_aspect_ratio, 4.5, 18.0), 2.0),
            (score_low(f.saturated_ratio, 0.02, 0.08), 0.6),
        ])

    # Anti-blooming gate: Only penalize if streak is NOT an unmistakable long slender trail
    if f.max_projection_diff >= 50.0 and (f.max_streak_length_ratio < 0.30 or f.max_streak_aspect_ratio < 12.0):
        base *= 0.10

    # Anti-TE gate: Only penalize if many stars (>= 8) drift in parallel with high consistency
    if f.elongated_star_count >= 8 and f.elongated_star_ratio >= 0.35 and f.elongated_angle_consistency >= 0.50:
        if f.max_streak_length_ratio < 0.30 and f.max_streak_aspect_ratio < 15.0:
            base *= 0.15

    return clamp(base)


RULE_SCORERS_V2 = {
    "01_Good": score_good_v2,
    "02_Out_of_Focus": score_out_of_focus_v2,
    "03_Tracking_Error": score_tracking_error_v2,
    "04_Over_Saturated": score_over_saturated_v2,
    "05_No_Star": score_no_star_v2,
    "06_Satellite": score_satellite_v2,
}


def score_rules_v2(features: AstroFeaturesV2) -> Dict[str, float]:
    return {cname: scorer(features) for cname, scorer in RULE_SCORERS_V2.items()}


# ============================================================================
# UNIFIED PIPELINE (V2)
# ============================================================================

def run_pipeline_v2(
    image_path: str | Path,
    onnx_path: Optional[str | Path] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Execute complete V2 pipeline with Star Selection & MaxIm DL rule scoring."""
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {img_p}")

    onnx_p = Path(onnx_path) if onnx_path else DEFAULT_ONNX_PATH
    active_th = dict(DEFAULT_THRESHOLDS_V2)
    if thresholds:
        active_th.update(thresholds)

    # 1. Image Read & Star Selection Feature Extraction
    gray = read_gray_image(img_p)
    feats = extract_features_v2(gray)

    # 2. Run V2 Rules
    raw_scores = score_rules_v2(feats)

    filter_details: Dict[str, Dict[str, Any]] = {}
    passed_filters = []
    for cname in CLASS_NAMES:
        sc = float(raw_scores.get(cname, 0.0))
        th = float(active_th.get(cname, 0.50))
        passed = bool(sc >= th)
        if passed:
            passed_filters.append(cname)
        filter_details[cname] = {
            "score": round(sc, 4),
            "score_pct": round(sc * 100, 2),
            "threshold": round(th, 4),
            "threshold_pct": round(th * 100, 2),
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        }

    top_filter_class = max(raw_scores, key=raw_scores.get)
    top_filter_score = float(raw_scores[top_filter_class])

    # 3. CNN Prediction
    has_model = onnx_p.exists()
    cnn_results: Dict[str, Any] = {}
    if has_model:
        cnn_raw = predict_onnx(img_p, onnx_p)
        cnn_pred = max(cnn_raw, key=cnn_raw.get)
        cnn_conf = float(cnn_raw[cnn_pred])
        cnn_results = {
            "available": True,
            "predicted_class": cnn_pred,
            "confidence": round(cnn_conf, 4),
            "confidence_pct": round(cnn_conf * 100, 2),
            "probabilities": {k: round(v, 4) for k, v in cnn_raw.items()},
            "probabilities_pct": {k: round(v * 100, 2) for k, v in cnn_raw.items()},
        }
    else:
        cnn_pred = None
        cnn_results = {
            "available": False,
            "predicted_class": None,
            "confidence": 0.0,
            "confidence_pct": 0.0,
            "probabilities": {},
            "probabilities_pct": {},
        }

    # 4. Agreement
    is_matched = bool(has_model and cnn_pred == top_filter_class)
    cnn_filter_passed = bool(has_model and filter_details.get(cnn_pred, {}).get("passed", False))

    if not has_model:
        agree_status = "FILTER_ONLY"
        msg_th = f"ผลจากตัวกรองหลัก: {top_filter_class} ({top_filter_score * 100:.1f}%)"
    elif is_matched and cnn_filter_passed:
        agree_status = "STRONG_MATCH"
        msg_th = f"✅ โมเดลและตัวกรองเห็นตรงกันสมบูรณ์ว่าเป็น '{cnn_pred}' และผ่านเกณฑ์ตัวกรอง"
    elif is_matched and not cnn_filter_passed:
        agree_status = "WEAK_MATCH"
        msg_th = f"⚠️ ทั้งคู่ชี้ไปที่ '{cnn_pred}' แต่คะแนนตัวกรองยังไม่ถึงเกณฑ์"
    else:
        agree_status = "MISMATCH"
        msg_th = f"⚡ ผลไม่ตรงกัน: โมเดลทาย '{cnn_pred}' แต่ตัวกรองชี้ไปที่ '{top_filter_class}'"

    return {
        "image_info": {
            "file_name": img_p.name,
            "file_path": str(img_p.resolve()),
            "width": feats.width,
            "height": feats.height,
        },
        "agreement": {
            "status": agree_status,
            "is_matched": is_matched,
            "cnn_filter_passed": cnn_filter_passed,
            "summary_th": msg_th,
        },
        "cnn": cnn_results,
        "filters": {
            "top_filter_class": top_filter_class,
            "top_filter_score": round(top_filter_score, 4),
            "top_filter_score_pct": round(top_filter_score * 100, 2),
            "passed_filters": passed_filters,
            "details": filter_details,
        },
        "key_physical_features": {
            "star_count": feats.star_count,
            "fwhm_median": round(feats.median_fwhm, 2),
            "eccentricity_median": round(feats.median_eccentricity, 3),
            "circularity_mean": round(feats.mean_circularity, 3),
            "angle_consistency": round(feats.elongated_angle_consistency, 3),
            "saturated_ratio_pct": round(feats.saturated_ratio * 100, 3),
            "streak_count": feats.streak_count,
        },
        "raw_features": asdict(feats),
    }


def resolve_true_class(name: str) -> Optional[str]:
    """
    Resolve folder or filename to a standard CLASS_NAMES item flexibly:
    - Exact match: '01_Good' -> '01_Good'
    - Case-insensitive / whitespace stripped: 'good', 'Good', '01-good'
    - Short name match: 'out_of_focus', 'oof' -> '02_Out_of_Focus'
    """
    cleaned = name.strip().lower().replace("-", "_")
    for c in CLASS_NAMES:
        c_lower = c.lower()
        c_short = c_lower.split("_", 1)[-1]  # 'good', 'out_of_focus', etc.
        if cleaned == c_lower or cleaned == c_short or cleaned.endswith(c_short) or c_short in cleaned:
            return c
    # Check abbreviations
    if cleaned in ("oof", "off"):
        return "02_Out_of_Focus"
    if cleaned in ("te", "tracking"):
        return "03_Tracking_Error"
    if cleaned in ("os", "oversat", "over_sat", "saturated"):
        return "04_Over_Saturated"
    if cleaned in ("ns", "nostar", "no_star"):
        return "05_No_Star"
    if cleaned in ("st", "sat", "satellite"):
        return "06_Satellite"
    return None


def run_batch_folder_v2(
    folder_path: str | Path,
    output_csv: str | Path,
    onnx_path: Optional[str | Path] = None,
) -> None:
    """Run batch evaluation on a folder of images (or root dataset directory) and export CSV."""
    fpath = Path(folder_path)
    if not fpath.exists():
        print(f"[Error] Directory not found: {fpath}")
        return

    # Check if this folder contains subdirectories matching CLASS_NAMES (e.g. Dataset_For_Rule_Base)
    subdirs = [d for d in fpath.iterdir() if d.is_dir() and resolve_true_class(d.name)]
    if subdirs and not any(fpath.glob("*.png")):
        print(f"\n[V2 BATCH] Detected dataset root with {len(subdirs)} class folders in '{fpath.name}'...")
        all_rows = []
        overall_total = 0
        overall_rule_correct = 0
        overall_cnn_correct = 0
        overall_agreed = 0
        overall_agreed_correct = 0

        for sdir in sorted(subdirs, key=lambda d: resolve_true_class(d.name) or ""):
            c_target = resolve_true_class(sdir.name)
            s_imgs = [p for p in sorted(sdir.glob("*.*")) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            if not s_imgs:
                continue
            c_correct = 0
            c_cnn_correct = 0
            for idx, img in enumerate(s_imgs, 1):
                res = run_pipeline_v2(img, onnx_path=onnx_path)
                cnn_pred = res["cnn"]["predicted_class"]
                filter_top = res["filters"]["top_filter_class"]
                filter_score = res["filters"]["top_filter_score_pct"]
                is_matched = res["agreement"]["is_matched"]

                if filter_top == c_target:
                    c_correct += 1
                    overall_rule_correct += 1
                if cnn_pred == c_target:
                    c_cnn_correct += 1
                    overall_cnn_correct += 1
                if is_matched:
                    overall_agreed += 1
                    if cnn_pred == c_target:
                        overall_agreed_correct += 1

                all_rows.append({
                    "image_name": img.name,
                    "folder": sdir.name,
                    "true_class": c_target,
                    "cnn_predicted_class": cnn_pred,
                    "cnn_confidence_pct": res["cnn"]["confidence_pct"],
                    "filter_top_class": filter_top,
                    "filter_top_score_pct": filter_score,
                    "is_matched": is_matched,
                    "agreement_status": res["agreement"]["status"],
                })
            overall_total += len(s_imgs)
            print(f"  {c_target:20s}: Rule={c_correct:2d}/{len(s_imgs):2d} ({c_correct/len(s_imgs)*100:5.1f}%) | CNN={c_cnn_correct:2d}/{len(s_imgs):2d} ({c_cnn_correct/len(s_imgs)*100:5.1f}%)")

        out_p = Path(output_csv)
        try:
            with open(out_p, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"\n[SAVED] CSV written to {out_p.resolve()}")
        except Exception as e:
            print(f"\n[WARN] Failed to write CSV: {e}")

        print(f"\n=== OVERALL BATCH RESULTS ===")
        print(f"Total Images:     {overall_total}")
        print(f"Rule Accuracy:    {overall_rule_correct}/{overall_total} ({overall_rule_correct/overall_total*100:.2f}%)")
        print(f"CNN Accuracy:     {overall_cnn_correct}/{overall_total} ({overall_cnn_correct/overall_total*100:.2f}%)")
        if overall_agreed > 0:
            print(f"Consensus Match:  {overall_agreed_correct}/{overall_agreed} ({overall_agreed_correct/overall_agreed*100:.2f}%) across {overall_agreed} agreed images\n")
        return

    # Single folder evaluation
    imgs = [
        p for p in sorted(fpath.glob("*.*"))
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not imgs:
        print(f"No images found in {fpath}")
        return

    true_class = resolve_true_class(fpath.name)
    if not true_class:
        print(f"[WARNING] โฟลเดอร์ '{fpath.name}' ไม่ตรงกับชื่อคลาสมาตรฐาน: {CLASS_NAMES}")
        print("          ระบบจะประมวลผลต่อ แต่จะไม่สามารถคิดคะแนน Accuracy ได้")

    print(f"\n[V2 BATCH] Evaluating {len(imgs)} images from {fpath.name} (Resolved True Class: {true_class or 'Unknown'})...")
    rows = []
    correct_count = 0
    match_count = 0

    for idx, img in enumerate(imgs, 1):
        res = run_pipeline_v2(img, onnx_path=onnx_path)
        cnn_pred = res["cnn"]["predicted_class"]
        filter_top = res["filters"]["top_filter_class"]
        filter_score = res["filters"]["top_filter_score_pct"]
        is_matched = res["agreement"]["is_matched"]

        if is_matched:
            match_count += 1
        if true_class and filter_top == true_class:
            correct_count += 1

        print(
            f"[{idx:>2}/{len(imgs)}] {img.name:<22} | "
            f"CNN: {str(cnn_pred):<18} | "
            f"Filter V2: {filter_top:<18} ({filter_score:>5.1f}%) | "
            f"{'MATCH' if is_matched else 'DIFF'}"
        )

        row = {
            "image_name": img.name,
            "true_class": true_class or fpath.name,
            "cnn_predicted_class": cnn_pred,
            "cnn_confidence_pct": res["cnn"]["confidence_pct"],
            "filter_top_class": filter_top,
            "filter_top_score_pct": filter_score,
            "is_matched": is_matched,
            "agreement_status": res["agreement"]["status"],
        }
        for cname in CLASS_NAMES:
            row[f"score_{cname}_pct"] = res["filters"]["details"][cname]["score_pct"]
            row[f"passed_{cname}"] = res["filters"]["details"][cname]["passed"]
        rows.append(row)

    # Save to CSV
    out_p = Path(output_csv)
    try:
        with open(out_p, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[SAVED] CSV written to {out_p.resolve()}")
    except PermissionError:
        alt_p = out_p.with_name(f"{out_p.stem}_alt{out_p.suffix}")
        with open(alt_p, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[SAVED] '{out_p.name}' กำลังถูกเปิดอยู่ใน Excel จึงบันทึกลง: {alt_p.resolve()} แทน")

    if true_class:
        print(f"Filter V2 Accuracy on {true_class}: {correct_count}/{len(imgs)} ({correct_count/len(imgs)*100:.1f}%)")
    else:
        print(f"Filter V2 finished on {fpath.name} (True Class Unknown)")
    print(f"Agreement Rate with CNN: {match_count}/{len(imgs)} ({match_count/len(imgs)*100:.1f}%)\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Astronomy Pipeline V2 (Star Selection & MaxIm DL)")
    parser.add_argument("image", nargs="?", help="Path to single astronomy image")
    parser.add_argument("--batch", help="Path to folder to batch evaluate")
    parser.add_argument("--output", default="batch_v2_results.csv", help="Output CSV path for batch")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX_PATH), help="Path to ONNX model file")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    if args.batch:
        run_batch_folder_v2(args.batch, args.output, onnx_path=args.onnx)
        return

    if not args.image:
        parser.print_help()
        return

    res = run_pipeline_v2(args.image, onnx_path=args.onnx)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # Print summary report
    print("\n" + "=" * 80)
    print(f" PIPELINE V2 REPORT: {res['image_info']['file_name']}")
    print("=" * 80)
    print(f" Summary: {res['agreement']['summary_th']}")
    print(f" CNN Top-1: {res['cnn']['predicted_class']} ({res['cnn']['confidence_pct']}%)")
    print(f" Filter V2 Winner: {res['filters']['top_filter_class']} ({res['filters']['top_filter_score_pct']}%)")
    print("-" * 80)
    print(f" {'Filter Class':<20} {'Score (%)':<12} {'Status':<8}")
    for cname in CLASS_NAMES:
        d = res["filters"]["details"][cname]
        mark = "★ Winner" if cname == res["filters"]["top_filter_class"] else ""
        print(f" {cname:<20} {d['score_pct']:>5.1f}%       [{d['status']}] {mark}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
