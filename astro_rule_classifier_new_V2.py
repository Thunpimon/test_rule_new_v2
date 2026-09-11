from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


CLASS_NAMES = [
    "01_Good",
    "02_Out_of_Focus",
    "03_Tracking_Error",
    "04_Over_Saturated",
    "05_No_Star",
    "06_Satellite",
]


@dataclass
class AstroFeatures:
    width: int
    height: int
    mean_intensity: float
    std_intensity: float
    background_median: float
    background_mad: float
    sharpness: float
    saturated_ratio: float
    bright_area_ratio: float
    star_count: int
    mean_star_area: float
    mean_circularity: float
    elongated_star_count: int
    elongated_star_ratio: float
    elongated_angle_consistency: float
    elongated_angle_std: float
    elongated_radial_alignment: float
    mean_aspect_ratio: float
    max_aspect_ratio: float
    mean_hollowness: float
    valid_profile_count: int
    mean_fwhm: float
    median_fwhm: float
    mean_hfr: float
    median_hfr: float
    median_eccentricity: float
    median_fwhm_ratio: float
    long_line_count: int
    longest_line_ratio: float
    axis_aligned_line_ratio: float
    diagonal_line_ratio: float
    line_angle_std: float
    streak_count: int
    max_streak_length_ratio: float
    max_streak_aspect_ratio: float
    max_streak_area_ratio: float


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(max(minimum, min(maximum, value)))


def score_high(value: float, weak_at: float, strong_at: float) -> float:
    if strong_at == weak_at:
        return 1.0 if value >= strong_at else 0.0
    return clamp((value - weak_at) / (strong_at - weak_at))


def score_low(value: float, strong_at: float, weak_at: float) -> float:
    return 1.0 - score_high(value, strong_at, weak_at)


def weighted_mean(parts: Iterable[Tuple[float, float]]) -> float:
    total_weight = 0.0
    total_score = 0.0

    for score, weight in parts:
        total_score += clamp(score) * weight
        total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0


def read_gray_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint8:
        return image

    image_f = image.astype(np.float32)
    min_val = float(np.min(image_f))
    max_val = float(np.max(image_f))

    if max_val <= min_val:
        return np.zeros(image.shape, dtype=np.uint8)

    return cv2.normalize(
        image_f,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)


def read_bgr_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype == np.uint8:
        return image

    image_f = image.astype(np.float32)
    min_val = float(np.min(image_f))
    max_val = float(np.max(image_f))

    if max_val <= min_val:
        return np.zeros(
            image.shape[:2] + (3,),
            dtype=np.uint8,
        )

    return cv2.normalize(
        image_f,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)


def flatten_background(
    gray: np.ndarray,
    blur_size: int = 51,
) -> np.ndarray:

    kernel = blur_size if blur_size % 2 == 1 else blur_size + 1
    kernel = max(kernel, 3)

    background = cv2.GaussianBlur(
        gray,
        (kernel, kernel),
        0,
    )

    flattened = cv2.subtract(
        gray,
        background,
    )

    return cv2.normalize(
        flattened,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )


def robust_mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0

    med = float(np.median(values))

    return float(
        1.4826
        * np.median(
            np.abs(
                values.astype(np.float64) - med
            )
        )
    )


def contour_hollowness(
    gray: np.ndarray,
    contour: np.ndarray,
) -> float:

    moments = cv2.moments(contour)

    if moments["m00"] <= 0:
        return 0.0

    cx = int(
        moments["m10"]
        / moments["m00"]
    )

    cy = int(
        moments["m01"]
        / moments["m00"]
    )

    if (
        cy < 0
        or cy >= gray.shape[0]
        or cx < 0
        or cx >= gray.shape[1]
    ):
        return 0.0

    mask = np.zeros_like(gray)

    cv2.drawContours(
        mask,
        [contour],
        0,
        255,
        -1,
    )

    _, max_val, _, _ = cv2.minMaxLoc(
        gray,
        mask=mask,
    )

    if max_val <= 0:
        return 0.0

    return clamp(
        1.0
        - (
            float(gray[cy, cx])
            / (float(max_val) + 1e-6)
        )
    )


def measure_star_profile(
    gray: np.ndarray,
    contour: np.ndarray,
) -> Optional[Dict[str, float]]:

    x, y, w, h = cv2.boundingRect(contour)

    pad = int(
        max(
            8,
            min(
                32,
                max(w, h) * 2,
            ),
        )
    )

    y0 = max(0, y - pad)
    y1 = min(gray.shape[0], y + h + pad)

    x0 = max(0, x - pad)
    x1 = min(gray.shape[1], x + w + pad)

    patch = gray[
        y0:y1,
        x0:x1,
    ].astype(np.float64)

    if patch.size < 25:
        return None

    border = np.concatenate(
        [
            patch[0, :],
            patch[-1, :],
            patch[:, 0],
            patch[:, -1],
        ]
    )

    background = float(
        np.median(border)
    )

    signal = np.maximum(
        patch - background,
        0.0,
    )

    total_flux = float(
        np.sum(signal)
    )

    peak_flux = float(
        np.max(signal)
    )

    if (
        total_flux <= 1e-6
        or peak_flux < 3.0
    ):
        return None

    yy, xx = np.indices(
        patch.shape,
        dtype=np.float64,
    )

    cx = float(
        np.sum(xx * signal)
        / total_flux
    )

    cy = float(
        np.sum(yy * signal)
        / total_flux
    )

    dx = xx - cx
    dy = yy - cy

    var_x = float(
        np.sum(
            signal * dx * dx
        )
        / total_flux
    )

    var_y = float(
        np.sum(
            signal * dy * dy
        )
        / total_flux
    )

    cov_xy = float(
        np.sum(
            signal * dx * dy
        )
        / total_flux
    )

    cov = np.array(
        [
            [var_x, cov_xy],
            [cov_xy, var_y],
        ],
        dtype=np.float64,
    )

    eigvals = np.linalg.eigvalsh(cov)

    sigma_minor = math.sqrt(
        max(
            float(eigvals[0]),
            1e-6,
        )
    )

    sigma_major = math.sqrt(
        max(
            float(eigvals[1]),
            1e-6,
        )
    )

    fwhm_major = (
        2.3548 * sigma_major
    )

    fwhm_minor = (
        2.3548 * sigma_minor
    )

    fwhm = 0.5 * (
        fwhm_major
        + fwhm_minor
    )

    fwhm_ratio = (
        fwhm_major
        / max(
            fwhm_minor,
            1e-6,
        )
    )

    eccentricity = math.sqrt(
        max(
            0.0,
            1.0
            - (
                sigma_minor
                * sigma_minor
            )
            / max(
                sigma_major
                * sigma_major,
                1e-6,
            ),
        )
    )

    radii = np.sqrt(
        dx * dx + dy * dy
    ).reshape(-1)

    flux = signal.reshape(-1)

    order = np.argsort(radii)

    cumulative = np.cumsum(
        flux[order]
    )

    half_flux_index = int(
        np.searchsorted(
            cumulative,
            total_flux * 0.5,
        )
    )

    half_flux_index = min(
        half_flux_index,
        len(order) - 1,
    )

    hfr = float(
        radii[order][
            half_flux_index
        ]
    )

    return {
        "fwhm": float(fwhm),
        "hfr": hfr,
        "eccentricity": float(
            eccentricity
        ),
        "fwhm_ratio": float(
            fwhm_ratio
        ),
    }


def detect_star_contours(
    gray: np.ndarray,
) -> Tuple[List[np.ndarray], float, float]:

    flat = flatten_background(gray)

    mean_val, std_val = cv2.meanStdDev(
        flat
    )

    thresh_val = max(
        float(
            mean_val[0][0]
            + 2.0 * std_val[0][0]
        ),
        35.0,
    )

    _, mask = cv2.threshold(
        flat,
        thresh_val,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    bright_area_ratio = float(
        np.count_nonzero(mask)
        / mask.size
    )

    return (
        list(contours),
        bright_area_ratio,
        thresh_val,
    )


def detect_lines(
    gray: np.ndarray,
) -> Tuple[int, float, float, float, float]:

    h, w = gray.shape
    diag = math.hypot(w, h)

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=max(
            30,
            int(min(h, w) * 0.20),
        ),
        maxLineGap=max(
            10,
            int(min(h, w) * 0.03),
        ),
    )

    if lines is None:
        return (
            0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    lengths: List[float] = []
    angles: List[float] = []

    axis_aligned = 0
    diagonal = 0

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [
            int(v)
            for v in line
        ]

        length = math.hypot(
            x2 - x1,
            y2 - y1,
        )

        angle = (
            abs(
                math.degrees(
                    math.atan2(
                        y2 - y1,
                        x2 - x1,
                    )
                )
            )
            % 180.0
        )

        angle_to_axis = min(
            angle,
            abs(angle - 90.0),
            abs(angle - 180.0),
        )

        lengths.append(length)
        angles.append(angle)

        if angle_to_axis <= 10.0:
            axis_aligned += 1

        elif 15.0 <= angle <= 165.0:
            diagonal += 1

    longest_line_ratio = (
        max(lengths) / diag
        if lengths
        else 0.0
    )

    count = len(lengths)

    return (
        count,
        float(longest_line_ratio),
        float(
            axis_aligned / count
        )
        if count
        else 0.0,
        float(
            diagonal / count
        )
        if count
        else 0.0,
        float(
            np.std(angles)
        )
        if len(angles) > 1
        else 0.0,
    )


def detect_bright_streaks(
    gray: np.ndarray,
) -> Tuple[int, float, float, float]:

    h, w = gray.shape
    diag = math.hypot(w, h)

    mean = float(
        np.mean(gray)
    )

    std = float(
        np.std(gray)
    )

    image_max = float(
        np.max(gray)
    )

    threshold = max(
        float(
            np.percentile(
                gray,
                95.0,
            )
        ),
        mean + 3.0 * std,
        40.0,
    )

    threshold = min(
        threshold,
        max(
            image_max - 1.0,
            0.0,
        ),
    )

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    close_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    streak_count = 0
    max_length_ratio = 0.0
    max_aspect_ratio = 0.0
    max_area_ratio = 0.0

    for contour in contours:

        area = float(
            cv2.contourArea(contour)
        )

        if area < 20.0:
            continue

        rect = cv2.minAreaRect(
            contour
        )

        rw, rh = rect[1]

        long_side = max(
            float(rw),
            float(rh),
        )

        short_side = max(
            min(
                float(rw),
                float(rh),
            ),
            1.0,
        )

        aspect_ratio = (
            long_side
            / short_side
        )

        length_ratio = (
            long_side / diag
        )

        area_ratio = (
            area
            / float(h * w)
        )

        max_length_ratio = max(
            max_length_ratio,
            length_ratio,
        )

        max_aspect_ratio = max(
            max_aspect_ratio,
            aspect_ratio,
        )

        max_area_ratio = max(
            max_area_ratio,
            area_ratio,
        )

        if (
            length_ratio >= 0.25
            and aspect_ratio >= 6.0
        ):
            streak_count += 1

    return (
        streak_count,
        max_length_ratio,
        max_aspect_ratio,
        max_area_ratio,
    )


def elongated_axis_angle(
    contour: np.ndarray,
) -> float:

    (_, _), (rw, rh), angle = (
        cv2.minAreaRect(contour)
    )

    if rw < rh:
        angle += 90.0

    return float(
        angle % 180.0
    )


def radial_direction_angle(
    cx: float,
    cy: float,
    img_cx: float,
    img_cy: float,
) -> float:
    """
    มุมของเวกเตอร์จากจุดศูนย์กลางภาพ
    ไปยังตำแหน่งดาว (mod 180)
    เพื่อเทียบกับแกนรีที่ไม่มีทิศ
    """

    dx = cx - img_cx
    dy = cy - img_cy

    if (
        abs(dx) < 1e-6
        and abs(dy) < 1e-6
    ):
        return 0.0

    return float(
        math.degrees(
            math.atan2(dy, dx)
        )
        % 180.0
    )


def angular_alignment(
    angle_a: float,
    angle_b: float,
) -> float:
    """
    ความสอดคล้องของมุม 2 มุม (mod 180)

    1.0 = ทิศเดียวกันเป๊ะ
    0.0 = ตั้งฉากกัน
    """

    diff = (
        abs(angle_a - angle_b)
        % 180.0
    )

    if diff > 90.0:
        diff = 180.0 - diff

    return float(
        1.0 - diff / 90.0
    )


def axial_angle_stats(
    angles: List[float],
) -> Tuple[float, float]:

    if len(angles) < 2:
        return 0.0, 180.0

    doubled = np.deg2rad(
        np.asarray(
            angles,
            dtype=np.float64,
        )
        * 2.0
    )

    mean_cos = float(
        np.mean(
            np.cos(doubled)
        )
    )

    mean_sin = float(
        np.mean(
            np.sin(doubled)
        )
    )

    resultant = clamp(
        math.hypot(
            mean_cos,
            mean_sin,
        )
    )

    if resultant <= 1e-6:
        return resultant, 90.0

    std_rad = (
        math.sqrt(
            max(
                0.0,
                -2.0
                * math.log(resultant),
            )
        )
        / 2.0
    )

    return (
        resultant,
        float(
            np.rad2deg(std_rad)
        ),
    )


def extract_features(
    gray: np.ndarray,
) -> AstroFeatures:

    if gray.ndim != 2:
        raise ValueError(
            "extract_features expects a grayscale image"
        )

    h, w = gray.shape
    total_pixels = h * w
    gray_f = gray.astype(
        np.float64
    )

    contours, bright_area_ratio, _ = (
        detect_star_contours(gray)
    )

    star_count = 0
    elongated_count = 0

    areas: List[float] = []
    circularities: List[float] = []
    aspect_ratios: List[float] = []
    elongated_angles: List[float] = []
    elongated_radial_alignments: List[float] = []
    hollownesses: List[float] = []
    fwhms: List[float] = []
    hfrs: List[float] = []
    eccentricities: List[float] = []
    fwhm_ratios: List[float] = []

    img_cx = w / 2.0
    img_cy = h / 2.0

    max_star_area = max(
        5000.0,
        total_pixels * 0.02,
    )

    for contour in contours:

        area = float(
            cv2.contourArea(contour)
        )

        if (
            area < 3.0
            or area > max_star_area
        ):
            continue

        perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        if perimeter <= 0:
            continue

        x, y, bw, bh = (
            cv2.boundingRect(contour)
        )

        aspect_ratio = (
            max(bw, bh)
            / max(
                min(bw, bh),
                1,
            )
        )

        circularity = (
            4.0
            * math.pi
            * area
            / (perimeter * perimeter)
        )

        star_count += 1

        areas.append(area)
        aspect_ratios.append(
            float(aspect_ratio)
        )

        circularities.append(
            float(circularity)
        )

        hollownesses.append(
            contour_hollowness(
                gray,
                contour,
            )
        )

        profile = (
            measure_star_profile(
                gray,
                contour,
            )
        )

        if profile is not None:
            fwhms.append(
                profile["fwhm"]
            )

            hfrs.append(
                profile["hfr"]
            )

            eccentricities.append(
                profile["eccentricity"]
            )

            fwhm_ratios.append(
                profile["fwhm_ratio"]
            )

        if (
            aspect_ratio > 1.45
            or (
                area > 80.0
                and aspect_ratio > 1.25
            )
        ):

            elongated_count += 1

            star_angle = (
                elongated_axis_angle(
                    contour
                )
            )

            elongated_angles.append(
                star_angle
            )

            star_cx = (
                x + bw / 2.0
            )

            star_cy = (
                y + bh / 2.0
            )

            radial_angle = (
                radial_direction_angle(
                    star_cx,
                    star_cy,
                    img_cx,
                    img_cy,
                )
            )

            elongated_radial_alignments.append(
                angular_alignment(
                    star_angle,
                    radial_angle,
                )
            )

    (
        long_line_count,
        longest_line_ratio,
        axis_ratio,
        diagonal_ratio,
        angle_std,
    ) = detect_lines(gray)

    (
        streak_count,
        max_streak_length_ratio,
        max_streak_aspect_ratio,
        max_streak_area_ratio,
    ) = detect_bright_streaks(gray)

    (
        elongated_angle_consistency,
        elongated_angle_std,
    ) = axial_angle_stats(
        elongated_angles
    )

    elongated_radial_alignment = (
        float(
            np.mean(
                elongated_radial_alignments
            )
        )
        if elongated_radial_alignments
        else 0.0
    )

    return AstroFeatures(
        width=w,
        height=h,
        mean_intensity=float(
            np.mean(gray_f)
        ),
        std_intensity=float(
            np.std(gray_f)
        ),
        background_median=float(
            np.median(gray_f)
        ),
        background_mad=robust_mad(
            gray_f
        ),
        sharpness=float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        ),
        saturated_ratio=float(
            np.count_nonzero(
                gray >= 250
            )
            / total_pixels
        ),
        bright_area_ratio=bright_area_ratio,
        star_count=star_count,
        mean_star_area=float(
            np.mean(areas)
        )
        if areas
        else 0.0,
        mean_circularity=float(
            np.mean(circularities)
        )
        if circularities
        else 0.0,
        elongated_star_count=elongated_count,
        elongated_star_ratio=float(
            elongated_count
            / star_count
        )
        if star_count
        else 0.0,
        elongated_angle_consistency=(
            elongated_angle_consistency
        ),
        elongated_angle_std=(
            elongated_angle_std
        ),
        elongated_radial_alignment=(
            elongated_radial_alignment
        ),
        mean_aspect_ratio=float(
            np.mean(aspect_ratios)
        )
        if aspect_ratios
        else 0.0,
        max_aspect_ratio=float(
            np.max(aspect_ratios)
        )
        if aspect_ratios
        else 0.0,
        mean_hollowness=float(
            np.mean(hollownesses)
        )
        if hollownesses
        else 0.0,
        valid_profile_count=len(
            fwhms
        ),
        mean_fwhm=float(
            np.mean(fwhms)
        )
        if fwhms
        else 0.0,
        median_fwhm=float(
            np.median(fwhms)
        )
        if fwhms
        else 0.0,
        mean_hfr=float(
            np.mean(hfrs)
        )
        if hfrs
        else 0.0,
        median_hfr=float(
            np.median(hfrs)
        )
        if hfrs
        else 0.0,
        median_eccentricity=float(
            np.median(eccentricities)
        )
        if eccentricities
        else 0.0,
        median_fwhm_ratio=float(
            np.median(fwhm_ratios)
        )
        if fwhm_ratios
        else 0.0,
        long_line_count=long_line_count,
        longest_line_ratio=longest_line_ratio,
        axis_aligned_line_ratio=axis_ratio,
        diagonal_line_ratio=diagonal_ratio,
        line_angle_std=angle_std,
        streak_count=streak_count,
        max_streak_length_ratio=(
            max_streak_length_ratio
        ),
        max_streak_aspect_ratio=(
            max_streak_aspect_ratio
        ),
        max_streak_area_ratio=(
            max_streak_area_ratio
        ),
    )


def score_good(
    f: AstroFeatures,
) -> float:

    if (
        f.star_count < 5
        or f.valid_profile_count < 5
    ):
        return 0.05

    base = weighted_mean(
        [
            (
                score_high(
                    f.star_count,
                    20,
                    300,
                ),
                1.3,
            ),
            (
                score_high(
                    f.sharpness,
                    1000,
                    15000,
                ),
                0.3,
            ),
            (
                score_low(
                    f.median_fwhm,
                    18.0,
                    30.0,
                ),
                0.8,
            ),
            (
                score_low(
                    f.median_eccentricity,
                    0.45,
                    0.75,
                ),
                0.8,
            ),
            (
                score_high(
                    f.mean_circularity,
                    0.20,
                    0.55,
                ),
                0.2,
            ),
            (
                score_low(
                    f.saturated_ratio,
                    0.003,
                    0.025,
                ),
                1.0,
            ),
            (
                score_low(
                    f.bright_area_ratio,
                    0.001,
                    0.010,
                ),
                0.8,
            ),
            (
                score_low(
                    f.elongated_star_ratio,
                    0.05,
                    0.15,
                ),
                1.1,
            ),
            (
                score_low(
                    f.longest_line_ratio,
                    0.10,
                    0.35,
                ),
                1.0,
            ),
        ]
    )

    outlier_penalty = score_low(
        f.max_aspect_ratio,
        5.0,
        10.0,
    )

    streak_length_penalty = score_low(
        f.max_streak_length_ratio,
        0.55,
        0.75,
    )

    streak_aspect_penalty = score_low(
        f.max_streak_aspect_ratio,
        7.0,
        15.0,
    )

    return (
        base
        * outlier_penalty
        * streak_length_penalty
        * streak_aspect_penalty
    )


def score_out_of_focus(
    f: AstroFeatures,
) -> float:

    return weighted_mean(
        [
            (
                score_high(
                    f.mean_hollowness,
                    0.005,
                    0.050,
                ),
                1.7,
            ),
            (
                score_high(
                    f.mean_star_area,
                    45.0,
                    140.0,
                ),
                1.3,
            ),
            (
                score_high(
                    f.median_fwhm,
                    18.0,
                    30.0,
                ),
                1.1,
            ),
            (
                score_high(
                    f.median_hfr,
                    7.0,
                    12.0,
                ),
                0.8,
            ),
            (
                score_low(
                    f.mean_circularity,
                    0.72,
                    0.82,
                ),
                0.8,
            ),
            (
                score_low(
                    f.sharpness,
                    15000,
                    60000,
                ),
                0.8,
            ),
            (
                score_low(
                    f.max_streak_length_ratio,
                    0.15,
                    0.35,
                ),
                0.5,
            ),
        ]
    )


def score_tracking_error(
    f: AstroFeatures,
) -> float:

    base = weighted_mean(
        [
            (
                score_high(
                    f.sharpness,
                    5000,
                    20000,
                ),
                0.8,
            ),
            (
                score_high(
                    f.elongated_star_ratio,
                    0.10,
                    0.25,
                ),
                1.3,
            ),
            (
                score_high(
                    f.mean_aspect_ratio,
                    1.12,
                    1.30,
                ),
                1.0,
            ),
            (
                score_high(
                    f.median_eccentricity,
                    0.35,
                    0.65,
                ),
                1.1,
            ),
            (
                score_high(
                    f.median_fwhm_ratio,
                    1.06,
                    1.25,
                ),
                1.0,
            ),
            (
                score_low(
                    f.mean_hollowness,
                    0.04,
                    0.16,
                ),
                0.8,
            ),
            (
                score_high(
                    f.star_count,
                    5,
                    80,
                ),
                0.8,
            ),
            (
                score_low(
                    f.bright_area_ratio,
                    0.008,
                    0.025,
                ),
                0.8,
            ),
            (
                score_low(
                    f.max_streak_length_ratio,
                    0.18,
                    0.55,
                ),
                0.7,
            ),
            (
                score_high(
                    f.elongated_angle_consistency,
                    0.35,
                    0.75,
                ),
                0.25,
            ),
        ]
    )

    if f.elongated_star_count >= 3:
        radial_gate = score_low(
            f.elongated_radial_alignment,
            0.55,
            0.80,
        )
    else:
        radial_gate = 1.0

    return base * radial_gate


def score_over_saturated(
    f: AstroFeatures,
) -> float:

    return weighted_mean(
        [
            (
                score_low(
                    f.sharpness,
                    1000.0,
                    5000.0,
                ),
                2.0,
            ),
            (
                score_high(
                    f.median_eccentricity,
                    0.62,
                    0.75,
                ),
                1.4,
            ),
            (
                score_high(
                    f.median_fwhm_ratio,
                    1.25,
                    1.50,
                ),
                1.3,
            ),
            (
                score_high(
                    f.bright_area_ratio,
                    0.015,
                    0.030,
                ),
                1.2,
            ),
            (
                score_high(
                    f.star_count,
                    180,
                    320,
                ),
                0.9,
            ),
            (
                score_low(
                    f.median_hfr,
                    5.0,
                    9.0,
                ),
                0.8,
            ),
            (
                score_low(
                    f.median_fwhm,
                    10.0,
                    20.0,
                ),
                0.8,
            ),
            (
                score_high(
                    f.axis_aligned_line_ratio,
                    0.10,
                    0.80,
                ),
                0.4,
            ),
        ]
    )


def score_no_star(
    f: AstroFeatures,
) -> float:

    return weighted_mean(
        [
            (
                score_low(
                    f.star_count,
                    3,
                    15,
                ),
                1.5,
            ),
            (
                score_low(
                    f.valid_profile_count,
                    2,
                    10,
                ),
                1.0,
            ),
            (
                score_low(
                    f.bright_area_ratio,
                    0.005,
                    0.050,
                ),
                1.1,
            ),
            (
                score_low(
                    f.std_intensity,
                    4,
                    20,
                ),
                0.8,
            ),
            (
                score_low(
                    f.long_line_count,
                    0.5,
                    2.0,
                ),
                0.5,
            ),
        ]
    )


def score_satellite(
    f: AstroFeatures,
) -> float:

    return weighted_mean(
        [
            (
                score_high(
                    f.streak_count,
                    0.5,
                    1.0,
                ),
                1.5,
            ),
            (
                score_high(
                    f.max_streak_length_ratio,
                    0.18,
                    0.55,
                ),
                1.4,
            ),
            (
                score_high(
                    f.max_streak_aspect_ratio,
                    4.0,
                    14.0,
                ),
                1.2,
            ),
            (
                score_high(
                    f.longest_line_ratio,
                    0.30,
                    0.80,
                ),
                0.4,
            ),
            (
                score_low(
                    f.saturated_ratio,
                    0.010,
                    0.080,
                ),
                0.6,
            ),
            (
                score_low(
                    f.axis_aligned_line_ratio,
                    0.15,
                    0.50,
                ),
                0.8,
            ),
        ]
    )


RULE_SCORERS: Dict[
    str,
    Callable[[AstroFeatures], float],
] = {
    "01_Good": score_good,
    "02_Out_of_Focus": score_out_of_focus,
    "03_Tracking_Error": score_tracking_error,
    "04_Over_Saturated": score_over_saturated,
    "05_No_Star": score_no_star,
    "06_Satellite": score_satellite,
}


def score_rules(
    features: AstroFeatures,
) -> Dict[str, float]:

    return {
        name: scorer(features)
        for name, scorer in RULE_SCORERS.items()
    }


# ============================================================
# CNN / ONNX
# ส่วนนี้คง logic เดิมไว้
# ============================================================


def softmax(
    logits: np.ndarray,
) -> np.ndarray:

    logits = logits.astype(
        np.float64
    )

    logits = (
        logits
        - np.max(logits)
    )

    exp = np.exp(logits)

    return exp / np.sum(exp)


def preprocess_for_onnx(
    gray: np.ndarray,
    input_shape: Sequence[int],
) -> np.ndarray:

    dims = list(input_shape)

    if len(dims) != 4:
        raise ValueError(
            "Only 4D ONNX image inputs "
            f"are supported, got {input_shape}"
        )

    n, c, h, w = dims

    h = (
        224
        if not isinstance(h, int)
        else h
    )

    w = (
        224
        if not isinstance(w, int)
        else w
    )

    c = (
        1
        if not isinstance(c, int)
        else c
    )

    resized = cv2.resize(
        gray,
        (w, h),
        interpolation=cv2.INTER_AREA,
    )

    image = (
        resized.astype(np.float32)
        / 255.0
    )

    if c == 1:

        image = image[
            None,
            None,
            :,
            :,
        ]

    elif c == 3:

        image = cv2.cvtColor(
            resized,
            cv2.COLOR_GRAY2RGB,
        ).astype(np.float32) / 255.0

        mean = np.array(
            [
                0.485,
                0.456,
                0.406,
            ],
            dtype=np.float32,
        )

        std = np.array(
            [
                0.229,
                0.224,
                0.225,
            ],
            dtype=np.float32,
        )

        image = (
            image - mean
        ) / std

        image = np.transpose(
            image,
            (2, 0, 1),
        )[
            None,
            :,
            :,
            :,
        ]

    else:

        raise ValueError(
            "Unsupported channel "
            f"count: {c}"
        )

    if (
        isinstance(n, int)
        and n != 1
    ):
        raise ValueError(
            "This helper supports "
            "batch size 1"
        )

    return image


def preprocess_bgr_for_cv2_dnn(
    image_bgr: np.ndarray,
    size: Tuple[int, int] = (
        640,
        640,
    ),
) -> np.ndarray:

    image_rgb = cv2.cvtColor(
        cv2.resize(
            image_bgr,
            size,
            interpolation=cv2.INTER_AREA,
        ),
        cv2.COLOR_BGR2RGB,
    )

    image = (
        image_rgb.astype(
            np.float32
        )
        / 255.0
    )

    mean = np.array(
        [
            0.485,
            0.456,
            0.406,
        ],
        dtype=np.float32,
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225,
        ],
        dtype=np.float32,
    )

    image = (
        image - mean
    ) / std

    return np.transpose(
        image,
        (2, 0, 1),
    )[
        None,
        :,
        :,
        :,
    ]


def predict_onnx_cv2_dnn(
    image_path: str | Path,
    onnx_path: str | Path,
) -> Dict[str, float]:

    image = read_bgr_image(
        image_path
    )

    net = cv2.dnn.readNetFromONNX(
        str(onnx_path)
    )

    net.setInput(
        preprocess_bgr_for_cv2_dnn(
            image
        )
    )

    raw = np.asarray(
        net.forward()
    ).reshape(-1)

    probs = (
        raw
        if (
            np.isclose(
                np.sum(raw),
                1.0,
                atol=1e-3,
            )
            and np.all(raw >= 0)
        )
        else softmax(raw)
    )

    if len(probs) != len(
        CLASS_NAMES
    ):
        raise ValueError(
            f"Model returned {len(probs)} "
            f"scores, expected "
            f"{len(CLASS_NAMES)}"
        )

    return {
        name: float(probs[i])
        for i, name in enumerate(
            CLASS_NAMES
        )
    }


def predict_onnx(
    image_path: str | Path,
    onnx_path: str | Path,
) -> Dict[str, float]:

    try:
        import onnxruntime as ort

    except ImportError:

        return predict_onnx_cv2_dnn(
            image_path,
            onnx_path,
        )

    gray = read_gray_image(
        image_path
    )

    session = ort.InferenceSession(
        str(onnx_path),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_meta = (
        session.get_inputs()[0]
    )

    input_tensor = (
        preprocess_for_onnx(
            gray,
            input_meta.shape,
        )
    )

    outputs = session.run(
        None,
        {
            input_meta.name:
                input_tensor
        },
    )

    raw = np.asarray(
        outputs[0]
    ).reshape(-1)

    probs = (
        raw
        if (
            np.isclose(
                np.sum(raw),
                1.0,
                atol=1e-3,
            )
            and np.all(raw >= 0)
        )
        else softmax(raw)
    )

    if len(probs) != len(
        CLASS_NAMES
    ):
        raise ValueError(
            f"Model returned {len(probs)} "
            f"scores, expected "
            f"{len(CLASS_NAMES)}"
        )

    return {
        name: float(probs[i])
        for i, name in enumerate(
            CLASS_NAMES
        )
    }


# ============================================================
# CNN → RULE → APPROVE / REJECT
# ============================================================


def predict_with_rules(
    image_path: str | Path,
    onnx_path: Optional[str | Path] = None,
    model_scores: Optional[
        Dict[str, float]
    ] = None,
    approve_threshold: float = 0.45,
) -> Dict[str, object]:
    """
    ระบบตัดสินแบบใหม่:

        Image
          ↓
        CNN
          ↓
        CNN predicted class
          ↓
        Rule ของ class นั้นเท่านั้น
          ↓
        Rule score
          ↓
        approve_threshold
          ↓
        APPROVE / REJECT

    สำคัญ:
    - CNN เป็นตัวเลือก class
    - Rule ไม่ได้ใช้เลือก class ใหม่
    - ไม่มีการ fuse CNN score กับ Rule score
    - ไม่มี final_class
    - ไม่มี rival class / margin
    """

    if not 0.0 <= approve_threshold <= 1.0:
        raise ValueError(
            "approve_threshold must be "
            "between 0.0 and 1.0"
        )

    gray = read_gray_image(
        image_path
    )

    features = extract_features(
        gray
    )

    if model_scores is None:

        if onnx_path is None:
            raise ValueError(
                "Either onnx_path or "
                "model_scores must be provided"
            )

        model_scores = predict_onnx(
            image_path,
            onnx_path,
        )

    missing = [
        name
        for name in CLASS_NAMES
        if name not in model_scores
    ]

    if missing:
        raise ValueError(
            "Missing model score keys: "
            f"{missing}"
        )

    model_scores = {
        name: float(
            model_scores[name]
        )
        for name in CLASS_NAMES
    }

    # --------------------------------------------------------
    # 1. CNN เลือก class
    # --------------------------------------------------------

    model_class = max(
        model_scores,
        key=model_scores.get,
    )

    model_confidence = float(
        model_scores[model_class]
    )

    # --------------------------------------------------------
    # 2. ใช้ Rule ของ CNN class เท่านั้น
    # --------------------------------------------------------

    selected_rule = RULE_SCORERS[
        model_class
    ]

    rule_score = float(
        selected_rule(features)
    )

    # --------------------------------------------------------
    # 3. ตัดสิน APPROVE / REJECT
    # --------------------------------------------------------

    approved = (
        rule_score
        >= approve_threshold
    )

    decision = (
        "APPROVE"
        if approved
        else "REJECT"
    )

    # --------------------------------------------------------
    # 4. คำนวณ rule scores ทั้งหมด
    #
    # เก็บไว้เพื่อ diagnostic / analysis เท่านั้น
    # ไม่ได้ใช้ตัดสิน APPROVE / REJECT
    # --------------------------------------------------------

    all_rule_scores = score_rules(
        features
    )

    return {
        "image_path": str(
            image_path
        ),

        # CNN result
        "model_class": model_class,
        "model_confidence": (
            model_confidence
        ),
        "model_scores": model_scores,

        # Rule ที่ถูกเลือกตาม CNN
        "selected_rule_class": (
            model_class
        ),
        "rule_score": rule_score,

        # Threshold
        "approve_threshold": float(
            approve_threshold
        ),

        # Final decision
        "decision": decision,

        # Diagnostic เท่านั้น
        "rule_scores": all_rule_scores,

        # Features
        "features": asdict(
            features
        ),
    }


def parse_model_scores(
    value: Optional[str],
) -> Optional[Dict[str, float]]:

    if not value:
        return None

    parsed = json.loads(
        value
    )

    missing = [
        name
        for name in CLASS_NAMES
        if name not in parsed
    ]

    if missing:
        raise ValueError(
            f"Missing model score keys: "
            f"{missing}"
        )

    return {
        name: float(parsed[name])
        for name in CLASS_NAMES
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Astro image classification "
            "with CNN + class-specific "
            "rule approval"
        )
    )

    parser.add_argument(
        "image",
        help=(
            "Path to an astronomy image"
        ),
    )

    parser.add_argument(
        "--onnx",
        help=(
            "Path to ONNX model"
        ),
    )

    parser.add_argument(
        "--model-scores",
        help=(
            "JSON object with "
            "probabilities for all "
            "6 classes"
        ),
    )

    parser.add_argument(
        "--approve-threshold",
        type=float,
        default=0.45,
        help=(
            "Rule score threshold for "
            "APPROVE. Default: 0.45"
        ),
    )

    args = parser.parse_args()

    result = predict_with_rules(
        image_path=args.image,
        onnx_path=args.onnx,
        model_scores=parse_model_scores(
            args.model_scores
        ),
        approve_threshold=(
            args.approve_threshold
        ),
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()