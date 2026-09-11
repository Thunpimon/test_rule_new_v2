"""
test_star_profile_graph.py
==========================
Dedicated Tool to Extract & Visualize Star Radial/Cross-Section Intensity Profiles
(การทดสอบและพล็อตกราฟหน้าตัดความเข้มแสงของดวงดาว: Good vs Out-of-Focus vs Over-Saturated)

This script:
  1. Finds representative stars from Good, Out of Focus, and Over Saturated images.
  2. Extracts a 1D cross-section slice across the star's core (Profile Curve).
  3. Computes profile morphology metrics:
       - Central Dip / Donut Ratio: checks if center is darker than rim (Hollow Donut)
       - Top Flatness Width: checks if the peak is clipped/flat-topped (หัวตัด)
       - Peak Sharpness: compares peak vs flank width
  4. Plots a high-resolution comparison chart and saves it as 'star_profile_comparison.png'.

Usage:
    # Run comparison on default dataset samples:
    python test_star_profile_graph.py

    # Run on specific images:
    python test_star_profile_graph.py --good path/to/good.png --oof path/to/oof.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Use base reader from existing pipeline
from astro_pipeline_v2 import read_gray_image

DEFAULT_GOOD_IMG = Path("Dataset_For_Rule_Base/01_Good/260505J2WH_1_2.png")
DEFAULT_OOF_IMG = Path("Dataset_For_Rule_Base/02_Out_of_Focus/FILE_CURRENT_FOCUS12058 (9).png")
DEFAULT_OOF_IMG_2 = Path("Dataset_For_Rule_Base/02_Out_of_Focus/2606065NHT_1_2 (2).png")
DEFAULT_OVERSAT_IMG = Path("Dataset_For_Rule_Base/04_Over_Saturated/2605045QHM_1_2.png")


from astro_rule_classifier_new import detect_star_contours, measure_star_profile


def find_best_representative_star(
    gray: np.ndarray,
    prefer_donut: bool = False,
    prefer_saturated: bool = False,
    patch_size: int = 64,
) -> Optional[Dict[str, Any]]:
    """
    Find the cleanest, most representative star in the image and return its zoomed patch and profile.
    """
    h, w = gray.shape
    half = patch_size // 2

    # Use robust star detection with morphological noise filtering
    contours, _, _ = detect_star_contours(gray)
    if not contours:
        return None

    candidates = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 8.0 or area > (w * h * 0.05):
            continue

        m = cv2.moments(cnt)
        if m["m00"] <= 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        # Ignore border stars
        if cx < half or cx >= (w - half) or cy < half or cy >= (h - half):
            continue

        patch = gray[cy - half : cy + half, cx - half : cx + half].copy()
        if patch.shape != (patch_size, patch_size):
            continue

        peak_val = float(np.max(patch))
        center_val = float(patch[half, half])
        bg = float(np.median(patch))
        signal = max(peak_val - bg, 1.0)

        prof = measure_star_profile(gray, cnt)
        fwhm = float(prof["fwhm"]) if prof else 0.0
        eccentricity = float(prof["eccentricity"]) if prof else 1.0

        # Donut & Flatness metrics
        hollowness = max(0.0, 1.0 - (center_val / max(peak_val, 1e-6)))
        flatness_pixels = int(np.count_nonzero(patch >= (peak_val * 0.92)))

        candidates.append({
            "cx": cx,
            "cy": cy,
            "area": area,
            "patch": patch,
            "peak_val": peak_val,
            "center_val": center_val,
            "bg": bg,
            "signal": signal,
            "fwhm": fwhm,
            "eccentricity": eccentricity,
            "hollowness": hollowness,
            "flatness_pixels": flatness_pixels,
        })

    if not candidates:
        return None

    if prefer_donut:
        # Sort by hollowness and area (OOF donuts have a dark center hole and large area)
        candidates.sort(key=lambda c: (c["hollowness"] > 0.15, c["hollowness"], c["area"]), reverse=True)
        return candidates[0]
    elif prefer_saturated:
        # Check connected components of saturation (>= 250) for a textbook flat-top plateau
        _, sat_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        sat_cnts, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_sat = []
        for sc in sat_cnts:
            s_area = float(cv2.contourArea(sc))
            sm = cv2.moments(sc)
            if sm["m00"] <= 0:
                continue
            scx = int(sm["m10"] / sm["m00"])
            scy = int(sm["m01"] / sm["m00"])
            if scx < half or scx >= (w - half) or scy < half or scy >= (h - half):
                continue
            valid_sat.append((s_area, scx, scy))
        if valid_sat:
            # Prefer a saturated star with area ~300-1500 so it displays both the flat plateau and sloping flanks in the 64x64 frame
            well_framed = [item for item in valid_sat if 200.0 <= item[0] <= 1500.0]
            if well_framed:
                well_framed.sort(key=lambda item: abs(item[0] - 600.0))
                s_area, scx, scy = well_framed[0]
            else:
                valid_sat.sort(key=lambda item: item[0], reverse=True)
                s_area, scx, scy = valid_sat[0]
            patch = gray[scy - half : scy + half, scx - half : scx + half].copy()
            return {
                "cx": scx,
                "cy": scy,
                "area": s_area,
                "patch": patch,
                "peak_val": float(np.max(patch)),
                "center_val": float(patch[half, half]),
                "bg": float(np.median(patch)),
                "signal": float(np.max(patch)) - float(np.median(patch)),
                "fwhm": 0.0,
                "eccentricity": 0.0,
                "hollowness": 0.0,
                "flatness_pixels": int(s_area),
            }
        candidates.sort(key=lambda c: (c["flatness_pixels"], c["area"]), reverse=True)
        return candidates[0]
    else:
        # For Good In-Focus stars:
        # Ideal: moderate area (12-50), low eccentricity (<0.4), low hollowness (<0.1), FWHM 12-25
        good_pool = [
            c for c in candidates
            if c["hollowness"] < 0.10 and c["eccentricity"] < 0.45 and 12.0 <= c["area"] <= 60.0
        ]
        if good_pool:
            # Pick one with ideal FWHM closest to ~16 px
            good_pool.sort(key=lambda c: abs(c["fwhm"] - 16.0))
            return good_pool[0]

        # Fallback: lowest hollowness and lowest eccentricity
        candidates.sort(key=lambda c: (c["hollowness"], c["eccentricity"]))
        return candidates[0]


def analyze_1d_profile(patch: np.ndarray) -> Dict[str, Any]:
    """
    Extract a 1D horizontal cut across the star's center and analyze its profile shape.
    Uses 3-row averaging to suppress high-frequency grain noise while preserving optical PSF morphology.
    """
    size = patch.shape[0]
    mid = size // 2

    # 3-row average cross-section
    profile_1d = np.mean(patch[mid - 1 : mid + 2, :], axis=0).astype(np.float64)

    bg = float(np.median(patch))
    net_profile = np.maximum(profile_1d - bg, 0.0)
    peak_val = float(np.max(profile_1d))
    net_peak = float(np.max(net_profile))
    center_val = float(profile_1d[mid])

    # 1. Central Dip (Hollow Donut signature from secondary mirror obstruction)
    donut_dip_ratio = float(max(0.0, (peak_val - center_val) / max(peak_val, 1e-6)))
    is_donut = (donut_dip_ratio >= 0.25) and (peak_val > 50)

    # 2. Top Flatness (Clipped/Flat-top width at 92% of peak)
    top_indices = np.where(profile_1d >= (peak_val * 0.92))[0]
    flatness_width = len(top_indices)
    is_flat_top = (flatness_width >= 8) and (not is_donut)

    # 3. Optical FWHM / Width measurement
    half_level = bg + 0.5 * (peak_val - bg)
    if is_donut:
        # For hollow donut: measure outer diameter between outer half-power edges
        left_idx = 0
        while left_idx < mid and profile_1d[left_idx] < half_level:
            left_idx += 1
        right_idx = size - 1
        while right_idx > mid and profile_1d[right_idx] < half_level:
            right_idx -= 1
        fwhm_1d = float(max(0, right_idx - left_idx + 1))
    else:
        # For solid peak: measure width around the core peak
        left_idx = mid
        while left_idx > 0 and profile_1d[left_idx] >= half_level:
            left_idx -= 1
        right_idx = mid
        while right_idx < size - 1 and profile_1d[right_idx] >= half_level:
            right_idx += 1
        fwhm_1d = float(max(0, right_idx - left_idx))

    return {
        "x_coords": np.arange(size) - mid,  # centered at 0
        "raw_profile": profile_1d,
        "net_profile": net_profile,
        "background": bg,
        "peak_val": peak_val,
        "center_val": center_val,
        "fwhm_1d": fwhm_1d,
        "is_donut": is_donut,
        "donut_dip_ratio": donut_dip_ratio,
        "flatness_width": flatness_width,
        "is_flat_top": is_flat_top,
    }


def extract_multiple_stars(gray: np.ndarray, max_stars: int = 4, patch_size: int = 64) -> List[Dict[str, Any]]:
    """
    Extract up to max_stars distinct stars from a single image.
    """
    h, w = gray.shape
    half = patch_size // 2
    contours, _, _ = detect_star_contours(gray)
    if not contours:
        return []

    candidates = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 6.0:
            continue
        m = cv2.moments(cnt)
        if m["m00"] <= 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        if cx < half or cx >= (w - half) or cy < half or cy >= (h - half):
            continue

        patch = gray[cy - half : cy + half, cx - half : cx + half].copy()
        if patch.shape != (patch_size, patch_size):
            continue

        peak_val = float(np.max(patch))
        center_val = float(patch[half, half])
        bg = float(np.median(patch))
        hollowness = max(0.0, 1.0 - (center_val / max(peak_val, 1e-6)))

        candidates.append({
            "cx": cx,
            "cy": cy,
            "area": area,
            "patch": patch,
            "peak_val": peak_val,
            "center_val": center_val,
            "bg": bg,
            "hollowness": hollowness,
        })

    if not candidates:
        return []

    # Sort to show varied stars: hollow/donut stars first, then largest/brightest
    candidates.sort(key=lambda c: (c["hollowness"] > 0.20, c["hollowness"], c["area"]), reverse=True)
    selected = []
    for c in candidates:
        if all(math.hypot(c["cx"] - s["cx"], c["cy"] - s["cy"]) >= patch_size for s in selected):
            selected.append(c)
        if len(selected) >= max_stars:
            break
    return selected


def generate_profile_comparison_figure(
    samples: List[Tuple[str, str, Dict[str, Any]]],
    output_path: str | Path = "star_profile_comparison.png",
    show_window: bool = True,
    title_text: str = "Astronomical Star Radial / Cross-Section Profile Analysis",
) -> Path:
    """
    Generate a clean multi-panel chart comparing star profiles.
    """
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(max(4.5 * n, 5.0), 8), dpi=150, squeeze=False)
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    for i, (title, category, data) in enumerate(samples):
        patch = data["patch"]
        prof = analyze_1d_profile(patch)
        x = prof["x_coords"]
        y = prof["raw_profile"]
        mid = len(patch) // 2

        # Top Panel: Zoomed 2D Star Patch with cross-section line
        ax_img = axes[0, i]
        im = ax_img.imshow(patch, cmap="inferno", origin="upper")
        # Draw horizontal cut line across the middle
        ax_img.axhline(y=mid, color="cyan", linestyle="--", linewidth=1.5, label="1D Cut Line")
        ax_img.scatter([mid], [mid], color="lime", s=25, zorder=5)
        ax_img.set_title(f"{title}\n[{category}]", fontsize=11, fontweight="bold")
        ax_img.axis("off")

        # Bottom Panel: 1D Intensity Profile Graph
        ax_prof = axes[1, i]

        # Choose color based on category/metrics
        if prof["is_donut"]:
            line_color = "#e74c3c"
        elif prof["is_flat_top"]:
            line_color = "#f39c12"
        else:
            line_color = "#2ecc71"

        ax_prof.plot(x, y, color=line_color, linewidth=2.5, label="Intensity Profile")
        ax_prof.axhline(y=prof["background"], color="gray", linestyle=":", alpha=0.7, label=f"BG ({prof['background']:.0f})")
        ax_prof.axvline(x=0, color="cyan", linestyle="--", alpha=0.5)

        # Annotations on the profile
        if prof["is_donut"]:
            annotation_text = f"DONUT CENTRAL DIP!\nDip Ratio: {prof['donut_dip_ratio']*100:.1f}%\nFWHM: {prof['fwhm_1d']:.1f} px"
            box_color = "#fadbd8"
        elif prof["is_flat_top"]:
            annotation_text = f"FLAT-TOP / CLIPPED!\nWidth: {prof['flatness_width']} px\nFWHM: {prof['fwhm_1d']:.1f} px"
            box_color = "#fef9e7"
        else:
            annotation_text = f"Sharp Gaussian Peak\nPeak: {prof['peak_val']:.0f}\nFWHM: {prof['fwhm_1d']:.1f} px"
            box_color = "#d4efdf"

        ax_prof.text(
            0.05, 0.92, annotation_text,
            transform=ax_prof.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, alpha=0.9, edgecolor="gray"),
        )

        ax_prof.set_xlabel("Pixel Offset from Center (px)", fontsize=9)
        ax_prof.set_ylabel("Pixel Intensity (0-255)", fontsize=9)
        ax_prof.set_ylim(-5, 265)
        ax_prof.grid(True, linestyle="--", alpha=0.4)

    out_file = Path(output_path)
    fig.suptitle(title_text, fontsize=13, fontweight="bold", y=0.98)
    fig.savefig(out_file, bbox_inches="tight")
    if show_window:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize and compare star intensity profiles")
    parser.add_argument("images", nargs="*", default=[], help="Image path(s) to inspect (e.g. python test_star_profile_graph.py my_image.png)")
    parser.add_argument("--image", default=None, help="Path to a single image to inspect")
    parser.add_argument("--good", default=None, help="Path to Good image for comparison mode")
    parser.add_argument("--oof", default=None, help="Path to Out of Focus image for comparison mode")
    parser.add_argument("--oof2", default=None, help="Path to second Out of Focus image")
    parser.add_argument("--oversat", "--over", default=None, help="Path to Over Saturated image")
    parser.add_argument("--output", default="star_profile_comparison.png", help="Output plot filename")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive GUI popup window")
    args = parser.parse_args()

    print("=" * 80)
    print(" ASTRONOMICAL STAR PROFILE GRAPH ANALYZER")
    print("=" * 80)

    # Determine input mode
    input_files: List[Path] = []
    if args.image:
        input_files = [Path(args.image)]
    elif args.images:
        input_files = [Path(p) for p in args.images]

    samples_to_plot = []
    title_text = "Astronomical Star Radial / Cross-Section Profile Analysis"

    if len(input_files) == 1:
        # SINGLE IMAGE MODE: inspect up to 4 stars in this single image
        img_path = input_files[0]
        if not img_path.exists():
            print(f"[Error] File does not exist: {img_path}")
            return
        print(f"[Inspecting Single Image] {img_path.name}")
        gray = read_gray_image(img_path)
        stars = extract_multiple_stars(gray, max_stars=4)
        if not stars:
            # Fallback to single best star
            s = find_best_representative_star(gray, prefer_donut=True)
            if s:
                stars = [s]

        for idx, star in enumerate(stars, 1):
            prof = analyze_1d_profile(star["patch"])
            shape_desc = "Donut" if prof["is_donut"] else ("Flat-top" if prof["is_flat_top"] else "Point Star")
            print(f"  -> Star #{idx}: Pos=({star['cx']}, {star['cy']}), Peak={prof['peak_val']:.0f}, FWHM={prof['fwhm_1d']:.1f}px, Shape={shape_desc}")
            samples_to_plot.append((f"Star #{idx} @ ({star['cx']},{star['cy']})", img_path.name, star))
        title_text = f"Star Profile Analysis for Image: {img_path.name}"

    elif len(input_files) > 1:
        # MULTIPLE SPECIFIC IMAGES MODE: pick 1 representative star from each
        for img_path in input_files[:4]:
            if not img_path.exists():
                print(f"[Warning] File not found: {img_path}")
                continue
            print(f"[Loading] {img_path.name}")
            gray = read_gray_image(img_path)
            star = find_best_representative_star(gray, prefer_donut=True)
            if star:
                prof = analyze_1d_profile(star["patch"])
                print(f"  -> Star found in {img_path.name}: Peak={prof['peak_val']:.0f}, FWHM={prof['fwhm_1d']:.1f}px")
                samples_to_plot.append((img_path.name, "Custom Image", star))
        title_text = "Comparison of Custom Astronomical Images"

    else:
        # COMPARISON MODE
        any_explicit = any([args.good, args.oof, args.oof2, args.oversat])
        if any_explicit:
            p_good = Path(args.good) if args.good else None
            p_oof = Path(args.oof) if args.oof else None
            p_oof2 = Path(args.oof2) if args.oof2 else None
            p_sat = Path(args.oversat) if args.oversat else None
        else:
            p_good = DEFAULT_GOOD_IMG
            p_oof = DEFAULT_OOF_IMG
            p_oof2 = DEFAULT_OOF_IMG_2
            p_sat = DEFAULT_OVERSAT_IMG

        # 1. Good Star
        if p_good and p_good.exists():
            print(f"[Loading] 01_Good image: {p_good.name}")
            gray = read_gray_image(p_good)
            star = find_best_representative_star(gray, prefer_donut=False)
            if star:
                prof = analyze_1d_profile(star["patch"])
                print(f"  -> Good Star found: Peak={prof['peak_val']:.0f}, FWHM={prof['fwhm_1d']:.1f}px, Flatness={prof['flatness_width']}px")
                samples_to_plot.append(("In-Focus Star", "01_Good", star))

        # 2. OOF Star (Donut Profile)
        if p_oof and p_oof.exists():
            print(f"[Loading] 02_Out_of_Focus (Donut): {p_oof.name}")
            gray = read_gray_image(p_oof)
            star = find_best_representative_star(gray, prefer_donut=True)
            if star:
                prof = analyze_1d_profile(star["patch"])
                print(f"  -> OOF Star (Donut) found: Peak={prof['peak_val']:.0f}, Dip={prof['donut_dip_ratio']*100:.1f}%, FWHM={prof['fwhm_1d']:.1f}px")
                samples_to_plot.append(("Out of Focus (Donut)", "02_Out_of_Focus", star))

        # 3. OOF Star 2 (Broad/Swollen Profile)
        if p_oof2 and p_oof2.exists():
            print(f"[Loading] 02_Out_of_Focus (Swollen): {p_oof2.name}")
            gray = read_gray_image(p_oof2)
            star = find_best_representative_star(gray, prefer_donut=True)
            if star:
                prof = analyze_1d_profile(star["patch"])
                print(f"  -> OOF Star 2 found: Peak={prof['peak_val']:.0f}, FWHM={prof['fwhm_1d']:.1f}px")
                samples_to_plot.append(("Out of Focus (Swollen)", "02_Out_of_Focus", star))

        # 4. Over Saturated Star (Flat-top Saturation)
        if p_sat and p_sat.exists():
            print(f"[Loading] 04_Over_Saturated image: {p_sat.name}")
            gray = read_gray_image(p_sat)
            star = find_best_representative_star(gray, prefer_saturated=True)
            if star:
                prof = analyze_1d_profile(star["patch"])
                print(f"  -> Saturated Star found: Peak={prof['peak_val']:.0f}, Flat Width={prof['flatness_width']}px, FWHM={prof['fwhm_1d']:.1f}px")
                samples_to_plot.append(("Over Saturated (Clipped 255)", "04_Over_Saturated", star))

    if not samples_to_plot:
        print("[Error] No stars could be extracted from provided images.")
        return

    out_file = generate_profile_comparison_figure(
        samples_to_plot, output_path=args.output, show_window=not args.no_show, title_text=title_text
    )
    print("\n" + "=" * 80)
    print(f"[SUCCESS] Star profile comparison figure generated:")
    print(f"  -> {out_file.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
