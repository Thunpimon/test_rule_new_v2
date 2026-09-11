"""
astro_pipeline.py
=================
Unified Astronomy Image Quality Classification & 6-Filter Verification Pipeline

Architecture:
    Input Image
        │
        ├──► [CNN Model (ONNX)] ──────► Predicted Class, Confidences (%)
        │
        └──► [Feature Extraction] ────► AstroFeatures (FWHM, Eccentricity, etc.)
                   │
                   └──► [6 Rule Filters] ──► Filter Scores (%), Pass/Fail (Thresholds)
                             │
                             ▼
                 [Agreement & Comparison]
                   - CNN vs Top Filter Match?
                   - CNN Class Passed its Filter?
                   - Comprehensive Verdict & Summary

Usage:
    # As a Python Module (for Web UI / API):
    from astro_pipeline import run_pipeline
    result = run_pipeline("path/to/image.png")

    # Command Line:
    python astro_pipeline.py path/to/image.png
    python astro_pipeline.py path/to/image.png --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

# Import verified feature extraction and rule scoring functions
try:
    from astro_rule_classifier_new import (
        CLASS_NAMES,
        AstroFeatures,
        extract_features,
        predict_onnx,
        read_gray_image,
        score_rules,
    )
except ImportError as err:
    raise ImportError(
        f"Cannot import from astro_rule_classifier_new: {err}. "
        f"Ensure astro_rule_classifier_new.py is in the same directory or in PYTHONPATH."
    ) from err

# Default path to ONNX model
DEFAULT_ONNX_PATH = Path(__file__).parent / "eff_b0_kfold_add_focal_r2.onnx"

# Default per-class pass/fail thresholds (configurable)
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "01_Good": 0.50,
    "02_Out_of_Focus": 0.50,
    "03_Tracking_Error": 0.50,
    "04_Over_Saturated": 0.50,
    "05_No_Star": 0.50,
    "06_Satellite": 0.50,
}


def run_pipeline(
    image_path: str | Path,
    onnx_path: Optional[str | Path] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Run full pipeline on an astronomical image:
      1. CNN Model Prediction (Class probabilities & top-1)
      2. Astronomical Feature Extraction (OpenCV)
      3. All 6 Physics-based Filters evaluation (Scores % and Pass/Fail)
      4. Comparison & Agreement Verdict

    Returns a clean dictionary ready for Web API / Frontend JSON response.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    onnx_model_path = Path(onnx_path) if onnx_path else DEFAULT_ONNX_PATH
    active_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        active_thresholds.update(thresholds)

    # 1. Read Image and Extract Physical Features
    gray_image = read_gray_image(img_path)
    features: AstroFeatures = extract_features(gray_image)

    # 2. Run All 6 Rule Filters
    rule_scores_raw = score_rules(features)

    filter_details: Dict[str, Dict[str, Any]] = {}
    passed_filters = []

    for cname in CLASS_NAMES:
        raw_score = float(rule_scores_raw.get(cname, 0.0))
        th = float(active_thresholds.get(cname, 0.50))
        is_passed = bool(raw_score >= th)
        if is_passed:
            passed_filters.append(cname)

        filter_details[cname] = {
            "score": round(raw_score, 4),
            "score_pct": round(raw_score * 100, 2),
            "threshold": round(th, 4),
            "threshold_pct": round(th * 100, 2),
            "passed": is_passed,
            "status": "PASS" if is_passed else "FAIL",
        }

    # Top winning filter
    top_filter_class = max(rule_scores_raw, key=rule_scores_raw.get)
    top_filter_score = float(rule_scores_raw[top_filter_class])

    # 3. Run CNN Model (if ONNX model exists)
    has_model = onnx_model_path.exists()
    cnn_results: Dict[str, Any] = {}

    if has_model:
        model_scores_raw = predict_onnx(img_path, onnx_model_path)
        cnn_pred_class = max(model_scores_raw, key=model_scores_raw.get)
        cnn_confidence = float(model_scores_raw[cnn_pred_class])

        cnn_results = {
            "available": True,
            "predicted_class": cnn_pred_class,
            "confidence": round(cnn_confidence, 4),
            "confidence_pct": round(cnn_confidence * 100, 2),
            "probabilities": {k: round(v, 4) for k, v in model_scores_raw.items()},
            "probabilities_pct": {k: round(v * 100, 2) for k, v in model_scores_raw.items()},
        }
    else:
        cnn_pred_class = None
        cnn_results = {
            "available": False,
            "predicted_class": None,
            "confidence": 0.0,
            "confidence_pct": 0.0,
            "probabilities": {},
            "probabilities_pct": {},
            "warning": f"ONNX model not found at {onnx_model_path}",
        }

    # 4. Agreement & Comparison Analysis
    is_matched = bool(has_model and cnn_pred_class == top_filter_class)
    cnn_filter_passed = bool(has_model and filter_details.get(cnn_pred_class, {}).get("passed", False))

    if not has_model:
        agreement_status = "FILTER_ONLY"
        summary_th = f"ผลจากตัวกรองหลัก: {top_filter_class} (คะแนน {top_filter_score * 100:.1f}%)"
        summary_en = f"Filter only: {top_filter_class} ({top_filter_score * 100:.1f}%)"
    elif is_matched and cnn_filter_passed:
        agreement_status = "STRONG_MATCH"
        summary_th = f"✅ โมเดลและตัวกรองเห็นตรงกันสมบูรณ์ว่าเป็น '{cnn_pred_class}' และผ่านเกณฑ์ตัวกรอง"
        summary_en = f"Strong Match: Both CNN and Filters agree on '{cnn_pred_class}' (Filter Passed)."
    elif is_matched and not cnn_filter_passed:
        agreement_status = "WEAK_MATCH"
        summary_th = f"⚠️ โมเดลและตัวกรองชี้ไปที่ '{cnn_pred_class}' เหมือนกัน แต่คะแนนตัวกรองยังไม่ถึงเกณฑ์ขั้นต่ำ"
        summary_en = f"Weak Match: Both agree on '{cnn_pred_class}', but filter score did not meet threshold."
    else:
        agreement_status = "MISMATCH"
        summary_th = f"⚡ ผลไม่ตรงกัน: โมเดลทาย '{cnn_pred_class}' แต่ตัวกรองที่มีคะแนนสูงสุดคือ '{top_filter_class}'"
        summary_en = f"Mismatch: CNN predicted '{cnn_pred_class}', but highest filter is '{top_filter_class}'."

    # 5. Key Physical Features for Quick Display
    key_features = {
        "fwhm_median": round(features.median_fwhm, 2),
        "eccentricity_median": round(features.median_eccentricity, 3),
        "star_count": int(features.star_count),
        "mean_circularity": round(features.mean_circularity, 3),
        "mean_aspect_ratio": round(features.mean_aspect_ratio, 2),
        "mean_hollowness": round(features.mean_hollowness, 4),
        "saturated_ratio_pct": round(features.saturated_ratio * 100, 3),
        "streak_count": int(features.streak_count),
        "long_line_count": int(features.long_line_count),
        "mean_intensity": round(features.mean_intensity, 2),
    }

    return {
        "image_info": {
            "file_name": img_path.name,
            "file_path": str(img_path.resolve()),
            "width": int(features.width),
            "height": int(features.height),
        },
        "agreement": {
            "status": agreement_status,
            "is_matched": is_matched,
            "cnn_filter_passed": cnn_filter_passed,
            "summary_th": summary_th,
            "summary_en": summary_en,
        },
        "cnn": cnn_results,
        "filters": {
            "top_filter_class": top_filter_class,
            "top_filter_score": round(top_filter_score, 4),
            "top_filter_score_pct": round(top_filter_score * 100, 2),
            "passed_filters": passed_filters,
            "details": filter_details,
        },
        "key_physical_features": key_features,
        "raw_features": asdict(features),
    }


def print_cli_report(result: Dict[str, Any]) -> None:
    """Pretty print the pipeline evaluation result to terminal."""
    img_info = result["image_info"]
    cnn = result["cnn"]
    filters = result["filters"]
    agree = result["agreement"]
    feats = result["key_physical_features"]

    print("=" * 80)
    print(f" ASTRONOMY IMAGE PIPELINE REPORT: {img_info['file_name']}")
    print(f" Resolution: {img_info['width']} x {img_info['height']} px")
    print("=" * 80)

    # Summary box
    print(f"\n[ สรุปผลการประเมิน ]")
    print(f"  สถานะความสอดคล้อง : {agree['status']}")
    print(f"  คำอธิบาย          : {agree['summary_th']}")

    # Model Section
    print("\n" + "-" * 80)
    print(" 1. ผลลัพธ์จากโมเดล CNN (EfficientNet-B0)")
    print("-" * 80)
    if cnn["available"]:
        print(f"  คลาสที่ทำนาย (Top-1) : {cnn['predicted_class']} (ความมั่นใจ {cnn['confidence_pct']}%)")
        print("  คะแนนทุกคลาส:")
        for cname in CLASS_NAMES:
            pct = cnn["probabilities_pct"].get(cname, 0.0)
            bar = "█" * int(pct // 5)
            marker = "◄ (Top-1)" if cname == cnn["predicted_class"] else ""
            print(f"    {cname:<20}: {pct:>5.1f}% [{bar:<20}] {marker}")
    else:
        print(f"  [คำเตือน] {cnn.get('warning')}")

    # 6 Filters Section
    print("\n" + "-" * 80)
    print(" 2. ผลลัพธ์จากตัวกรองฟิสิกส์ทั้ง 6 (Physics-based Rule Filters)")
    print("-" * 80)
    print(f"  ตัวกรองคะแนนสูงสุด : {filters['top_filter_class']} ({filters['top_filter_score_pct']}%)")
    print(f"  ตัวกรองที่ผ่านเกณฑ์ : {', '.join(filters['passed_filters']) if filters['passed_filters'] else 'ไม่มีคลาสใดผ่านเกณฑ์'}\n")

    print(f"  {'คลาสตัวกรอง':<20} {'คะแนน (%)':<12} {'เกณฑ์ (%)':<12} {'สถานะ':<8} {'Progress'}")
    print(f"  {'-'*18:<20} {'-'*10:<12} {'-'*10:<12} {'-'*6:<8} {'-'*20}")
    for cname in CLASS_NAMES:
        det = filters["details"][cname]
        score_pct = det["score_pct"]
        th_pct = det["threshold_pct"]
        passed = det["passed"]
        status_str = "PASS" if passed else "FAIL"
        bar = "█" * int(score_pct // 5)
        top_mark = "★ Winner" if cname == filters["top_filter_class"] else ""
        print(f"  {cname:<20} {score_pct:>5.1f}%      {th_pct:>5.1f}%      [{status_str:<4}] {bar:<20} {top_mark}")

    # Physical Features Section
    print("\n" + "-" * 80)
    print(" 3. คุณลักษณะทางกายภาพสำคัญ (Key Astronomical Features)")
    print("-" * 80)
    print(f"  • Median FWHM     : {feats['fwhm_median']} px")
    print(f"  • Eccentricity    : {feats['eccentricity_median']} (ความรี 0=กลม, 1=รีมาก)")
    print(f"  • Star Count      : {feats['star_count']} ดวง")
    print(f"  • Circularity     : {feats['mean_circularity']}")
    print(f"  • Mean Hollowness : {feats['mean_hollowness']}")
    print(f"  • Saturated Ratio : {feats['saturated_ratio_pct']}%")
    print(f"  • Streak Lines    : {feats['streak_count']} เส้น")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Astronomy Image Quality: CNN + 6-Filter Pipeline")
    parser.add_argument("image", help="Path to astronomical image file")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX_PATH), help="Path to ONNX model file")
    parser.add_argument("--json", action="store_true", help="Output results strictly as JSON (for API integration)")
    args = parser.parse_args()

    try:
        res = run_pipeline(image_path=args.image, onnx_path=args.onnx)
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2))
        else:
            print(f"\n[ERROR] ประมวลผลภาพไม่สำเร็จ: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print_cli_report(res)


if __name__ == "__main__":
    main()
