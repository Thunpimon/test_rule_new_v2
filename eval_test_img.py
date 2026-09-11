"""
eval_test_img.py
================
Evaluation and Benchmarking Script for 'test_img' (or any custom astronomy dataset).

Features:
  1. Automatically parses Ground Truth labels from filename prefixes:
     - G_       -> 01_Good
     - OOF_/OFF_-> 02_Out_of_Focus
     - TE_      -> 03_Tracking_Error
     - OS_      -> 04_Over_Saturated
     - NS_      -> 05_No_Star
     - ST_      -> 06_Satellite
     - Handles hybrid tags (OOF_TE_, TE_ST_)
  2. Runs astro_pipeline_v2 (extract_features_v2, score_rules_v2, CNN inference)
  3. Exports detailed per-image results to CSV
  4. Generates visual comparison plots (Accuracy bar chart, Confusion Matrices, Rule Scores Heatmap)
  5. Displays concise summary statistics in console

Usage:
  python eval_test_img.py
  python eval_test_img.py --input-dir test_img --output-csv test_img_results.csv --output-plot test_img_evaluation_summary.png
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from astro_pipeline_v2 import run_pipeline_v2, extract_features_v2, score_rules_v2, AstroFeaturesV2

CLASS_NAMES = [
    "01_Good",
    "02_Out_of_Focus",
    "03_Tracking_Error",
    "04_Over_Saturated",
    "05_No_Star",
    "06_Satellite",
]

SHORT_LABELS = {
    "01_Good": "Good",
    "02_Out_of_Focus": "OOF",
    "03_Tracking_Error": "TE",
    "04_Over_Saturated": "OverSat",
    "05_No_Star": "NoStar",
    "06_Satellite": "Satellite",
}


def parse_ground_truth(filename: str) -> Tuple[str, Optional[str]]:
    """Parse Ground Truth class and any hybrid note from filename."""
    name = Path(filename).name
    if name.startswith("OOF_TE_"):
        return "02_Out_of_Focus", "Hybrid (OOF + TE)"
    elif name.startswith("TE_ST_"):
        return "03_Tracking_Error", "Hybrid (TE + ST)"
    elif name.startswith("G_"):
        return "01_Good", None
    elif name.startswith("OOF_") or name.startswith("OFF_"):
        return "02_Out_of_Focus", None
    elif name.startswith("TE_"):
        return "03_Tracking_Error", None
    elif name.startswith("OS_"):
        return "04_Over_Saturated", None
    elif name.startswith("NS_"):
        return "05_No_Star", None
    elif name.startswith("ST_"):
        return "06_Satellite", None
    return "Unknown", None


def evaluate_folder(input_dir: str | Path) -> List[Dict]:
    """Run full evaluation on all images in folder."""
    input_path = Path(input_dir)
    image_files = sorted(
        [f for f in input_path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".fits"]]
    )
    if not image_files:
        raise FileNotFoundError(f"No valid image files found in {input_dir}")

    results = []
    print(f"\nProcessing {len(image_files)} images from: {input_path} ...")

    for idx, fpath in enumerate(image_files, 1):
        gt_class, hybrid_note = parse_ground_truth(fpath.name)
        pipeline_res = run_pipeline_v2(str(fpath))

        gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        feat = extract_features_v2(gray)
        filters = pipeline_res["filters"]
        details = filters["details"]
        cnn = pipeline_res.get("cnn", {})

        top_rule_class = filters["top_filter_class"]
        top_rule_score = filters["top_filter_score"] * 100.0
        cnn_class = cnn.get("predicted_class", "N/A")
        cnn_conf = cnn.get("confidence_pct", 0.0)

        rule_correct = (top_rule_class == gt_class) if gt_class != "Unknown" else None
        cnn_correct = (cnn_class == gt_class) if gt_class != "Unknown" else None

        row = {
            "filename": fpath.name,
            "ground_truth": gt_class,
            "hybrid_note": hybrid_note or "",
            "predicted_rule": top_rule_class,
            "rule_top_score_pct": round(top_rule_score, 2),
            "rule_correct": rule_correct,
            "score_01_Good_pct": round(details["01_Good"]["score_pct"], 2),
            "score_02_OOF_pct": round(details["02_Out_of_Focus"]["score_pct"], 2),
            "score_03_TE_pct": round(details["03_Tracking_Error"]["score_pct"], 2),
            "score_04_OverSat_pct": round(details["04_Over_Saturated"]["score_pct"], 2),
            "score_05_NoStar_pct": round(details["05_No_Star"]["score_pct"], 2),
            "score_06_Satellite_pct": round(details["06_Satellite"]["score_pct"], 2),
            "cnn_predicted": cnn_class,
            "cnn_confidence_pct": round(cnn_conf, 2),
            "cnn_correct": cnn_correct,
            # Physical features
            "star_count": feat.star_count,
            "median_fwhm": round(feat.median_fwhm, 2),
            "mean_fwhm": round(feat.mean_fwhm, 2),
            "median_eccentricity": round(feat.median_eccentricity, 3),
            "mean_aspect_ratio": round(feat.mean_aspect_ratio, 2),
            "mean_circularity": round(feat.mean_circularity, 3),
            "mean_hollowness": round(feat.mean_hollowness, 3),
            "elongated_star_count": feat.elongated_star_count,
            "elongated_angle_consistency": round(feat.elongated_angle_consistency, 3),
            "max_streak_length_ratio": round(feat.max_streak_length_ratio, 3),
            "max_streak_aspect_ratio": round(feat.max_streak_aspect_ratio, 2),
            "max_projection_diff": round(feat.max_projection_diff, 2),
            "saturated_ratio": round(feat.saturated_ratio, 4),
            "background_median": round(feat.background_median, 1),
            "background_mad": round(feat.background_mad, 2),
            "sharpness": round(feat.sharpness, 1),
        }
        results.append(row)
        if idx % 10 == 0 or idx == len(image_files):
            print(f"  [{idx:>2}/{len(image_files)}] Processed {fpath.name}")

    return results


def export_csv(results: List[Dict], output_csv_path: str | Path):
    """Save results dictionary list to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(output_csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] CSV report successfully saved to: {output_csv_path}")


def generate_plots(results: List[Dict], output_plot_path: str | Path):
    """Generate 4-panel comparison visualization figure."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.patch.set_facecolor("#fcfcfc")

    eval_rows = [r for r in results if r["ground_truth"] in CLASS_NAMES]
    classes = [c for c in CLASS_NAMES if any(r["ground_truth"] == c for r in eval_rows)]

    # 1. Per-Class Accuracy Comparison (Rule-Based vs CNN)
    ax1 = axes[0, 0]
    x = np.arange(len(classes))
    bar_w = 0.35

    rule_accs = []
    cnn_accs = []
    class_counts = []

    for c in classes:
        c_rows = [r for r in eval_rows if r["ground_truth"] == c]
        cnt = len(c_rows)
        class_counts.append(cnt)
        r_corr = sum(1 for r in c_rows if r["rule_correct"])
        c_corr = sum(1 for r in c_rows if r["cnn_correct"])
        rule_accs.append((r_corr / cnt) * 100.0 if cnt else 0.0)
        cnn_accs.append((c_corr / cnt) * 100.0 if cnt else 0.0)

    rects1 = ax1.bar(x - bar_w / 2, rule_accs, bar_w, label="Physics Rule-Based V2", color="#2563eb", alpha=0.90)
    rects2 = ax1.bar(x + bar_w / 2, cnn_accs, bar_w, label="CNN ONNX Model", color="#059669", alpha=0.90)

    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("1. Accuracy Comparison: Rule-Based V2 vs CNN Model", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{SHORT_LABELS[c]}\n(n={cnt})" for c, cnt in zip(classes, class_counts)], fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.axhline(75.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Target (75%)")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#065f46")

    # 2. Confusion Matrix: Rule-Based V2
    ax2 = axes[0, 1]
    cm_rule = np.zeros((len(classes), len(classes)), dtype=int)
    cls_idx = {c: i for i, c in enumerate(classes)}
    for r in eval_rows:
        gt_i = cls_idx[r["ground_truth"]]
        pred = r["predicted_rule"]
        pred_i = cls_idx.get(pred, -1)
        if pred_i >= 0:
            cm_rule[gt_i, pred_i] += 1

    im2 = ax2.imshow(cm_rule, interpolation="nearest", cmap="Blues")
    ax2.set_title("2. Confusion Matrix: Rule-Based V2 Predictions", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_xticklabels([SHORT_LABELS[c] for c in classes], fontsize=9)
    ax2.set_yticklabels([SHORT_LABELS[c] for c in classes], fontsize=9)
    ax2.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_rule[i, j]
            color = "white" if val > cm_rule.max() / 2 else "black"
            ax2.text(j, i, str(val), ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    # 3. Score Distribution Heatmap for Ground Truth vs Assigned Rule Scores
    ax3 = axes[1, 0]
    avg_scores = np.zeros((len(classes), len(classes)), dtype=float)
    score_cols = [
        "score_01_Good_pct", "score_02_OOF_pct", "score_03_TE_pct",
        "score_04_OverSat_pct", "score_05_NoStar_pct", "score_06_Satellite_pct"
    ]
    for i, gt_c in enumerate(classes):
        c_rows = [r for r in eval_rows if r["ground_truth"] == gt_c]
        if c_rows:
            for j, pred_c in enumerate(classes):
                col = score_cols[CLASS_NAMES.index(pred_c)]
                avg_scores[i, j] = np.mean([r[col] for r in c_rows])

    im3 = ax3.imshow(avg_scores, interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=100)
    ax3.set_title("3. Mean Rule Score (%) by Ground Truth Group", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xticks(np.arange(len(classes)))
    ax3.set_yticks(np.arange(len(classes)))
    ax3.set_xticklabels([f"Rule {SHORT_LABELS[c]}" for c in classes], fontsize=9)
    ax3.set_yticklabels([f"GT {SHORT_LABELS[c]}" for c in classes], fontsize=9)
    ax3.set_xlabel("Assigned Rule Score", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")

    for i in range(len(classes)):
        for j in range(len(classes)):
            v = avg_scores[i, j]
            color = "white" if v > 55 else "black"
            ax3.text(j, i, f"{v:.1f}%", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    # 4. Overall Summary Table / Metric Dashboard
    ax4 = axes[1, 1]
    ax4.axis("off")

    tot = len(eval_rows)
    tot_rule_corr = sum(1 for r in eval_rows if r["rule_correct"])
    tot_cnn_corr = sum(1 for r in eval_rows if r["cnn_correct"])
    agreement = sum(1 for r in eval_rows if r["predicted_rule"] == r["cnn_predicted"])

    summary_text = (
        f"DATASET BENCHMARK SUMMARY ({tot} IMAGES)\n"
        f"----------------------------------------------------\n"
        f"• Total Evaluated Images : {tot}\n"
        f"• Physics Rule-Based V2  : {tot_rule_corr}/{tot} ({tot_rule_corr/tot*100:.1f}%)\n"
        f"• CNN ONNX Deep Learning : {tot_cnn_corr}/{tot} ({tot_cnn_corr/tot*100:.1f}%)\n"
        f"• Rule & CNN Agreement   : {agreement}/{tot} ({agreement/tot*100:.1f}%)\n\n"
        f"PER-CLASS PERFORMANCE BREAKDOWN:\n"
    )
    for c, cnt, r_acc, c_acc in zip(classes, class_counts, rule_accs, cnn_accs):
        diff = r_acc - c_acc
        diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        status = "PASSED" if r_acc >= 70.0 else "TUNING"
        summary_text += f"  [{status:<6}] {SHORT_LABELS[c]:<10} (n={cnt:>2}) | Rule: {r_acc:>5.1f}% | CNN: {c_acc:>5.1f}% | Diff: {diff_str}\n"

    ax4.text(
        0.05, 0.95, summary_text,
        transform=ax4.transAxes,
        fontsize=10.5,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5)
    )

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Comparison plot successfully generated at: {output_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate astronomy image dataset with V2 rules.")
    parser.add_argument("--input-dir", type=str, default="test_img", help="Directory containing images to test.")
    parser.add_argument("--output-csv", type=str, default="test_img_results.csv", help="Path to output CSV.")
    parser.add_argument("--output-plot", type=str, default="test_img_evaluation_summary.png", help="Path to output plot PNG.")
    args = parser.parse_args()

    results = evaluate_folder(args.input_dir)
    export_csv(results, args.output_csv)
    generate_plots(results, args.output_plot)

    app_data_plot = Path(r"C:\Users\thanp\.gemini\antigravity\brain\3e239669-3415-4bcb-9bdf-a492e8d8712f") / Path(args.output_plot).name
    try:
        shutil.copy(args.output_plot, app_data_plot)
        print(f"[OK] Plot copied to artifact directory: {app_data_plot}")
    except Exception as e:
        pass

    tot = len(results)
    corr = sum(1 for r in results if r["rule_correct"])
    print(f"\n========================================================")
    print(f" FINAL BENCHMARK: {corr}/{tot} ({corr/tot*100:.2f}%) ON '{args.input_dir}'")
    print(f"========================================================")


if __name__ == "__main__":
    main()
