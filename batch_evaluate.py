"""
batch_evaluate.py
=================
Batch Evaluation Script for Astronomy Quality Classification & 6-Filter Verification

Reads images across class subfolders (or a single folder), runs the unified pipeline
(CNN + 6 Filters), compares results against True Classes, prints summary metrics,
and exports detailed results into a CSV file.

Usage:
    # Evaluate entire Dataset_For_Rule_Base:
    python batch_evaluate.py

    # Quick test (5 images per class):
    python batch_evaluate.py --max-per-class 5

    # Specify custom dataset and output CSV:
    python batch_evaluate.py --dataset path/to/dataset --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import unified pipeline and class names
from astro_pipeline import CLASS_NAMES, DEFAULT_ONNX_PATH, run_pipeline

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def infer_true_class(file_path: Path) -> Optional[str]:
    """Infer true class from the parent directory name."""
    parent_name = file_path.parent.name
    for cname in CLASS_NAMES:
        if cname.lower() in parent_name.lower():
            return cname
    return None


def run_batch_evaluation(
    dataset_dir: str | Path,
    output_csv: str | Path = "batch_evaluation_results.csv",
    onnx_path: Optional[str | Path] = None,
    max_images_per_class: Optional[int] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Scan dataset folder, run full pipeline on every image, calculate accuracy,
    and save comparative results to CSV.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Gather images organized by class
    images_by_class: Dict[str, List[Path]] = defaultdict(list)
    loose_images: List[Path] = []

    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            matched_class = None
            for cname in CLASS_NAMES:
                if cname.lower() in item.name.lower():
                    matched_class = cname
                    break
            
            sub_images = [
                p for p in sorted(item.glob("*.*"))
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if matched_class:
                images_by_class[matched_class].extend(sub_images)
            else:
                loose_images.extend(sub_images)
        elif item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            loose_images.append(item)

    # Collect final list of (image_path, true_class)
    job_list: List[tuple[Path, Optional[str]]] = []
    for cname in CLASS_NAMES:
        imgs = images_by_class.get(cname, [])
        if max_images_per_class:
            imgs = imgs[:max_images_per_class]
        for img in imgs:
            job_list.append((img, cname))

    for img in loose_images:
        inferred = infer_true_class(img)
        job_list.append((img, inferred))

    total_images = len(job_list)
    if total_images == 0:
        print(f"[Warning] No valid images found in {dataset_path}")
        return []

    print("=" * 80)
    print(f" BATCH EVALUATION PIPELINE (CNN vs 6 FILTERS)")
    print(f" Dataset Path : {dataset_path.resolve()}")
    print(f" Total Images : {total_images} files")
    print(f" Output CSV   : {Path(output_csv).resolve()}")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    start_time = time.time()

    # Track metrics
    cnn_correct_count = 0
    filter_correct_count = 0
    both_correct_count = 0
    agreement_count = 0
    known_class_count = 0

    for idx, (img_path, true_cls) in enumerate(job_list, 1):
        try:
            res = run_pipeline(
                image_path=img_path,
                onnx_path=onnx_path,
                thresholds=thresholds,
            )
        except Exception as e:
            print(f"  [{idx}/{total_images}] ERROR on {img_path.name}: {e}")
            continue

        cnn_pred = res["cnn"].get("predicted_class")
        cnn_conf = res["cnn"].get("confidence_pct", 0.0)

        filter_top = res["filters"]["top_filter_class"]
        filter_top_score = res["filters"]["top_filter_score_pct"]

        is_agreed = res["agreement"]["is_matched"]
        status = res["agreement"]["status"]

        if is_agreed:
            agreement_count += 1

        cnn_is_correct = (true_cls == cnn_pred) if true_cls else None
        filter_is_correct = (true_cls == filter_top) if true_cls else None
        both_is_correct = (cnn_is_correct and filter_is_correct) if true_cls else None

        if true_cls:
            known_class_count += 1
            if cnn_is_correct:
                cnn_correct_count += 1
            if filter_is_correct:
                filter_correct_count += 1
            if both_is_correct:
                both_correct_count += 1

        # Print progress line
        cnn_mark = "✓" if cnn_is_correct else ("✗" if true_cls else "-")
        fil_mark = "✓" if filter_is_correct else ("✗" if true_cls else "-")
        agree_mark = "MATCH" if is_agreed else "DIFF"

        print(
            f"[{idx:>3}/{total_images}] {img_path.name:<24} | "
            f"True: {str(true_cls):<18} | "
            f"CNN: {str(cnn_pred):<18} [{cnn_mark}] | "
            f"Filter: {str(filter_top):<18} [{fil_mark}] | "
            f"{agree_mark}"
        )

        # Build CSV Row
        row: Dict[str, Any] = {
            "image_name": img_path.name,
            "image_path": str(img_path.resolve()),
            "true_class": true_cls or "UNKNOWN",
            "cnn_predicted_class": cnn_pred or "N/A",
            "cnn_confidence_pct": cnn_conf,
            "cnn_is_correct": cnn_is_correct if cnn_is_correct is not None else "N/A",
            "filter_top_class": filter_top,
            "filter_top_score_pct": filter_top_score,
            "filter_is_correct": filter_is_correct if filter_is_correct is not None else "N/A",
            "agreement_status": status,
            "is_matched": is_agreed,
            "both_correct": both_is_correct if both_is_correct is not None else "N/A",
        }

        # Add per-filter scores and passed status
        for cname in CLASS_NAMES:
            f_detail = res["filters"]["details"].get(cname, {})
            row[f"score_{cname}_pct"] = f_detail.get("score_pct", 0.0)
            row[f"passed_{cname}"] = f_detail.get("passed", False)

        # Add key physical features
        for k, v in res["key_physical_features"].items():
            row[f"feat_{k}"] = v

        csv_rows.append(row)
        results.append(res)

    elapsed = time.time() - start_time

    # Save to CSV
    if csv_rows:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(csv_rows[0].keys())
        with open(out_p, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[SAVED] บันทึกผลลัพธ์ลง CSV เรียบร้อย: {out_p.resolve()}")

    # Print Summary Report
    print("\n" + "=" * 80)
    print(" BATCH EVALUATION SUMMARY REPORT")
    print("=" * 80)
    print(f" Total Processed : {len(csv_rows)} images (in {elapsed:.2f}s, ~{elapsed/max(1, len(csv_rows)):.2f}s/img)")
    print(f" Agreement Rate  : {agreement_count}/{len(csv_rows)} ({agreement_count/max(1, len(csv_rows))*100:.1f}%)")

    if known_class_count > 0:
        cnn_acc = (cnn_correct_count / known_class_count) * 100
        fil_acc = (filter_correct_count / known_class_count) * 100
        both_acc = (both_correct_count / known_class_count) * 100

        print(f"\n[ Accuracy Comparison (บนภาพที่มี True Class: n={known_class_count}) ]")
        print(f"  • CNN Accuracy         : {cnn_correct_count}/{known_class_count} ({cnn_acc:.2f}%)")
        print(f"  • Filters Top Accuracy : {filter_correct_count}/{known_class_count} ({fil_acc:.2f}%)")
        print(f"  • Both Correct (Ensemble): {both_correct_count}/{known_class_count} ({both_acc:.2f}%)")
    print("=" * 80 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluation of CNN + 6 Filters on dataset")
    parser.add_argument(
        "--dataset",
        default=r"Dataset_For_Rule_Base",
        help="Path to folder containing class subdirectories or images",
    )
    parser.add_argument(
        "--output",
        default="batch_evaluation_results.csv",
        help="Output CSV filepath",
    )
    parser.add_argument(
        "--onnx",
        default=str(DEFAULT_ONNX_PATH),
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum images to process per class (for quick testing)",
    )
    args = parser.parse_args()

    run_batch_evaluation(
        dataset_dir=args.dataset,
        output_csv=args.output,
        onnx_path=args.onnx,
        max_images_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
