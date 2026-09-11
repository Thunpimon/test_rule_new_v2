"""
export_mismatched_images.py
===========================
Exports mismatched images from test_img_results.csv into categorized folders.
"""

import os
import shutil
import pandas as pd
from pathlib import Path

def export_mismatches(csv_path="test_img_results.csv", src_dir="test_img", out_root="mismatches_test_img"):
    df = pd.read_csv(csv_path)
    src_path = Path(src_dir)
    out_path = Path(out_root)

    # 1. Rule vs Ground Truth mismatches (rule_correct == False)
    rule_gt_dir = out_path / "01_rule_vs_ground_truth"
    if rule_gt_dir.exists():
        shutil.rmtree(rule_gt_dir)
    rule_gt_dir.mkdir(parents=True, exist_ok=True)

    rule_gt_df = df[df["rule_correct"] == False].copy()
    print(f"Exporting {len(rule_gt_df)} images where Rule != Ground Truth...")

    summary_rule_gt = []
    for idx, row in rule_gt_df.iterrows():
        fname = row["filename"]
        gt = row["ground_truth"]
        pred = row["predicted_rule"]
        cnn = row["cnn_predicted"]
        
        subfolder_name = f"GT_{gt}__Pred_{pred}"
        dest_subfolder = rule_gt_dir / subfolder_name
        dest_subfolder.mkdir(parents=True, exist_ok=True)
        
        src_file = src_path / fname
        dest_file = dest_subfolder / fname
        if src_file.exists():
            shutil.copy2(src_file, dest_file)
            
        summary_rule_gt.append({
            "filename": fname,
            "ground_truth": gt,
            "predicted_rule": pred,
            "rule_top_score_pct": row["rule_top_score_pct"],
            "cnn_predicted": cnn,
            "cnn_confidence_pct": row["cnn_confidence_pct"],
            "hybrid_note": row["hybrid_note"],
            "subfolder": subfolder_name
        })

    pd.DataFrame(summary_rule_gt).to_csv(rule_gt_dir / "rule_vs_gt_mismatches.csv", index=False, encoding="utf-8-sig")

    # 2. Rule vs CNN disagreements (predicted_rule != cnn_predicted)
    rule_cnn_dir = out_path / "02_rule_vs_cnn_disagreement"
    if rule_cnn_dir.exists():
        shutil.rmtree(rule_cnn_dir)
    rule_cnn_dir.mkdir(parents=True, exist_ok=True)

    rule_cnn_df = df[df["predicted_rule"] != df["cnn_predicted"]].copy()
    print(f"Exporting {len(rule_cnn_df)} images where Rule != CNN...")

    summary_rule_cnn = []
    for idx, row in rule_cnn_df.iterrows():
        fname = row["filename"]
        gt = row["ground_truth"]
        pred = row["predicted_rule"]
        cnn = row["cnn_predicted"]

        subfolder_name = f"Rule_{pred}__CNN_{cnn}"
        dest_subfolder = rule_cnn_dir / subfolder_name
        dest_subfolder.mkdir(parents=True, exist_ok=True)

        src_file = src_path / fname
        dest_file = dest_subfolder / fname
        if src_file.exists():
            shutil.copy2(src_file, dest_file)

        summary_rule_cnn.append({
            "filename": fname,
            "ground_truth": gt,
            "predicted_rule": pred,
            "rule_top_score_pct": row["rule_top_score_pct"],
            "cnn_predicted": cnn,
            "cnn_confidence_pct": row["cnn_confidence_pct"],
            "rule_correct": row["rule_correct"],
            "cnn_correct": row["cnn_correct"],
            "subfolder": subfolder_name
        })

    pd.DataFrame(summary_rule_cnn).to_csv(rule_cnn_dir / "rule_vs_cnn_disagreements.csv", index=False, encoding="utf-8-sig")

    # 3. Create README.md
    readme_content = f"""# Mismatched Images Summary (from test_img)

This folder contains images that had prediction discrepancies during evaluation on `test_img` (50 images).

## Folder Structure

### 1. `01_rule_vs_ground_truth/` ({len(rule_gt_df)} images)
Images where the Physics Rule-Based Pipeline prediction did **NOT** match the Ground Truth label.
Grouped by category `GT_[TrueClass]__Pred_[PredictedClass]/`:
"""
    for sub, count in pd.DataFrame(summary_rule_gt)["subfolder"].value_counts().items():
        readme_content += f"- **`{sub}/`**: {count} image(s)\n"

    readme_content += f"""
### 2. `02_rule_vs_cnn_disagreement/` ({len(rule_cnn_df)} images)
Images where the Physics Rule-Based Pipeline and the CNN ONNX model disagreed with each other.
Grouped by category `Rule_[RulePrediction]__CNN_[CNNPrediction]/`:
"""
    for sub, count in pd.DataFrame(summary_rule_cnn)["subfolder"].value_counts().items():
        readme_content += f"- **`{sub}/`**: {count} image(s)\n"

    with open(out_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"\n[OK] All mismatched images exported to: {out_path.resolve()}")

if __name__ == "__main__":
    export_mismatches()
