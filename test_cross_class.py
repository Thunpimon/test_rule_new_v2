"""
test_cross_class.py
--------------------
ทดสอบ rule-based filter (astro_rule_classifier_new.py) แบบ "per-class threshold"
ตรงกับการใช้งานจริง (CNN ทายคลาสก่อน -> เอาเข้ากฎเฉพาะคลาสนั้นเพื่อยืนยัน) ไม่ใช่การให้
กฎทั้ง 6 คลาสแข่งกันหาคะแนนสูงสุด (argmax) แบบเดิม

สำหรับกฎของแต่ละคลาส c วัด 2 อย่าง:
  - Recall(c)  = ภาพคลาส c เอง กี่ % ที่ได้คะแนนกฎของคลาส c เกิน RULE_THRESHOLD (กฎจับตัวเองได้ดีแค่ไหน)
  - FPR(c)     = ภาพคลาส "อื่น" ทั้งหมด กี่ % ที่ดันได้คะแนนกฎของคลาส c เกิน RULE_THRESHOLD ไปด้วย
                 (กฎหลวมเกินไปจนรับรองภาพผิดคลาสไหม)

วิธีรัน:
    python test_cross_class.py

ก่อนรันต้องมี:
    pip install opencv-python numpy
"""

import sys
import csv
import statistics
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# ตั้งค่า
# ---------------------------------------------------------------------------

CLASSIFIER_DIR = Path(r"D:\Internship\Test_Rule\test_rule_new_V2")
DATASET_DIR = Path(r"D:\Internship\Test_Rule\test_rule_new_V2\Dataset_For_Rule_Base")

IMG_EXTS = {".png", ".jpg", ".jpeg"}
MAX_IMAGES_PER_CLASS = 50  # None = รันทุกภาพ (อาจช้าถ้ามีเยอะ)
RULE_THRESHOLD = 0.5  # ปรับตรงนี้ถ้าอยากลอง threshold อื่น (เช่น 0.4, 0.6)

# ---------------------------------------------------------------------------
sys.path.insert(0, str(CLASSIFIER_DIR))
try:
    from astro_rule_classifier_new import (
        CLASS_NAMES,
        read_gray_image,
        extract_features,
        score_rules,
    )
except ImportError as e:
    raise SystemExit(
        f"[error] import astro_rule_classifier_new ไม่สำเร็จ: {e}\n"
        f"ตรวจว่า CLASSIFIER_DIR ตั้งไว้ถูกไหม: {CLASSIFIER_DIR}\n"
        f"และไฟล์ astro_rule_classifier_new.py อยู่ในโฟลเดอร์นั้นจริง"
    )


def main() -> None:
    if not DATASET_DIR.exists():
        raise SystemExit(
            f"[error] ไม่พบโฟลเดอร์ dataset: {DATASET_DIR}\n"
            f"แก้ตัวแปร DATASET_DIR ในไฟล์นี้ให้ชี้ไปที่โฟลเดอร์ภาพจริง"
        )

    # pass_matrix[true_class][target_class] = จำนวนภาพของ true_class ที่ "ผ่าน" threshold
    # ของกฎ target_class (ไม่ใช่แค่ตัวที่คะแนนสูงสุดแบบ argmax เดิม)
    pass_matrix = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(int)

    recall_misses = []  # ภาพคลาสตัวเอง แต่คะแนนกฎตัวเองต่ำกว่า threshold
    false_positives = []  # ภาพคลาสอื่น แต่คะแนนกฎคลาส c สูงกว่า threshold

    feature_values = defaultdict(lambda: defaultdict(list))
    raw_rows = []

    for true_class in CLASS_NAMES:
        class_dir = DATASET_DIR / true_class
        if not class_dir.exists():
            print(f"[ข้าม] ไม่พบโฟลเดอร์: {class_dir}")
            continue

        paths = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
        if MAX_IMAGES_PER_CLASS:
            paths = paths[:MAX_IMAGES_PER_CLASS]

        print(f"[กำลังประมวลผล] {true_class}: {len(paths)} ภาพ")

        for img_path in paths:
            try:
                gray = read_gray_image(img_path)
                features = extract_features(gray)
                scores = score_rules(features)
            except Exception as e:
                print(f"  [error] {img_path.name}: {e}")
                continue

            class_counts[true_class] += 1
            feat_dict = features.__dict__
            for feat_name, feat_val in feat_dict.items():
                feature_values[true_class][feat_name].append(feat_val)
            raw_rows.append(
                {
                    "image": img_path.name,
                    "true_class": true_class,
                    **feat_dict,
                    **{f"score_{k}": v for k, v in scores.items()},
                }
            )

            for target_class, score in scores.items():
                passed = score >= RULE_THRESHOLD
                if passed:
                    pass_matrix[true_class][target_class] += 1

                if target_class == true_class and not passed:
                    recall_misses.append(
                        {"image": img_path.name, "true_class": true_class, "score": round(score, 3)}
                    )
                elif target_class != true_class and passed:
                    false_positives.append(
                        {
                            "image": img_path.name,
                            "true_class": true_class,
                            "rule_class": target_class,
                            "score": round(score, 3),
                        }
                    )

    # ---- ตาราง pass-rate เต็ม (แถว=คลาสจริง, คอลัมน์=กฎที่เอามาเช็ค, ค่า=จำนวนภาพที่ "ผ่าน" threshold) ----
    print(f"\n=== Pass matrix ที่ threshold={RULE_THRESHOLD} (แถว=คลาสจริง, คอลัมน์=ผ่านกฎคลาสไหนบ้าง) ===")
    header = f"{'True \\ Rule':<20}" + "".join(f"{c[3:16]:<14}" for c in CLASS_NAMES)
    print(header)
    for true_class in CLASS_NAMES:
        n = class_counts.get(true_class, 0)
        row = pass_matrix[true_class]
        line = f"{true_class:<20}" + "".join(f"{row.get(c, 0):<14}" for c in CLASS_NAMES)
        print(f"{line}  (n={n})")
    print("(แนวทแยง = Recall ของแต่ละคลาส, นอกแนวทแยง = False Positive ของคลาสคอลัมน์นั้นจากภาพคลาสแถวนั้น)")

    # ---- Recall / False Positive Rate สรุปต่อคลาส ----
    print(f"\n=== Recall / False Positive Rate ต่อคลาส (threshold={RULE_THRESHOLD}) ===")
    print(f"{'คลาส':<20}{'Recall':<18}{'False Positive Rate':<22}")
    total_n = sum(class_counts.values())
    for c in CLASS_NAMES:
        n_self = class_counts.get(c, 0)
        recall = pass_matrix[c].get(c, 0) / n_self if n_self else float("nan")

        n_others = total_n - n_self
        fp_count = sum(pass_matrix[other].get(c, 0) for other in CLASS_NAMES if other != c)
        fpr = fp_count / n_others if n_others else float("nan")

        print(f"{c:<20}{recall:.1%} ({pass_matrix[c].get(c,0)}/{n_self}){'':<3}{fpr:.1%} ({fp_count}/{n_others})")

    # ---- 10 อันดับ recall miss ที่คะแนนต่ำสุด ----
    recall_misses.sort(key=lambda m: m["score"])
    print(f"\n=== Recall miss ทั้งหมด {len(recall_misses)} รูป — 10 อันดับคะแนนต่ำสุด (กฎตัวเองควรจับได้แต่ไม่ผ่าน) ===")
    for m in recall_misses[:10]:
        print(f"  {m['image']:<28} คลาส={m['true_class']:<18} คะแนนกฎตัวเอง={m['score']}")

    # ---- 10 อันดับ false positive ที่คะแนนสูงสุด ----
    false_positives.sort(key=lambda m: m["score"], reverse=True)
    print(f"\n=== False positive ทั้งหมด {len(false_positives)} รูป — 10 อันดับคะแนนสูงสุด (กฎหลวมเกินไป) ===")
    for m in false_positives[:10]:
        print(
            f"  {m['image']:<28} คลาสจริง={m['true_class']:<18} ผ่านกฎของ={m['rule_class']:<18} คะแนน={m['score']}"
        )

    # ---- สถิติ min / median / max ต่อฟีเจอร์ แยกตามคลาสจริง (ไว้ช่วยตั้ง threshold) ----
    print("\n=== สถิติค่าฟีเจอร์ (min / median / max) แยกตามคลาสจริง ===")
    all_feature_names = []
    for cname in CLASS_NAMES:
        if feature_values.get(cname):
            all_feature_names = list(feature_values[cname].keys())
            break

    stats_rows = []
    for feat_name in all_feature_names:
        print(f"\n--- {feat_name} ---")
        for true_class in CLASS_NAMES:
            values = feature_values.get(true_class, {}).get(feat_name)
            if not values:
                continue
            v_min = min(values)
            v_median = statistics.median(values)
            v_max = max(values)
            print(f"  {true_class:<20} min={v_min!s:<14} median={v_median!s:<14} max={v_max!s:<14} (n={len(values)})")
            stats_rows.append(
                {"feature": feat_name, "true_class": true_class, "min": v_min, "median": v_median, "max": v_max, "n": len(values)}
            )

    # ---- export CSV ----
    raw_csv_path = CLASSIFIER_DIR / "feature_values_raw.csv"
    stats_csv_path = CLASSIFIER_DIR / "feature_stats_summary.csv"

    if raw_rows:
        with open(raw_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)
        print(f"\n[บันทึกแล้ว] ค่าฟีเจอร์ + คะแนนกฎทั้ง 6 คลาส รายภาพ -> {raw_csv_path}")

    if stats_rows:
        with open(stats_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stats_rows)
        print(f"[บันทึกแล้ว] สรุป min/median/max ต่อฟีเจอร์ต่อคลาส -> {stats_csv_path}")


if __name__ == "__main__":
    main()
