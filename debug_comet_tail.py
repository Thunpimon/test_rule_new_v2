"""
debug_comet_tail.py
====================
วางไฟล์นี้ไว้โฟลเดอร์เดียวกับ astro_pipeline_v2.py แล้วรัน:
    python debug_comet_tail.py 2607027VZF_899.png

สคริปต์นี้ทำสิ่งเดียวกับ check_comet_tail_guarded() ในไฟล์หลักทุกขั้นตอน
เป๊ะๆ แต่ print ค่ากลางออกมาให้เห็นว่า track ไหน pass/fail และเพราะอะไร
เพื่อหาว่าอะไรบล็อก has_comet_tail ของภาพ 899/906 อยู่จริงๆ
"""

import sys

import cv2
import numpy as np

import astro_pipeline_v2 as ap


def debug_comet_tail(gray, fwhm, star_area, mean_hollowness, bg_med):
    print("--- Inputs ---")
    print(f"fwhm={fwhm:.2f}  star_area={star_area:.2f}  "
          f"mean_hollowness={mean_hollowness:.4f}  bg_med={bg_med:.2f}")

    # Anti-OOF guard
    if (fwhm > 28.0 and star_area > 120.0) or mean_hollowness > 0.04:
        print(">>> BLOCKED at Anti-OOF guard (fwhm>28 & star_area>120, or hollowness>0.04)")
        return False
    print("[OK] Anti-OOF guard passed")

    if bg_med < 25.0:
        print(">>> BLOCKED at dark-frame guard (bg_med < 25)")
        return False

    # Track 1: massive core saturation (>=254), slender vertical -> comet
    thresh_sat = (gray >= 254).astype(np.uint8)
    num_sat, _, stats_sat, _ = cv2.connectedComponentsWithStats(thresh_sat)
    print(f"\n[Track 1: >=254 saturated blobs] num_blobs={num_sat - 1}")
    if num_sat > 1:
        for i in range(1, num_sat):
            bw = stats_sat[i, cv2.CC_STAT_WIDTH]
            bh = stats_sat[i, cv2.CC_STAT_HEIGHT]
            area = stats_sat[i, cv2.CC_STAT_AREA]
            aspect = bh / max(bw, 1)
            print(f"  blob#{i}: w={bw} h={bh} area={area} aspect={aspect:.2f}")
            if bh >= 150 and aspect >= 3.8 and bh < 900:
                print("  >>> TRACK 1 MATCH (bh>=150, aspect>=3.8) -> return True")
                return True

    # Track 2: bright slender drift tail (>=195)
    thresh_b = (gray >= 195).astype(np.uint8)
    num_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(thresh_b)
    print(f"\n[Track 2: >=195 bright blobs] num_blobs={num_b - 1}")

    # There can be thousands of blobs (every star in the field lights up at 195+).
    # Filter to only sizeable ones (area>=40) and look at the biggest 10 first --
    # the star+tail we care about will be one of the largest, not a 1px noise speck.
    if num_b > 1:
        areas_idx = [(int(stats_b[i, cv2.CC_STAT_AREA]), i) for i in range(1, num_b)]
        areas_idx = [t for t in areas_idx if t[0] >= 40]
        areas_idx.sort(reverse=True)
        print(f"  ({len(areas_idx)} blobs with area>=40, showing top 10 by area)")
        for area, i in areas_idx[:10]:
            bw = stats_b[i, cv2.CC_STAT_WIDTH]
            bh = stats_b[i, cv2.CC_STAT_HEIGHT]
            aspect = bh / max(bw, 1)
            ys, xs = np.where(labels_b == i)
            y_mid = (ys.min() + ys.max()) / 2.0
            top_xs = xs[ys <= y_mid]
            bot_xs = xs[ys > y_mid]
            top_w = int(top_xs.max() - top_xs.min()) + 1 if top_xs.size else 0
            bot_w = int(bot_xs.max() - bot_xs.min()) + 1 if bot_xs.size else 0
            asymmetry = abs(top_w - bot_w) / max(top_w, bot_w, 1)
            print(f"  blob#{i}: w={bw} h={bh} area={area} aspect={aspect:.2f}  "
                  f"top_w={top_w} bot_w={bot_w} asymmetry={asymmetry:.3f}")
            if bh >= 115 and aspect >= 5.0 and bh < 300:
                print("  >>> TRACK 2 MATCH (bh>=115, aspect>=5.0) -> return True")
                return True

    # Guard: massive ROUND core saturation (>=600px area) blocks everything below
    if num_sat > 1:
        max_idx = int(np.argmax(stats_sat[1:, cv2.CC_STAT_AREA]) + 1)
        max_blob_area = int(stats_sat[max_idx, cv2.CC_STAT_AREA])
        bw = int(stats_sat[max_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats_sat[max_idx, cv2.CC_STAT_HEIGHT])
        aspect = bh / max(bw, 1)
        print(f"\nmax saturated(>=254) blob: area={max_blob_area} w={bw} h={bh} aspect={aspect:.2f}")

        if max_blob_area >= 600:
            # Measure head-tail asymmetry: does the top half of the blob have a
            # different width than the bottom half? A round saturated star has
            # top_width ~= bottom_width. A "big head + tapering tail" comet-style
            # blob has one half much wider than the other.
            labels_sat = cv2.connectedComponentsWithStats(thresh_sat)[1]
            ys, xs = np.where(labels_sat == max_idx)
            y_mid = (ys.min() + ys.max()) / 2.0
            top_xs = xs[ys <= y_mid]
            bot_xs = xs[ys > y_mid]
            top_w = int(top_xs.max() - top_xs.min()) + 1 if top_xs.size else 0
            bot_w = int(bot_xs.max() - bot_xs.min()) + 1 if bot_xs.size else 0
            asymmetry = abs(top_w - bot_w) / max(top_w, bot_w, 1)
            print(f"  top_half_width={top_w}  bottom_half_width={bot_w}  asymmetry={asymmetry:.3f}")
            print(">>> BLOCKED at massive-round-core guard (area >= 600) -> return False EARLY")
            print("    (this guard does NOT check aspect ratio - a big head + short tail")
            print("     combined into one >=600px blob gets rejected here even if it IS a comet shape)")
            return False

    # Track 3: per-column downward-streak scan
    h, w = gray.shape
    thresh = bg_med + 25.0
    col_sums = np.sum(gray.astype(np.float32), axis=0)
    spike_thresh = np.median(col_sums) + 12000
    spikes = np.where(col_sums > spike_thresh)[0]
    print(f"\n[Track 3: column-scan] spike_thresh={spike_thresh:.0f}  num_spike_columns={len(spikes)}")
    if len(spikes) == 0:
        print(">>> No column spikes found -> return False")
        return False

    streaks = []
    curr = [spikes[0]]
    for s in spikes[1:]:
        if s == curr[-1] + 1:
            curr.append(s)
        else:
            streaks.append(int(np.mean(curr)))
            curr = [s]
    streaks.append(int(np.mean(curr)))
    print(f"  streak column groups at x = {streaks}")

    down_streak_stars = []
    for x in streaks:
        col = gray[:, x]
        pk_y = int(np.argmax(col))
        peak_val = int(col[pk_y])
        print(f"\n  column x={x}: peak_y={pk_y} peak_val={peak_val}")
        if peak_val < 210 or pk_y > h * 0.75:
            print(f"    skip: peak_val<210={peak_val < 210}  pk_y>{h * 0.75:.0f}={pk_y > h * 0.75}")
            continue

        down_streak = 0
        for y in range(pk_y + 8, min(h - 5, pk_y + 250)):
            if col[y] >= thresh:
                down_streak += 1
            else:
                break

        up_streak = 0
        for y in range(pk_y - 8, max(5, pk_y - 250), -1):
            if col[y] >= thresh:
                up_streak += 1
            else:
                break

        need = 3.0 * max(up_streak, 4)
        print(f"    down_streak={down_streak}  up_streak={up_streak}  "
              f"needs down>=35 and down>={need:.0f}")
        if down_streak >= 35 and down_streak >= need:
            print("    >>> qualifies as down-streak star")
            down_streak_stars.append(x)

    print(f"\ndown_streak_stars = {down_streak_stars}")
    if len(down_streak_stars) >= 2 and (max(down_streak_stars) - min(down_streak_stars)) >= 60:
        print(">>> TRACK 3 MATCH (>=2 down-streak columns, spread>=60px) -> return True")
        return True

    print(">>> No track matched -> return False (has_comet_tail = False)")
    return False


def main():
    if len(sys.argv) < 2:
        print("usage: python debug_comet_tail.py <image.png>")
        return
    img_path = sys.argv[1]
    gray = ap.read_gray_image(img_path)
    feats = ap.extract_features_v2(gray)
    print(f"=== {img_path} ===")
    print(f"star_count={feats.star_count}  mean_aspect_ratio={feats.mean_aspect_ratio:.3f}  "
          f"elongated_star_ratio={feats.elongated_star_ratio:.3f}  "
          f"angle_consistency={feats.elongated_angle_consistency:.3f}\n")
    result = debug_comet_tail(
        gray,
        feats.median_fwhm,
        feats.mean_star_area,
        feats.mean_hollowness,
        feats.background_median,
    )
    print(f"\nFINAL has_comet_tail (this script) = {result}")
    print(f"pipeline reported has_comet_tail       = {feats.has_comet_tail}")


if __name__ == "__main__":
    main()
