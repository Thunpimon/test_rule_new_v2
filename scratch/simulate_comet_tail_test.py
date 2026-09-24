import sys
sys.path.insert(0, '.')
import pickle, cv2, numpy as np
from pathlib import Path
from astro_pipeline_v2 import (
    read_gray_image, extract_features_v2, score_rules_v2,
    score_good_v2, score_out_of_focus_v2, score_tracking_error_v2,
    score_over_saturated_v2, score_no_star_v2, score_satellite_v2
)

with open(r'scratch\features_cache_300.pkl', 'rb') as fp:
    dataset = pickle.load(fp)

def check_comet_tail_v3(gray, fwhm, star_area, max_hollowness, bg_med):
    # Guard against dark frame focusing spikes
    if bg_med < 25.0:
        return False

    # Check saturated blob shape: Only round saturated cores (asp < 1.60) are Over Sat
    thresh_sat = (gray >= 254).astype(np.uint8)
    num_sat, _, stats_sat, _ = cv2.connectedComponentsWithStats(thresh_sat)
    if num_sat > 1:
        areas = stats_sat[1:, cv2.CC_STAT_AREA]
        max_idx = int(np.argmax(areas) + 1)
        max_blob_area = stats_sat[max_idx, cv2.CC_STAT_AREA]
        w = stats_sat[max_idx, cv2.CC_STAT_WIDTH]
        h = stats_sat[max_idx, cv2.CC_STAT_HEIGHT]
        blob_asp = max(w, h) / max(min(w, h), 1)
        # If the largest saturated blob is a round core, it is NOT comet tail
        if max_blob_area >= 600 and blob_asp < 1.60:
            return False

    h, w = gray.shape
    thresh = bg_med + 25.0
    col_sums = np.sum(gray.astype(np.float32), axis=0)
    spikes = np.where(col_sums > np.median(col_sums) + 12000)[0]
    if len(spikes) == 0:
        return False

    streaks = []
    curr = [spikes[0]]
    for s in spikes[1:]:
        if s == curr[-1] + 1: curr.append(s)
        else:
            streaks.append(int(np.mean(curr)))
            curr = [s]
    streaks.append(int(np.mean(curr)))

    down_streak_stars = []
    for x in streaks:
        col = gray[:, x]
        pk_y = int(np.argmax(col))
        if col[pk_y] < 210:
            continue
        down_streak = 0
        for y in range(pk_y + 8, min(h - 5, pk_y + 250)):
            if col[y] >= thresh: down_streak += 1
            else: break
        up_streak = 0
        for y in range(pk_y - 8, max(5, pk_y - 250), -1):
            if col[y] >= thresh: up_streak += 1
            else: break
        if down_streak >= 35 and down_streak >= 2.5 * max(up_streak, 4):
            down_streak_stars.append(x)

    if len(down_streak_stars) >= 1:
        return True
    return False

print("Testing Comet Tail V3 on all 300 images...")
comet_detected = {}
for item in dataset:
    gray = read_gray_image(Path(item['path']))
    f = item['feats']
    res = check_comet_tail_v3(gray, f.median_fwhm, f.mean_star_area, f.mean_hollowness, f.background_median)
    tc = item['true_class']
    if res:
        if tc not in comet_detected: comet_detected[tc] = []
        comet_detected[tc].append(item['name'])

for tc, fnames in sorted(comet_detected.items()):
    print(f"  {tc} ({len(fnames)} images): {fnames}")
