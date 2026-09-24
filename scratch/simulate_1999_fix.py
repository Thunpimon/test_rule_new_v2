import sys
sys.path.insert(0, '.')
import pickle
from scratch.test_candidate_4 import dataset, candidate_good_4, candidate_te_4
from astro_pipeline_v2 import (
    score_out_of_focus_v2, score_no_star_v2, score_satellite_v2,
    AstroFeaturesV2, clamp, score_high, score_low, weighted_mean
)

def candidate_os_test(f: AstroFeaturesV2) -> float:
    if f.has_comet_tail:
        return 0.15
    track1 = 0.0
    if f.max_projection_diff >= 50.0:
        if f.max_streak_length_ratio >= 0.25 and f.max_streak_aspect_ratio >= 10.0:
            track1 = 0.10
        # Relaxed from 1.50 to (aspect >= 1.40 or max_aspect >= 2.50)
        elif f.elongated_star_count >= 5 and f.elongated_angle_consistency >= 0.45 and (f.mean_aspect_ratio >= 1.40 or f.max_aspect_ratio >= 2.50):
            track1 = 0.10
        elif f.star_count <= 2 and f.bright_area_ratio < 0.001:
            track1 = 0.05
        else:
            track1 = max(score_high(f.max_projection_diff, 50.0, 75.0), 0.90)

    if f.mean_hollowness > 0.04 or (f.mean_star_area > 120.0 and f.median_fwhm > 30.0) or (f.median_fwhm > 28.0 and f.sharpness < 15000.0 and f.max_projection_diff < 45.0):
        return clamp(track1 if track1 > 0.0 else 0.05)

    track2 = 0.0
    if f.max_val <= 165.0 and f.star_count >= 10 and f.background_median <= 25.0:
        track2 = max(score_high(f.star_count, 10, 50) * score_low(f.max_val, 130.0, 165.0), 0.85)

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
    final = max(base, track1, track2, track3)
    return clamp(final)

rules = {
    '01_Good': candidate_good_4,
    '02_Out_of_Focus': score_out_of_focus_v2,
    '03_Tracking_Error': candidate_te_4,
    '04_Over_Saturated': candidate_os_test,
    '05_No_Star': score_no_star_v2,
    '06_Satellite': score_satellite_v2,
}

correct = 0
per_class = {}
changes = []
for item in dataset:
    tc = item['true_class']
    feats = item['feats']
    scores = {c: fn(feats) for c, fn in rules.items()}
    pred = max(scores, key=scores.get)
    if tc not in per_class: per_class[tc] = 0
    if pred == tc:
        correct += 1
        per_class[tc] += 1
    if item['name'] == '2607027VZF_1999.png':
        te_sc = round(scores['03_Tracking_Error'] * 100, 1)
        os_sc = round(scores['04_Over_Saturated'] * 100, 1)
        print(f"2607027VZF_1999.png pred: {pred} | TE={te_sc}% | OS={os_sc}%")

print(f"Total: {correct}/300")
for k, v in sorted(per_class.items()):
    print(f"  {k}: {v}/50")
