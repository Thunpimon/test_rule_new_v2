import sys
sys.path.insert(0, '.')
import pickle
from scratch.test_candidate_4 import dataset, candidate_good_4
from astro_pipeline_v2 import (
    score_out_of_focus_v2, score_no_star_v2, score_satellite_v2,
    AstroFeaturesV2, clamp, score_high, score_low, weighted_mean
)

def candidate_te_unblock(f: AstroFeaturesV2) -> float:
    if f.star_count < 3 or f.elongated_star_count < 2:
        if f.faint_streak_evidence >= 0.40:
            return clamp(f.faint_streak_evidence)
        return 0.05

    if f.median_fwhm > 28.0 and f.sharpness < 15000.0 and f.mean_aspect_ratio < 1.40 and f.elongated_angle_consistency < 0.65:
        return 0.10

    # Donut rejection: Only apply if consistency is NOT high parallel drift
    if f.mean_aspect_ratio < 1.45 and f.elongated_angle_consistency < 0.65:
        if f.mean_hollowness > 0.04:
            return 0.05
        if f.mean_star_area > 115.0:
            return 0.08

    if (f.mean_hollowness > 0.08 or (f.mean_star_area > 200.0 and f.mean_aspect_ratio < 1.60)) and f.elongated_angle_consistency < 0.70:
        return 0.05

    shape_score = weighted_mean([
        (score_high(f.median_eccentricity, 0.42, 0.75), 1.8),
        (score_high(f.mean_aspect_ratio, 1.22, 1.55), 2.2),
        (score_high(f.elongated_star_ratio, 0.22, 0.55), 2.0),
        (score_high(f.max_aspect_ratio, 1.60, 4.0), 1.0),
    ])

    cons_score = score_high(f.elongated_angle_consistency, 0.22, 0.58)

    if f.elongated_star_count >= 20 and f.elongated_star_ratio >= 0.30 and f.elongated_angle_consistency >= 0.30 and f.mean_aspect_ratio >= 1.32:
        cons_score = max(cons_score, 0.65)
        shape_score = max(shape_score, 0.60)

    if f.mean_aspect_ratio > 1.45 and f.elongated_star_ratio > 0.30:
        cons_score = max(cons_score, 0.75)
        shape_score = max(shape_score, 0.70)

    score = weighted_mean([
        (shape_score, 2.2),
        (cons_score, 2.0),
    ])

    if f.mean_aspect_ratio >= 1.8 and f.elongated_angle_consistency >= 0.50:
        count_gate = score_high(f.elongated_star_count, 1, 4)
    else:
        count_gate = score_high(f.elongated_star_count, 2, 6)

    if f.elongated_angle_consistency < 0.20 and f.mean_aspect_ratio < 1.35:
        score *= 0.20

    is_field_drift = (f.elongated_star_ratio >= 0.40 and f.elongated_angle_consistency >= 0.75)
    if not is_field_drift and f.max_streak_aspect_ratio >= 14.0 and f.max_streak_length_ratio >= 0.14:
        score *= 0.30
    elif f.max_streak_length_ratio >= 0.28 and f.max_streak_aspect_ratio >= 14.0 and f.elongated_star_count < 10:
        score *= 0.15

    final_score = clamp(score * count_gate)

    has_unmistakable_drift = (
        f.has_extended_galaxy
        and (
            (f.elongated_star_count >= 15 and f.elongated_star_ratio >= 0.45 and f.elongated_angle_consistency >= 0.70)
            or (f.elongated_star_count >= 20 and f.elongated_star_ratio >= 0.35 and f.elongated_angle_consistency >= 0.70)
            or (f.elongated_star_count >= 12 and f.elongated_star_ratio >= 0.35 and f.elongated_angle_consistency >= 0.55 and f.mean_aspect_ratio >= 1.33)
        )
    )

    if f.has_comet_tail:
        final_score = max(final_score, 0.95)
    elif has_unmistakable_drift:
        final_score = max(final_score, 0.85)
    elif f.has_extended_galaxy and f.mean_aspect_ratio < 1.45 and f.median_eccentricity < 0.40:
        final_score = min(final_score, 0.35)
    elif not f.has_comet_tail and f.mean_aspect_ratio < 1.35 and f.median_eccentricity < 0.38 and f.elongated_angle_consistency < 0.40:
        final_score = min(final_score, 0.05)
    return final_score

def candidate_os_unblock(f: AstroFeaturesV2) -> float:
    if f.has_comet_tail:
        return 0.15
    track1 = 0.0
    if f.max_projection_diff >= 50.0:
        if f.max_streak_length_ratio >= 0.25 and f.max_streak_aspect_ratio >= 10.0:
            track1 = 0.10
        # Guard against TE: if parallel trails across stars exist
        elif (
            (f.elongated_star_count >= 5 and f.elongated_angle_consistency >= 0.45 and f.mean_aspect_ratio >= 1.48)
            or (f.elongated_star_count >= 4 and f.elongated_angle_consistency >= 0.70 and f.max_aspect_ratio >= 2.2)
        ):
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
    '03_Tracking_Error': candidate_te_unblock,
    '04_Over_Saturated': candidate_os_unblock,
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
    if item['name'] in ['2607027VZF_1731.png', '2607027VZF_1751.png', '2607027VZF_1999.png']:
        print(f"{item['name']} pred: {pred} | TE={scores['03_Tracking_Error']*100:.1f}% | OS={scores['04_Over_Saturated']*100:.1f}%")

print(f"Total: {correct}/300 (Baseline: 250/300)")
for k, v in sorted(per_class.items()):
    print(f"  {k}: {v}/50")
