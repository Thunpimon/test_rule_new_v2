import sys
sys.path.insert(0, '.')
from scratch.test_candidate_2 import dataset, candidate_good_1, candidate_te_2, evaluate_candidate
from astro_pipeline_v2 import AstroFeaturesV2, weighted_mean, score_high, score_low, clamp

def candidate_te_4(f: AstroFeaturesV2) -> float:
    if f.star_count < 3 or f.elongated_star_count < 2:
        if f.faint_streak_evidence >= 0.40:
            return clamp(f.faint_streak_evidence)
        return 0.05

    if f.median_fwhm > 28.0 and f.sharpness < 15000.0 and f.mean_aspect_ratio < 1.40 and f.elongated_angle_consistency < 0.65:
        return 0.10

    if f.mean_aspect_ratio < 1.45:
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

def candidate_good_4(f: AstroFeaturesV2) -> float:
    if f.star_count < 2:
        return 0.05
    if (f.median_fwhm > 28.0 and f.sharpness < 15000.0) or f.mean_star_area > 115.0 or f.mean_hollowness > 0.05:
        return 0.10
    if f.median_fwhm > 33.0 and f.mean_hollowness > 0.035:
        return 0.10

    base = weighted_mean([
        (score_low(f.median_fwhm, 18.0, 35.0), 1.8),
        (score_low(f.median_eccentricity, 0.38, 0.58), 1.6),
        (score_high(f.mean_circularity, 0.52, 0.78), 1.2),
        (score_high(f.sharpness, 6000.0, 35000.0), 1.5),
        (score_high(f.star_count, 3, 25), 0.8),
        (score_low(f.saturated_ratio, 0.010, 0.040), 1.0),
        (score_low(f.mean_hollowness, 0.008, 0.04), 1.2),
    ])

    if f.star_count == 2:
        base *= 0.35

    if (f.median_fwhm <= 30.0 and f.mean_star_area <= 80.0 and f.mean_hollowness <= 0.02
            and f.median_eccentricity <= 0.40 and f.saturated_ratio <= 0.02):
        base = min(1.0, base * 1.18)

    te_penalty = 1.0
    has_unmistakable_drift = (
        f.has_extended_galaxy
        and (
            (f.elongated_star_count >= 15 and f.elongated_star_ratio >= 0.45 and f.elongated_angle_consistency >= 0.70)
            or (f.elongated_star_count >= 20 and f.elongated_star_ratio >= 0.35 and f.elongated_angle_consistency >= 0.70)
            or (f.elongated_star_count >= 12 and f.elongated_star_ratio >= 0.35 and f.elongated_angle_consistency >= 0.55 and f.mean_aspect_ratio >= 1.33)
        )
    )

    if f.has_comet_tail or has_unmistakable_drift:
        te_penalty = 0.20
    elif f.elongated_star_count >= 5 and f.elongated_star_ratio > 0.35 and f.elongated_angle_consistency > 0.45:
        if f.has_extended_galaxy and f.mean_aspect_ratio < 1.45 and f.median_eccentricity < 0.40:
            te_penalty = 1.0
        else:
            te_penalty = 0.25
    elif f.elongated_star_count >= 20 and f.elongated_star_ratio >= 0.30 and f.elongated_angle_consistency >= 0.30 and f.mean_aspect_ratio >= 1.32:
        if f.has_extended_galaxy and f.mean_aspect_ratio < 1.45 and f.median_eccentricity < 0.40:
            te_penalty = 1.0
        else:
            te_penalty = 0.25
    elif f.elongated_star_count >= 5 and f.mean_aspect_ratio > 1.48:
        te_penalty = 0.30

    sat_penalty = score_low(f.max_streak_length_ratio, 0.20, 0.55)
    if f.max_streak_length_ratio >= 0.18 and f.max_streak_aspect_ratio >= 5.0:
        sat_penalty = min(sat_penalty, 0.15)
    elif f.max_streak_length_ratio >= 0.12 and f.max_streak_aspect_ratio >= 10.0:
        sat_penalty = min(sat_penalty, 0.15)

    if f.max_projection_diff >= 50.0 or (f.max_val <= 165.0 and f.background_median <= 25.0 and f.star_count >= 10):
        base *= 0.15

    return clamp(base * te_penalty * sat_penalty)

print("--- Testing Candidate 4 ---")
evaluate_candidate(candidate_good_4, candidate_te_4)
