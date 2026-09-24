import sys
sys.path.insert(0, '.')
from scratch.simulate_tracking_tuning import dataset, candidate_good_1, candidate_te_1, evaluate_candidate
from astro_pipeline_v2 import AstroFeaturesV2, weighted_mean, score_high, score_low, clamp

def candidate_te_2(f: AstroFeaturesV2) -> float:
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

    # Count gate: relaxed if stars are clearly long parallel trails (mean aspect >= 1.8 and consistency >= 0.50)
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

print("--- Testing Candidate 2 (Count Gate relaxation for long trails) ---")
evaluate_candidate(candidate_good_1, candidate_te_2)
