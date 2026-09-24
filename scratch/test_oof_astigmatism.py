import sys
sys.path.insert(0, '.')
import pickle
from astro_pipeline_v2 import (
    score_good_v2, score_out_of_focus_v2, score_tracking_error_v2,
    score_over_saturated_v2, score_no_star_v2, score_satellite_v2,
    weighted_mean, score_high, score_low, clamp, AstroFeaturesV2
)

with open(r'scratch\features_cache_300.pkl', 'rb') as fp:
    dataset = pickle.load(fp)

def candidate_oof_astig(f: AstroFeaturesV2) -> float:
    if f.star_count < 2:
        return 0.02

    donut_evidence = 0.0
    if f.star_count >= 4 and f.mean_star_area >= 20.0:
        donut_evidence = score_high(f.mean_hollowness, 0.04, 0.12)

    donut_score = (
        weighted_mean([
            (donut_evidence, 2.5),
            (score_high(f.median_fwhm, 18.0, 35.0), 1.2),
            (score_low(f.median_eccentricity, 0.35, 0.60), 1.0),
        ])
        if donut_evidence > 0.0
        else 0.0
    )

    # Swollen Star: Allow aspect up to 1.65 if angles are random (astigmatic defocus, not parallel tracking drift)
    swollen_evidence = (
        max(score_high(f.mean_star_area, 80.0, 200.0), score_high(f.median_fwhm, 26.0, 38.0))
        if (f.mean_aspect_ratio < 1.50 or (f.mean_aspect_ratio < 1.65 and f.elongated_angle_consistency < 0.35))
        else 0.0
    )

    swollen_score = (
        weighted_mean([
            (swollen_evidence, 2.5),
            (score_high(f.median_fwhm, 26.0, 42.0), 2.0),
            (score_low(f.sharpness, 3000.0, 15000.0), 1.8),
            (score_high(f.median_hfr, 12.0, 22.0), 1.5),
            (score_low(f.median_eccentricity, 0.40, 0.75), 1.2),
            (score_low(f.saturated_ratio, 0.005, 0.030), 1.2),
        ])
        if swollen_evidence > 0.0
        else 0.0
    )

    if f.mean_aspect_ratio > 1.40 and f.elongated_angle_consistency > 0.65:
        swollen_score *= 0.20
        if donut_evidence < 0.50:
            donut_score *= 0.20

    final_score = max(donut_score, swollen_score)
    return clamp(final_score)

rules = {
    '01_Good': score_good_v2,
    '02_Out_of_Focus': candidate_oof_astig,
    '03_Tracking_Error': score_tracking_error_v2,
    '04_Over_Saturated': score_over_saturated_v2,
    '05_No_Star': score_no_star_v2,
    '06_Satellite': score_satellite_v2,
}

correct = 0
per_class = {}
changes = []
for item in dataset:
    tc = item['true_class']
    feats = item['feats']
    base_scores = {
        '01_Good': score_good_v2(feats),
        '02_Out_of_Focus': score_out_of_focus_v2(feats),
        '03_Tracking_Error': score_tracking_error_v2(feats),
        '04_Over_Saturated': score_over_saturated_v2(feats),
        '05_No_Star': score_no_star_v2(feats),
        '06_Satellite': score_satellite_v2(feats),
    }
    base_pred = max(base_scores, key=base_scores.get)
    new_scores = {c: fn(feats) for c, fn in rules.items()}
    new_pred = max(new_scores, key=new_scores.get)
    
    if tc not in per_class: per_class[tc] = {'base': 0, 'new': 0, 'total': 0}
    per_class[tc]['total'] += 1
    if base_pred == tc: per_class[tc]['base'] += 1
    if new_pred == tc:
        correct += 1
        per_class[tc]['new'] += 1
    if base_pred != new_pred:
        changes.append((item['name'], tc, base_pred, new_pred))

print(f"Total: {correct}/300 (Baseline: 250/300)")
print(f"Changes: {len(changes)}")
for c in changes:
    print(f"  {c[0]} ({c[1]}): {c[2]} -> {c[3]}")
for k, v in sorted(per_class.items()):
    print(f"  {k}: {v['new']}/{v['total']} (was {v['base']})")
