import pickle
from astro_pipeline_v2 import score_rules_v2

with open(r'scratch\features_cache_300.pkl', 'rb') as fp:
    dataset = pickle.load(fp)

print("=== 21 MISSES IN 03_TRACKING_ERROR ===")
for item in dataset:
    if item['true_class'] == '03_Tracking_Error':
        feats = item['feats']
        scores = score_rules_v2(feats)
        pred = max(scores, key=scores.get)
        if pred != '03_Tracking_Error':
            n = item['name']
            p_sc = round(scores[pred]*100, 1)
            t_sc = round(scores['03_Tracking_Error']*100, 1)
            print(f"{n}: pred={pred} ({p_sc}%) | te={t_sc}% | nStars={feats.star_count} | elong={feats.elongated_star_count} ({feats.elongated_star_ratio:.2f}) | cons={feats.elongated_angle_consistency:.2f} | asp={feats.mean_aspect_ratio:.2f} (max={feats.max_aspect_ratio:.2f}) | ecc={feats.median_eccentricity:.2f} | streak_asp={feats.max_streak_aspect_ratio:.1f} | streak_len={feats.max_streak_length_ratio:.2f} | gal={feats.has_extended_galaxy} | sat_ratio={feats.saturated_ratio:.4f}")
