import sys
sys.path.insert(0, '.')
import pickle

with open(r'scratch\features_cache_300.pkl', 'rb') as fp:
    dataset = pickle.load(fp)

target_names = ['2607027VZF_1731.png', '2607027VZF_1751.png', '2607027VZF_1999.png', '2607027VZF_2219.png']

print("--- TARGET TRACKING ERROR IMAGES ---")
for item in dataset:
    if item['name'] in target_names:
        f = item['feats']
        n = item['name']
        print(f"{n}: nStars={f.star_count} | elong_cnt={f.elongated_star_count} ({f.elongated_star_ratio:.2f}) | cons={f.elongated_angle_consistency:.2f} | asp={f.mean_aspect_ratio:.2f} (max={f.max_aspect_ratio:.2f}) | ecc={f.median_eccentricity:.2f} | proj_diff={f.max_projection_diff:.1f} | sat_ratio={f.saturated_ratio:.4f} | fwhm={f.median_fwhm:.1f} | hollowness={f.mean_hollowness:.4f}")

print("\n--- SAMPLE 04_OVER_SATURATED IMAGES ---")
for item in dataset:
    if item['true_class'] == '04_Over_Saturated' and item['name'] in ['2605045QHM_1_2.png', '260702AT87_1_2.png', 'FILE_CURRENT_FOCUS13458 (67).png']:
        f = item['feats']
        n = item['name']
        print(f"{n}: nStars={f.star_count} | elong_cnt={f.elongated_star_count} ({f.elongated_star_ratio:.2f}) | cons={f.elongated_angle_consistency:.2f} | asp={f.mean_aspect_ratio:.2f} (max={f.max_aspect_ratio:.2f}) | ecc={f.median_eccentricity:.2f} | proj_diff={f.max_projection_diff:.1f} | sat_ratio={f.saturated_ratio:.4f} | fwhm={f.median_fwhm:.1f} | hollowness={f.mean_hollowness:.4f}")
