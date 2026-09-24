import sys
sys.path.insert(0, '.')
import pickle, cv2, numpy as np
from pathlib import Path
from astro_pipeline_v2 import read_gray_image

with open(r'scratch\features_cache_300.pkl', 'rb') as fp:
    dataset = pickle.load(fp)

def classify_streak_type(gray):
    h, w = gray.shape
    bg_med = float(np.median(gray))
    thresh = bg_med + 25.0
    col_sums = np.sum(gray.astype(np.float32), axis=0)
    spikes = np.where(col_sums > np.median(col_sums) + 12000)[0]
    if len(spikes) == 0:
        return 0, 0
        
    streaks = []
    curr = [spikes[0]]
    for s in spikes[1:]:
        if s == curr[-1] + 1: curr.append(s)
        else:
            streaks.append(int(np.mean(curr)))
            curr = [s]
    streaks.append(int(np.mean(curr)))
    
    bilateral = 0
    unilateral = 0
    
    for x in streaks:
        col = gray[:, x]
        pk_y = int(np.argmax(col))
        if col[pk_y] < 210: continue
        
        down = 0
        for y in range(pk_y + 8, min(h - 5, pk_y + 250)):
            if col[y] >= thresh: down += 1
            else: break
            
        up = 0
        for y in range(pk_y - 8, max(5, pk_y - 250), -1):
            if col[y] >= thresh: up += 1
            else: break
            
        if down >= 25 and up >= 25:
            bilateral += 1
        elif (down >= 35 and down >= 2.5 * max(up, 4)) or (up >= 35 and up >= 2.5 * max(down, 4)):
            unilateral += 1
            
    return bilateral, unilateral

print("Evaluating 04_Over_Saturated images with projection_diff >= 50:")
for item in dataset:
    if item['true_class'] == '04_Over_Saturated' and item['feats'].max_projection_diff >= 50.0:
        gray = read_gray_image(Path(item['path']))
        b, u = classify_streak_type(gray)
        print(f"  {item['name']}: bilateral={b}, unilateral={u}")

print("\nEvaluating 03_Tracking_Error images with projection_diff >= 50:")
for item in dataset:
    if item['true_class'] == '03_Tracking_Error' and item['feats'].max_projection_diff >= 50.0:
        gray = read_gray_image(Path(item['path']))
        b, u = classify_streak_type(gray)
        print(f"  {item['name']}: bilateral={b}, unilateral={u}")
