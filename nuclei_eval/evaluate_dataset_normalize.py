from scipy.optimize import linear_sum_assignment
import scipy
import os
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
import cv2

from metrics import get_fast_aji, get_fast_pq, get_dice, remove_ambiguous_and_remap, remap_label




# -------------------------------------------------
# Paths
# -------------------------------------------------
pred_dir = r"C:\Users\amahbod\projects\fulbright\results\NuInsSeg_sub\cellvit\sub_NuInsSeg_data_cellvit_preds_29.07"
gt_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\label masks modify"
amb_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\mask binary_vague"
output_csv = r"C:\Users\amahbod\projects\fulbright\results\metrics_results.csv"
overlap_thresh_amb=0.25


# -------------------------------------------------
# Collect prediction files
# -------------------------------------------------
pred_files = sorted([
    f for f in os.listdir(pred_dir)
    if f.endswith(".tiff") or f.endswith(".npy")])



results = []


# -------------------------------------------------
# Loop through all samples
# -------------------------------------------------
total_stats = {
    'gt_original': 0,
    'pred_original': 0,
    'gt_removed_total': 0,
    'gt_removed_inside': 0,
    'gt_removed_border': 0,
    'pred_removed_total': 0,
    'pred_removed_inside': 0,
    'pred_removed_border': 0
}


for name in tqdm(pred_files, desc="Evaluating images"):

    pred_path = os.path.join(pred_dir, name)

    base = os.path.splitext(name)[0]
    gt_name = base + ".tif"
    gt_path = os.path.join(gt_dir, gt_name)

    if not os.path.exists(gt_path):
        print("GT not found:", gt_name)
        continue

    #pred = tifffile.imread(pred_path)
    if name.endswith(".npy"):
        pred = np.load(pred_path)
    elif name.endswith(".tiff") or name.endswith(".tif"):
        pred = tifffile.imread(pred_path)
    else:
        print("Unsupported format:", name)
        continue
    gt = tifffile.imread(gt_path)

    amb = None
    amb_path = os.path.join(amb_dir, base + ".png")

    if os.path.exists(amb_path):
        amb = cv2.imread(amb_path, cv2.IMREAD_GRAYSCALE)

        if amb.shape != gt.shape:
            print(f"Shape mismatch (amb vs gt) for {name}")
            continue

    if pred.shape != gt.shape:
        print(f"Shape mismatch for {name}: pred {pred.shape}, gt {gt.shape}")
        continue

    pred = pred.astype(np.int32)
    gt = gt.astype(np.int32)

    pred = remap_label(pred)
    gt = remap_label(gt)

    gt_clean, pred_clean, stats = remove_ambiguous_and_remap(gt, pred, amb, overlap_thresh=overlap_thresh_amb)
    for key in total_stats:
        total_stats[key] += stats[key]

    # dice = get_dice(gt, pred, amb)
    # aji = get_fast_aji(gt, pred, amb=amb)
    # pq_stats, _ = get_fast_pq(gt, pred, match_iou=0.5, amb=amb)

    dice = get_dice(gt_clean, pred_clean, amb)
    aji = get_fast_aji(gt_clean, pred_clean, amb=amb)
    pq_stats, _ = get_fast_pq(gt_clean, pred_clean, match_iou=0.5, amb=amb)


    dq, sq, pq = pq_stats

    # Count the number of nuclei in the cleaned GT (excluding background 0)
    num_nuclei = len(np.unique(gt_clean)) - 1  # subtract background
    num_nuclei = max(num_nuclei, 0)

    results.append({
        "image": name,
        "dice": dice,
        "aji": aji,
        "dq": dq,
        "sq": sq,
        "pq": pq,
        "num_nuclei": num_nuclei
    })


# -------------------------------------------------
# Save results
# -------------------------------------------------

df = pd.DataFrame(results)

metrics = ["dice", "aji", "dq", "sq", "pq"]

# --- Simple (unweighted) mean ---
mean_row = {"image": "MEAN"}
for m in metrics:
    mean_row[m] = df[m].mean()
mean_row["num_nuclei"] = ""

# --- Weighted mean (normalized by number of GT nuclei) ---
total_nuclei = df["num_nuclei"].sum()
weighted_row = {"image": "WEIGHTED_MEAN"}
for m in metrics:
    if total_nuclei > 0:
        weighted_row[m] = (df[m] * df["num_nuclei"]).sum() / total_nuclei
    else:
        weighted_row[m] = 0.0
weighted_row["num_nuclei"] = total_nuclei

df = pd.concat([df, pd.DataFrame([mean_row, weighted_row])], ignore_index=True)

df.to_csv(output_csv, index=False)

print("\nSaved results to:", output_csv)
print("\nDataset averages (simple mean):")
print(mean_row)
print("\nDataset averages (weighted by num_nuclei):")
print(weighted_row)

print("\n=== TOTAL STATS ===")
for k, v in total_stats.items():
    print(f"{k}: {v}")