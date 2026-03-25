import os
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
import cv2

from metrics_roi import (
    load_roi_masks,
    get_dice_roi,
    get_fast_aji_roi,
    get_fast_pq_roi,
    remove_ambiguous_roi_masks,
    remap_label,
)


# -------------------------------------------------
# Paths
# -------------------------------------------------
pred_dir = r"C:\Users\amahbod\projects\fulbright\results\NuInsSeg_entire\cellvit\preds_cellvit_56.44\nuinsseg"
gt_roi_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\Imagj_zips"
# gt_roi_dir should contain subdirectories like:
#   human_bladder_01_roiset/
#       0026-0051.roi
#       0028-0196.roi
#       ...
#   human_bladder_02_roiset/
#       ...

amb_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\mask binary_vaguelll"
output_csv = r"C:\Users\amahbod\projects\fulbright\results\metrics_results_roi.csv"
overlap_thresh_amb = 0.25

# Image dimensions (needed to rasterize ROI polygons into masks).
# Set to None to auto-detect from the prediction file shape.
image_shape = None  # e.g. (512, 512) or None


# -------------------------------------------------
# Collect prediction files
# -------------------------------------------------
pred_files = sorted([
    f for f in os.listdir(pred_dir)
    if f.endswith(".tiff") or f.endswith(".tif") or f.endswith(".npy")
])

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
    'pred_removed_border': 0,
}

for name in tqdm(pred_files, desc="Evaluating images"):

    # --- Load prediction ---
    pred_path = os.path.join(pred_dir, name)
    base = os.path.splitext(name)[0]

    if name.endswith(".npy"):
        pred = np.load(pred_path)
    elif name.endswith(".tiff") or name.endswith(".tif"):
        pred = tifffile.imread(pred_path)
    else:
        print("Unsupported format:", name)
        continue

    pred = pred.astype(np.int32)
    shape = pred.shape  # (H, W)

    # --- Locate the ROI directory for this sample ---
    roi_subdir = base + "_roiset"
    roi_path = os.path.join(gt_roi_dir, roi_subdir)

    if not os.path.isdir(roi_path):
        print("ROI directory not found:", roi_subdir)
        continue

    # --- Load GT as list of binary masks from ROI files ---
    gt_masks = load_roi_masks(roi_path, shape)

    if len(gt_masks) == 0:
        print("No ROI files found in:", roi_subdir)
        continue

    # --- Load ambiguous mask (optional) ---
    amb = None
    amb_path = os.path.join(amb_dir, base + ".png")
    if os.path.exists(amb_path):
        amb = cv2.imread(amb_path, cv2.IMREAD_GRAYSCALE)
        if amb.shape != shape:
            print(f"Shape mismatch (amb vs pred) for {name}")
            continue

    # --- Remap pred labels ---
    pred = remap_label(pred)

    # --- Remove ambiguous instances ---
    gt_masks_clean, pred_clean, stats = remove_ambiguous_roi_masks(
        gt_masks, pred, amb, overlap_thresh=overlap_thresh_amb
    )
    for key in total_stats:
        total_stats[key] += stats[key]

    # --- Compute metrics ---
    dice = get_dice_roi(gt_masks_clean, pred_clean)
    aji = get_fast_aji_roi(gt_masks_clean, pred_clean)
    pq_stats, _ = get_fast_pq_roi(gt_masks_clean, pred_clean, match_iou=0.5)

    dq, sq, pq = pq_stats

    num_nuclei = len(gt_masks_clean)

    results.append({
        "image": name,
        "dice": dice,
        "aji": aji,
        "dq": dq,
        "sq": sq,
        "pq": pq,
        "num_nuclei": num_nuclei,
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
