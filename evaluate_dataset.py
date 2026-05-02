"""
Example: Evaluate a dataset using NucEval and save results to CSV.

This script shows how to loop through prediction files, auto-detect GT format
(ROI directories or label maps), and compute metrics using NucEval.
"""

import os
import numpy as np
import pandas as pd
import tifffile
import cv2
from mahotas import imread
from tqdm import tqdm
from nuceval import NucEval



# -------------------------------------------------
# Paths (update these to match your setup)
# -------------------------------------------------
pred_dir = r"C:\Users\amahbod\projects\fulbright\results\NuInsSeg_entire\hovernext\preds_hovernext_54.17\nuinsseg\hovernext\all"

# GT paths — the script checks both locations per sample:
#   1) ROI directory: gt_roi_dir / {base}_roiset / *.roi   (preferred)
#   2) Label map:     gt_label_dir / {base}.tif             (fallback)
gt_roi_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\Imagj_zipslll"
gt_label_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\label masks modify"

amb_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\mask binary_vague"
output_csv = r"C:\Users\amahbod\projects\fulbright\results\metrics_results.csv"

# -------------------------------------------------
# NucEval hyperparameters
# -------------------------------------------------
zone_width = 0
overlap_thresh_amb = 10
match_iou = 0.5


# -------------------------------------------------
# Collect prediction files
# -------------------------------------------------
pred_files = sorted([
    f for f in os.listdir(pred_dir)
    if f.endswith(".tiff") or f.endswith(".tif") or f.endswith(".npy")
])

results = []

# -------------------------------------------------
# Evaluate each image
# -------------------------------------------------
for name in tqdm(pred_files, desc="Evaluating"):

    pred_path = os.path.join(pred_dir, name)
    base = os.path.splitext(name)[0]

    # --- Load prediction ---
    if name.endswith(".npy"):
        pred = np.load(pred_path)
    else:
        pred = tifffile.imread(pred_path)

    pred = pred.astype(np.int32)
    shape = pred.shape

    # --- Auto-detect GT format ---
    roi_path = os.path.join(gt_roi_dir, base + "_roiset")
    gt_label_path = os.path.join(gt_label_dir, base + ".tif")
    gt_npy_path = os.path.join(gt_label_dir, base + ".npy")

    gt = None
    gt_format = None

    if os.path.isdir(roi_path):
        gt = roi_path          # pass directory path directly
        gt_format = "roi"
    elif os.path.exists(gt_label_path):
        gt = tifffile.imread(gt_label_path).astype(np.int32)
        if gt.shape != shape:
            print(f"Shape mismatch for {name}: pred {shape}, gt {gt.shape}")
            continue
        gt_format = "label_map"
    elif os.path.exists(gt_npy_path):
        gt = np.load(gt_npy_path).astype(np.int32)
        gt_format = "label_map"
    else:
        print(f"GT not found for: {base}")
        continue

    # --- Load ambiguous mask (optional) ---
    amb = None
    amb_path = os.path.join(amb_dir, base + ".png")
    if os.path.exists(amb_path):
        amb = cv2.imread(amb_path, cv2.IMREAD_GRAYSCALE)
        if amb.shape != shape:
            print(f"Shape mismatch (amb vs pred) for {name}")
            continue

    # --- Call NucEval ---
    scores = NucEval(
        gt, pred,
        amb=amb,
        normalized=True,
        zone_width=zone_width,
        overlap_thresh_amb=overlap_thresh_amb,
        match_iou=match_iou,
    )

    scores["image"] = name
    scores["gt_format"] = gt_format
    results.append(scores)


# -------------------------------------------------
# Save results to CSV
# -------------------------------------------------
df = pd.DataFrame(results)

# Reorder columns
cols = ["image", "gt_format", "dice", "aji", "dq", "sq", "pq", "num_nuclei"]
df = df[cols]

metrics = ["dice", "aji", "dq", "sq", "pq"]

# Simple (unweighted) mean
mean_row = {"image": "MEAN", "gt_format": ""}
for m in metrics:
    mean_row[m] = df[m].mean()
mean_row["num_nuclei"] = ""

# Weighted mean (normalized by number of GT nuclei)
total_nuclei = df["num_nuclei"].sum()
weighted_row = {"image": "WEIGHTED_MEAN", "gt_format": ""}
for m in metrics:
    if total_nuclei > 0:
        weighted_row[m] = (df[m] * df["num_nuclei"]).sum() / total_nuclei
    else:
        weighted_row[m] = 0.0
weighted_row["num_nuclei"] = total_nuclei

df = pd.concat([df, pd.DataFrame([mean_row, weighted_row])], ignore_index=True)
df.to_csv(output_csv, index=False)

print(f"\nSaved results to: {output_csv}")
print(f"\nSimple mean:   {mean_row}")
print(f"Weighted mean: {weighted_row}")

# Report GT format usage
fmt_counts = df[~df["image"].isin(["MEAN", "WEIGHTED_MEAN"])]["gt_format"].value_counts()
print("\nGT format usage:")
for fmt, count in fmt_counts.items():
    print(f"  {fmt}: {count} images")
