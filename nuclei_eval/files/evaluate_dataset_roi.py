import os
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
import cv2

from metrics_roi import (
    load_roi_masks,
    label_map_to_masks,
    build_ring_mask,
    apply_ring_mask,
    get_dice_roi,
    get_fast_aji_roi,
    get_fast_pq_roi,
    remove_ambiguous_roi_masks,
    remap_label,
)


# -------------------------------------------------
# Paths
# -------------------------------------------------
pred_dir = r"C:\Users\amahbod\projects\fulbright\results\NuInsSeg_sub\hovernet\sub_NuInsSeg_data_hovernet_preds_28.68"

# GT paths — the script checks BOTH locations per sample:
#   1) ROI directory: gt_roi_dir / {base}_roiset / *.roi   (preferred, handles overlaps)
#   2) Label map:     gt_label_dir / {base}.tif             (fallback)
gt_roi_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\Imagj_zips"
# gt_roi_dir should contain subdirectories like:
#   human_bladder_01_roiset/
#       0026-0051.roi
#       0028-0196.roi
#       ...
#   human_bladder_02_roiset/
#       ...

gt_label_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\label masks modifylll"
amb_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\mask binary_vague"
output_csv = r"C:\Users\amahbod\projects\fulbright\results\metrics_results.csv"
overlap_thresh_amb = 0.25
ring_width = 0 # pixels to erode/dilate for boundary uncertainty ring (set to 0 to disable)


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

    # -------------------------------------------------
    # Auto-detect GT format: ROI directory or label map
    # -------------------------------------------------
    roi_subdir = base + "_roiset"
    roi_path = os.path.join(gt_roi_dir, roi_subdir)

    gt_label_name = base + ".tif"
    gt_label_path = os.path.join(gt_label_dir, gt_label_name)

    gt_masks = None
    gt_format = None

    # Priority 1: ROI files (handles overlaps properly)
    if os.path.isdir(roi_path):
        gt_masks = load_roi_masks(roi_path, shape)
        if len(gt_masks) > 0:
            gt_format = "roi"

    # Priority 2: Label map .tif (no overlap info, but ring still applies)
    if gt_masks is None or len(gt_masks) == 0:
        if os.path.exists(gt_label_path):
            gt_label = tifffile.imread(gt_label_path).astype(np.int32)

            if gt_label.shape != shape:
                print(f"Shape mismatch for {name}: pred {shape}, gt {gt_label.shape}")
                continue

            gt_label = remap_label(gt_label)
            gt_masks = label_map_to_masks(gt_label)
            gt_format = "label_map"

    if gt_masks is None or len(gt_masks) == 0:
        print(f"GT not found for: {base} (checked ROI dir and label map)")
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

    # --- Apply boundary uncertainty ring ---
    if ring_width > 0:
        ring_mask = build_ring_mask(gt_masks_clean, ring_width=ring_width)
        if ring_mask is not None:
            gt_masks_clean, pred_clean = apply_ring_mask(gt_masks_clean, pred_clean, ring_mask)

    # --- Compute metrics ---
    dice = get_dice_roi(gt_masks_clean, pred_clean)
    aji = get_fast_aji_roi(gt_masks_clean, pred_clean)
    pq_stats, _ = get_fast_pq_roi(gt_masks_clean, pred_clean, match_iou=0.5)

    dq, sq, pq = pq_stats

    num_nuclei = len(gt_masks_clean)

    results.append({
        "image": name,
        "gt_format": gt_format,
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
mean_row = {"image": "MEAN", "gt_format": ""}
for m in metrics:
    mean_row[m] = df[m].mean()
mean_row["num_nuclei"] = ""

# --- Weighted mean (normalized by number of GT nuclei) ---
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

print("\nSaved results to:", output_csv)
print("\nDataset averages (simple mean):")
print(mean_row)
print("\nDataset averages (weighted by num_nuclei):")
print(weighted_row)

print("\n=== TOTAL STATS ===")
for k, v in total_stats.items():
    print(f"{k}: {v}")

# --- Report GT format usage ---
if "gt_format" in df.columns:
    fmt_counts = df[~df["image"].isin(["MEAN", "WEIGHTED_MEAN"])]["gt_format"].value_counts()
    print("\n=== GT FORMAT USAGE ===")
    for fmt, count in fmt_counts.items():
        print(f"{fmt}: {count} images")
