"""
Example: Evaluate a dataset using NucEval and save results to CSV.

This script loops through prediction items, auto-detects both the prediction
and the GT format (ROI-set directories or label maps), and computes metrics
using NucEval.

Predictions in pred_dir may be either:
  - label-map files : {base}.tif / .tiff / .npy
  - ROI-set folders : {base}_roiset/  (containing ImageJ .roi files)
Both are matched to GT by their shared {base} name.

It prints diagnostics up front so that, if nothing gets evaluated, you can
see immediately WHY (empty prediction list, base-name mismatch with GT, or
shape mismatch) instead of getting a cryptic pandas KeyError at the end.
"""

import os
import numpy as np
import pandas as pd
import tifffile
import cv2
from tqdm import tqdm
from nuceval import NucEval, _load_roi_masks


# -------------------------------------------------
# Paths (update these to match your setup)
# -------------------------------------------------
# NOTE: pred_dir must contain the PREDICTIONS, either as label maps
#       (.tif/.tiff/.npy) or as ROI-set folders named '{base}_roiset'.
#       If it points at the same folder as gt_roi_dir, predictions and GT
#       get confused. Keep them in separate folders.
pred_dir = r"examples/Prediction"   # replace with your prediction folder

# GT paths. The script checks both locations per sample:
#   1) ROI directory: gt_roi_dir / {base}_roiset / *.roi   (preferred)
#   2) Label map:     gt_label_dir / {base}.tif             (fallback)
gt_roi_dir = r"examples/GT"          # replace with your GT ROI folder (or None)
gt_label_dir = r"examples/GT"        # replace with your GT label map folder (or None)

amb_dir = r"examples/amb_masks"      # replace with your ambiguous mask folder (or None)
output_csv = r"metrics_results.csv"  # output file


# Fallback image shape (H, W) used ONLY when BOTH the prediction and the GT
# are ROI sets; neither carries image dimensions on its own, so ROIs need a
# canvas size to rasterize into. Set to your patch size (CryoNuSeg is 512x512).
# Set to None to instead skip such images with a warning.
default_shape = (512, 512)

# -------------------------------------------------
# NucEval hyperparameters
# -------------------------------------------------
zone_width = 1
overlap_thresh_amb = 0.25
match_iou = 0.5


# -------------------------------------------------
# Collect prediction items
# -------------------------------------------------
# A prediction can be either:
#   - a label-map file  : {base}.tif / .tiff / .npy
#   - an ROI-set folder : {base}_roiset/  (case-insensitive suffix, e.g.
#                         "{base}_RoiSet") containing ImageJ .roi files
# Each item is stored as (base, kind, path) where kind is "label_map" or "roi".
if not os.path.isdir(pred_dir):
    raise SystemExit(f"[ERROR] pred_dir does not exist:\n  {pred_dir}")

pred_items = []
for f in sorted(os.listdir(pred_dir)):
    full = os.path.join(pred_dir, f)
    if os.path.isfile(full) and f.lower().endswith((".tiff", ".tif", ".npy")):
        pred_items.append((os.path.splitext(f)[0], "label_map", full))
    elif os.path.isdir(full) and f.lower().endswith("_roiset"):
        # Strip the trailing "_roiset"/"_RoiSet"/etc. (any case) to get the base.
        base_name = f[:len(f) - len("_roiset")]
        pred_items.append((base_name, "roi", full))

# -------------------------------------------------
# Up-front diagnostics
# -------------------------------------------------
print(f"[INFO] pred_dir: {pred_dir}")
n_lm = sum(1 for _, k, _ in pred_items if k == "label_map")
n_roi = sum(1 for _, k, _ in pred_items if k == "roi")
print(f"[INFO] Found {len(pred_items)} prediction items "
      f"({n_lm} label maps, {n_roi} ROI-set folders).")

if len(pred_items) == 0:
    all_entries = os.listdir(pred_dir)
    exts = sorted({os.path.splitext(f)[1].lower() for f in all_entries if os.path.splitext(f)[1]})
    raise SystemExit(
        "[ERROR] No prediction items found in pred_dir.\n"
        f"        The folder contains {len(all_entries)} entries with extensions: {exts}\n"
        "        pred_dir must contain PREDICTION label maps (.tif/.tiff/.npy)\n"
        "        and/or ROI-set folders named '{base}_roiset'.\n"
        "        Check that pred_dir is not accidentally the same as gt_roi_dir."
    )

# Show, for the first prediction, exactly what GT paths are checked and whether
# they exist; this is where base-name mismatches reveal themselves.
_b = pred_items[0][0]
print(f"[INFO] First prediction base name: '{_b}' (kind: {pred_items[0][1]})")
print(f"       GT ROI dir : {os.path.join(gt_roi_dir, _b + '_roiset')} -> "
      f"{os.path.isdir(os.path.join(gt_roi_dir, _b + '_roiset'))}")
print(f"       GT tif     : {os.path.join(gt_label_dir, _b + '.tif')} -> "
      f"{os.path.exists(os.path.join(gt_label_dir, _b + '.tif'))}")
print(f"       GT npy     : {os.path.join(gt_label_dir, _b + '.npy')} -> "
      f"{os.path.exists(os.path.join(gt_label_dir, _b + '.npy'))}")
print()

results = []
skipped = {"gt_not_found": 0, "shape_mismatch": 0, "amb_mismatch": 0, "error": 0}

# -------------------------------------------------
# Evaluate each image
# -------------------------------------------------
for base, pred_kind, pred_path in tqdm(pred_items, desc="Evaluating"):

    name = os.path.basename(pred_path)

    # --- Load prediction (label map now; ROI dir resolved after we know shape) ---
    pred = None          # np.ndarray label map, OR
    pred_roi_dir = None  # str path to a prediction ROI-set folder
    if pred_kind == "label_map":
        try:
            if pred_path.lower().endswith(".npy"):
                pred = np.load(pred_path)
            else:
                pred = tifffile.imread(pred_path)
        except Exception as e:
            print(f"[WARN] Could not read prediction {name}: {e}")
            skipped["error"] += 1
            continue
        pred = pred.astype(np.int32)
        shape = pred.shape
    else:  # "roi": an ROI dir has no intrinsic shape; GT must provide it
        pred_roi_dir = pred_path
        shape = None

    # --- Auto-detect GT format ---
    roi_path      = os.path.join(gt_roi_dir, base + "_roiset")
    gt_label_path = os.path.join(gt_label_dir, base + ".tif")
    gt_npy_path   = os.path.join(gt_label_dir, base + ".npy")

    gt = None
    gt_format = None

    if os.path.isdir(roi_path):
        gt = roi_path                      # pass directory path directly
        gt_format = "roi"
    elif os.path.exists(gt_label_path):
        gt = tifffile.imread(gt_label_path).astype(np.int32)
        gt_format = "label_map"
    elif os.path.exists(gt_npy_path):
        gt = np.load(gt_npy_path).astype(np.int32)
        gt_format = "label_map"
    else:
        print(f"[WARN] GT not found for: {base}")
        skipped["gt_not_found"] += 1
        continue

    # --- Resolve the reference shape ---
    # If the prediction is a label map, shape came from it. Otherwise try the
    # GT (if it is an array). If BOTH sides are ROI sets, fall back to
    # default_shape so the ROIs can be rasterized onto a known canvas.
    if shape is None:
        if isinstance(gt, np.ndarray):
            shape = gt.shape
        elif default_shape is not None:
            shape = tuple(default_shape)
            print(f"[INFO] Both prediction and GT are ROI sets for {base}; "
                  f"using default_shape={shape}.")
        else:
            print(f"[WARN] Both prediction and GT are ROI dirs for {base} and "
                  f"default_shape is None; skipping. Set default_shape to your "
                  f"patch size (e.g. (512, 512)) to evaluate these.")
            skipped["error"] += 1
            continue

    # If GT is an ROI directory, load it into binary masks now, using the
    # resolved shape. (When GT is a label map it is already an array.)
    if gt_format == "roi":
        gt = _load_roi_masks(gt, shape)   # -> list of binary masks

    # --- Shape check for label-map GT vs label-map prediction ---
    if isinstance(gt, np.ndarray) and pred is not None and gt.shape != pred.shape:
        print(f"[WARN] Shape mismatch for {name}: pred {pred.shape}, gt {gt.shape}")
        skipped["shape_mismatch"] += 1
        continue

    # The value actually passed to NucEval as the prediction. For ROI-set
    # predictions, rasterize now using the resolved shape so both sides share
    # the same canvas (this also covers the both-ROI case).
    if pred is not None:
        pred_arg = pred
    else:
        pred_arg = _load_roi_masks(pred_roi_dir, shape)  # list of binary masks

    # --- Load ambiguous mask (optional) ---
    amb = None
    if amb_dir:
        amb_path = os.path.join(amb_dir, base + ".png")
        if os.path.exists(amb_path):
            amb = cv2.imread(amb_path, cv2.IMREAD_GRAYSCALE)
            if amb is None or amb.shape != shape:
                print(f"[WARN] Ambiguous mask shape mismatch (or unreadable) for {name}")
                skipped["amb_mismatch"] += 1
                continue

    # --- Call NucEval ---
    try:
        scores = NucEval(
            gt, pred_arg,
            amb=amb,
            normalized=True,
            zone_width=zone_width,
            overlap_thresh_amb=overlap_thresh_amb,
            match_iou=match_iou,
        )
    except Exception as e:
        print(f"[WARN] NucEval failed for {name}: {e}")
        skipped["error"] += 1
        continue

    scores["image"] = name
    scores["gt_format"] = gt_format
    scores["pred_format"] = pred_kind
    results.append(scores)


# -------------------------------------------------
# Guard: nothing evaluated
# -------------------------------------------------
if not results:
    raise SystemExit(
        "[ERROR] No images were evaluated; every prediction was skipped.\n"
        f"        Prediction files found : {len(pred_files)}\n"
        f"        Skipped (GT not found) : {skipped['gt_not_found']}\n"
        f"        Skipped (shape mismatch): {skipped['shape_mismatch']}\n"
        f"        Skipped (amb mismatch) : {skipped['amb_mismatch']}\n"
        f"        Skipped (read/eval err): {skipped['error']}\n"
        "        Most common cause: the prediction base name does not match the\n"
        "        GT file/dir name. See the '[INFO] First prediction base name'\n"
        "        block above and adjust the base -> GT name mapping accordingly."
    )


# -------------------------------------------------
# Save results to CSV
# -------------------------------------------------
df = pd.DataFrame(results)

cols = ["image", "gt_format", "pred_format", "dice", "aji", "dq", "sq", "pq", "num_nuclei"]
df = df[cols]

metrics = ["dice", "aji", "dq", "sq", "pq"]

# Simple (unweighted) mean
mean_row = {"image": "MEAN", "gt_format": "", "pred_format": ""}
for m in metrics:
    mean_row[m] = df[m].mean()
mean_row["num_nuclei"] = ""

# Weighted mean (normalized by number of GT nuclei)
total_nuclei = df["num_nuclei"].sum()
weighted_row = {"image": "WEIGHTED_MEAN", "gt_format": "", "pred_format": ""}
for m in metrics:
    if total_nuclei > 0:
        weighted_row[m] = (df[m] * df["num_nuclei"]).sum() / total_nuclei
    else:
        weighted_row[m] = 0.0
weighted_row["num_nuclei"] = total_nuclei

df = pd.concat([df, pd.DataFrame([mean_row, weighted_row])], ignore_index=True)

out_parent = os.path.dirname(output_csv)
if out_parent and not os.path.isdir(out_parent):
    os.makedirs(out_parent, exist_ok=True)
df.to_csv(output_csv, index=False)

print(f"\n[OK] Evaluated {len(results)} images. Skipped: {skipped}")
print(f"[OK] Saved results to: {output_csv}")
print(f"\nSimple mean:   {mean_row}")
print(f"Weighted mean: {weighted_row}")

# Report GT format usage
fmt_counts = df[~df["image"].isin(["MEAN", "WEIGHTED_MEAN"])]["gt_format"].value_counts()
print("\nGT format usage:")
for fmt, count in fmt_counts.items():
    print(f"  {fmt}: {count} images")
