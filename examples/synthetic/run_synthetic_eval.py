"""
Evaluate the synthetic example with NucEval, showing the effect of each
modification in isolation (baseline, #1, #3, #4) and cumulatively.

Files expected (in the same folder as this script):
  gt_labelmap.tif                      GT label map (overlap merged), for #1 and #4
  Overlap-preserving GT/inst_*.tif     GT overlap-preserving masks, needed for #3
  pred_labelmap.tif                    prediction label map
  amb_mask.png                         ambiguous mask (single circle)

Run from the repository root:  python examples/synthetic/run_synthetic_eval.py

Key point:
  - Modification #3 (overlap handling) can ONLY be shown when the GT preserves
    overlaps, i.e. the LIST-OF-MASKS format. With the merged label map, the
    shared pixels belong to a single instance, so #3 has nothing to recover.
"""

import os
import sys
import glob
import numpy as np
import tifffile
import cv2

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))   # repo root, where nuceval.py is
from nuceval import NucEval

# -------------------------------------------------
# Load inputs
# -------------------------------------------------
gt_labelmap = tifffile.imread(os.path.join(here, "gt_labelmap.tif")).astype(np.int32)
pred        = tifffile.imread(os.path.join(here, "pred_labelmap.tif")).astype(np.int32)
amb         = cv2.imread(os.path.join(here, "amb_mask.png"), cv2.IMREAD_GRAYSCALE)

# Overlap-preserving GT: a list of binary masks, one per instance.
gt_masks = [tifffile.imread(f).astype(np.uint8)
            for f in sorted(glob.glob(os.path.join(here, "Overlap-preserving GT", "inst_*.tif")))]

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
overlap_thresh_amb = 0.25   # for Mod #1
zone_width_on = 1          # for Mod #4 (0 = off)
match_iou = 0.5


def show(tag, r):
    print(f"{tag:32s} "
          f"PQ={r['pq']*100:6.2f}  AJI={r['aji']*100:6.2f}  "
          f"Dice={r['dice']*100:6.2f}  DQ={r['dq']*100:6.2f}  SQ={r['sq']*100:6.2f}")


print("=" * 92)
print("Using MERGED label-map GT (cannot show Mod #3, overlaps are merged)")
print("=" * 92)

# Baseline: no modifications
show("Baseline",
     NucEval(gt_labelmap, pred, zone_width=0, match_iou=match_iou))

# Modification #1 only: exclude ambiguous region
show("Modification #1 (vague)",
     NucEval(gt_labelmap, pred, amb=amb,
             overlap_thresh_amb=overlap_thresh_amb,
             zone_width=0, match_iou=match_iou))

# Modification #4 only: boundary-uncertainty zone
show("Modification #4 (boundary)",
     NucEval(gt_labelmap, pred,
             zone_width=zone_width_on, match_iou=match_iou))


print()
print("=" * 92)
print("Using OVERLAP-PRESERVING GT (list of masks), enables Mod #3")
print("=" * 92)

# Baseline with overlap-preserving GT (this alone is the effect of Mod #3,
# since preserving overlaps is what #3 does relative to the merged map).
show("Baseline + Modification #3",
     NucEval(gt_masks, pred, zone_width=0, match_iou=match_iou))

# Modification #1 with overlap-preserving GT
show("Modification #1 + #3",
     NucEval(gt_masks, pred, amb=amb,
             overlap_thresh_amb=overlap_thresh_amb,
             zone_width=0, match_iou=match_iou))

# All modifications cumulatively (#1 + #3 + #4)
show("Modification #1 + #3 + #4",
     NucEval(gt_masks, pred, amb=amb,
             overlap_thresh_amb=overlap_thresh_amb,
             zone_width=zone_width_on, match_iou=match_iou))
