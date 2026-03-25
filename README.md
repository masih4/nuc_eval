<p align="center">
  <img src="nuceval_scientific_dark.png" alt="NucEval Workflow" width="100%">
</p>

<h1 align="center">NucEval</h1>

<p align="center">
  <b>Robust Evaluation Pipeline for Nuclei Instance Segmentation</b>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#examples">Examples</a> •
  <a href="#citation">Citation</a>
</p>

---

## Why NucEval?

Standard nuclei segmentation evaluation has several known issues that can lead to unfair or inaccurate metric scores:

| Problem | Impact | NucEval Solution |
|---------|--------|------------------|
| **Ambiguous regions** | Annotators flag uncertain areas, but metrics still penalize predictions there | Instances overlapping vague zones are excluded before scoring |
| **Overlapping annotations** | ROI-based GT allows overlapping nuclei, but label maps force one label per pixel — losing boundary information | Accepts ROI files directly as a list of independent binary masks |
| **Border uncertainty** | Exact nucleus boundaries are inherently uncertain by a few pixels | Boundary ring masks out uncertain pixels from both GT and prediction |
| **Score normalization** | Simple averaging treats images with 3 nuclei the same as images with 300 | Optional nuclei-weighted averaging across a dataset |

NucEval wraps all of these into **a single function call** with sensible defaults — so the simplest call gives you standard evaluation, and each feature can be enabled independently.

---

## Installation

```bash
pip install numpy scipy opencv-python
```

Then copy `nuceval.py` into your project. That's it — single file, no build step.

**Dependencies:** NumPy, SciPy, OpenCV (cv2)

---

## Quick Start

```python
from nuceval import NucEval

# Standard evaluation — identical to traditional metrics
result = NucEval(gt_label_map, pred_label_map)
print(result)
# {'dice': 0.92, 'aji': 0.85, 'dq': 0.90, 'sq': 0.88, 'pq': 0.79}
```

That's all you need for basic usage. Every other feature is opt-in.

---

## Features

### 1. Flexible GT formats

NucEval auto-detects the ground truth format:

```python
# Label map (numpy array, 0 = background)
result = NucEval(gt_label_map, pred)

# ROI directory (ImageJ/FIJI .roi files — handles overlapping instances)
result = NucEval("path/to/roiset/", pred)

# List of binary masks (from any source)
result = NucEval([mask1, mask2, mask3], pred)
```

### 2. Ambiguous region handling

Exclude nuclei that overlap with annotator-flagged uncertain regions:

```python
result = NucEval(gt, pred, amb=amb_mask)

# Custom overlap threshold (default: 0.25)
result = NucEval(gt, pred, amb=amb_mask, overlap_thresh_amb=0.5)
```

An instance is removed if more than `overlap_thresh_amb` fraction of its area falls in the ambiguous zone.

### 3. Boundary uncertainty ring

Account for annotation uncertainty at nucleus borders:

```python
result = NucEval(gt, pred, ring_width=2)
```

For each GT instance, the mask is eroded and dilated by `ring_width` pixels. The ring between the eroded core and the dilated boundary is masked out from **both** GT and prediction before scoring. This prevents penalizing predictions for disagreeing with GT in the inherently uncertain boundary zone.

<p align="center">
  <img src="ring_width_comparison.png" alt="Ring Width Comparison" width="90%">
</p>

### 4. Score normalization

Enable nuclei-weighted averaging across a dataset:

```python
result = NucEval(gt, pred, normalized=True)
# result now includes 'num_nuclei': 24
```

Use `num_nuclei` to compute weighted means, so images with more nuclei contribute proportionally more to the dataset score.

### 5. Configurable PQ matching

Adjust the IoU threshold for instance matching:

```python
# Standard (default)
result = NucEval(gt, pred, match_iou=0.5)

# Lenient matching
result = NucEval(gt, pred, match_iou=0.25)

# Strict matching
result = NucEval(gt, pred, match_iou=0.75)
```

### 6. Selective metrics

Compute only what you need:

```python
# Only Dice and PQ
result = NucEval(gt, pred, metrics=["dice", "pq"])
# {'dice': 0.92, 'pq': 0.79}

# Only instance-level metrics
result = NucEval(gt, pred, metrics=["aji", "dq", "sq", "pq"])
```

---

## API Reference

```python
NucEval(ground_truth, prediction,
        amb=None,                  # Ambiguous mask (H,W), non-zero = ambiguous
        normalized=False,          # Include num_nuclei in output
        ring_width=0,              # Boundary ring width in pixels (0 = disabled)
        overlap_thresh_amb=0.25,   # Fraction threshold for ambiguous removal
        match_iou=0.5,             # IoU threshold for PQ matching
        metrics=None)              # List of metrics, or None for all
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ground_truth` | `ndarray` / `list` / `str` | *required* | Label map, list of binary masks, or path to ROI directory |
| `prediction` | `ndarray` | *required* | Instance label map (H,W), 0 = background |
| `amb` | `ndarray` or `None` | `None` | Ambiguous region mask |
| `normalized` | `bool` | `False` | If True, include `num_nuclei` in output |
| `ring_width` | `int` | `0` | Boundary ring width (0 = no ring) |
| `overlap_thresh_amb` | `float` | `0.25` | Ambiguous overlap threshold |
| `match_iou` | `float` | `0.5` | PQ matching IoU threshold |
| `metrics` | `list` or `None` | `None` | Subset of `["dice", "aji", "dq", "sq", "pq"]` |

**Returns:** `dict` with requested metric scores.

---

## Examples

### Single image evaluation

```python
from nuceval import NucEval
import tifffile

gt = tifffile.imread("ground_truth.tif")
pred = tifffile.imread("prediction.tif")

# All features enabled
result = NucEval(gt, pred,
                 amb=amb_mask,
                 normalized=True,
                 ring_width=2,
                 match_iou=0.5)

print(f"Dice: {result['dice']:.4f}")
print(f"PQ:   {result['pq']:.4f}")
print(f"Nuclei: {result['num_nuclei']}")
```

### Dataset evaluation with CSV output

```python
import os
import numpy as np
import pandas as pd
import tifffile
import cv2
from tqdm import tqdm
from nuceval import NucEval

pred_dir = "path/to/predictions"
gt_dir = "path/to/ground_truth"       # .tif label maps
gt_roi_dir = "path/to/roi_sets"       # ROI directories (optional)
amb_dir = "path/to/ambiguous_masks"   # ambiguous masks (optional)

pred_files = sorted([f for f in os.listdir(pred_dir)
                     if f.endswith((".tif", ".tiff", ".npy"))])

results = []

for name in tqdm(pred_files):
    base = os.path.splitext(name)[0]

    # Load prediction
    if name.endswith(".npy"):
        pred = np.load(os.path.join(pred_dir, name))
    else:
        pred = tifffile.imread(os.path.join(pred_dir, name))

    # Auto-detect GT format
    roi_path = os.path.join(gt_roi_dir, base + "_roiset")
    gt_path = os.path.join(gt_dir, base + ".tif")

    if os.path.isdir(roi_path):
        gt = roi_path
    elif os.path.exists(gt_path):
        gt = tifffile.imread(gt_path)
    else:
        continue

    # Load ambiguous mask (optional)
    amb = None
    amb_path = os.path.join(amb_dir, base + ".png")
    if os.path.exists(amb_path):
        amb = cv2.imread(amb_path, cv2.IMREAD_GRAYSCALE)

    # Evaluate
    scores = NucEval(gt, pred, amb=amb, normalized=True, ring_width=2)
    scores["image"] = name
    results.append(scores)

# Save to CSV
df = pd.DataFrame(results)
metrics = ["dice", "aji", "dq", "sq", "pq"]

# Weighted mean
total = df["num_nuclei"].sum()
weighted = {m: (df[m] * df["num_nuclei"]).sum() / total for m in metrics}
print("Weighted mean:", weighted)

df.to_csv("results.csv", index=False)
```

---

## Metrics

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **Dice** | Pixel-level overlap (binary foreground) | 0–1 |
| **AJI** | Aggregated Jaccard Index across all instances | 0–1 |
| **DQ** | Detection Quality — F1 score of instance matching | 0–1 |
| **SQ** | Segmentation Quality — mean IoU of matched pairs | 0–1 |
| **PQ** | Panoptic Quality = DQ × SQ | 0–1 |

---

## How the boundary ring works

For each GT instance, the ring is computed per-instance then unioned into a global mask:

1. **Erode** the instance mask by `ring_width` pixels → confident core
2. **Dilate** the instance mask by `ring_width` pixels → outer boundary
3. **Ring** = dilated − eroded (covers uncertainty in both directions)
4. Union all per-instance rings into one mask
5. Zero out ring pixels from **both** GT and prediction
6. Compute metrics on the remaining pixels

---

## Citation

If you use NucEval in your research, please cite:

```bibtex
@software{nuceval,
  title={NucEval: Robust Evaluation Pipeline for Nuclei Instance Segmentation},
  author={},
  year={2025},
  url={https://github.com/}
}
```

---

## License

[MIT License](LICENSE)
