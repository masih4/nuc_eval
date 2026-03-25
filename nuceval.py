from scipy.optimize import linear_sum_assignment
import os
import struct
import numpy as np
import cv2


# =============================================================================
# Main API
# =============================================================================

def NucEval(ground_truth, prediction, amb=None, normalized=False,
            ring_width=0, overlap_thresh_amb=0.25, match_iou=0.5,
            metrics=None):
    """
    Unified evaluation function for nuclei instance segmentation.

    With default parameters, returns standard Dice, AJI, DQ, SQ, PQ scores
    — identical to traditional evaluation.

    Args:
        ground_truth : GT instance masks. Accepts:
                       - np.ndarray (H, W) : label map (int, 0 = background)
                       - list of np.ndarray : list of binary masks (one per instance),
                         supports overlapping instances (e.g., from ROI files)
                       - str : path to a directory containing .roi files

        prediction   : Predicted instance label map, np.ndarray (H, W) of int,
                       0 = background.

        amb          : Ambiguous region mask, np.ndarray (H, W), non-zero = ambiguous.
                       Instances overlapping too much with this mask are removed
                       before scoring. Default: None (no ambiguous handling).

        normalized   : If True, return num_nuclei along with metrics so that
                       weighted (normalized) averaging across images is possible.
                       Default: False.

        ring_width   : Boundary uncertainty ring width in pixels. For each GT
                       instance, a ring of this width is built by eroding and
                       dilating the mask. Pixels inside the ring are excluded
                       from both GT and prediction before scoring.
                       Set to 0 to disable. Default: 0.

        overlap_thresh_amb : Fraction threshold for ambiguous removal. An instance
                             is removed if (overlap_with_amb / instance_area) exceeds
                             this value. Default: 0.25.

        match_iou    : IoU threshold for PQ instance matching. A GT-prediction
                       pair counts as a true positive only if IoU > match_iou.
                       Standard value is 0.5. Lower values (e.g. 0.25) are more
                       lenient; higher values (e.g. 0.75) are stricter.
                       Default: 0.5.

        metrics      : List of metric names to compute. Choose from:
                       ["dice", "aji", "dq", "sq", "pq"].
                       Default: None (compute all metrics).

    Returns:
        dict with metric scores as float values.
        If normalized=True, also includes "num_nuclei" (int).

    Examples:
        # Simplest call — standard scores
        result = NucEval(gt_label_map, pred_label_map)

        # With ambiguous mask
        result = NucEval(gt, pred, amb=amb_mask)

        # With boundary ring
        result = NucEval(gt, pred, ring_width=2)

        # Custom IoU threshold for PQ matching
        result = NucEval(gt, pred, match_iou=0.25)

        # Only compute specific metrics
        result = NucEval(gt, pred, metrics=["dice", "pq"])

        # GT as ROI directory (handles overlaps)
        result = NucEval("path/to/roiset/", pred, ring_width=2)

        # GT as list of binary masks
        result = NucEval([mask1, mask2, mask3], pred)

        # Everything enabled
        result = NucEval(gt, pred, amb=amb_mask, normalized=True,
                         ring_width=2, match_iou=0.5)
    """

    # Default: compute all metrics
    all_metrics = ["dice", "aji", "dq", "sq", "pq"]
    if metrics is None:
        metrics = all_metrics
    else:
        for m in metrics:
            if m not in all_metrics:
                raise ValueError(f"Unknown metric '{m}'. Choose from {all_metrics}")

    # -------------------------------------------------
    # 1. Parse GT into list of binary masks
    # -------------------------------------------------
    if isinstance(ground_truth, str):
        if not os.path.isdir(ground_truth):
            raise ValueError(f"ROI directory not found: {ground_truth}")
        shape = prediction.shape
        gt_masks = _load_roi_masks(ground_truth, shape)
    elif isinstance(ground_truth, np.ndarray):
        gt_masks = _label_map_to_masks(ground_truth.astype(np.int32))
    elif isinstance(ground_truth, list):
        gt_masks = ground_truth
    else:
        raise TypeError("ground_truth must be np.ndarray, list of masks, or path to ROI dir")

    # -------------------------------------------------
    # 2. Prepare prediction
    # -------------------------------------------------
    pred = prediction.astype(np.int32)
    pred = _remap_label(pred)

    # -------------------------------------------------
    # 3. Remove ambiguous regions
    # -------------------------------------------------
    if amb is not None:
        gt_masks, pred, _ = _remove_ambiguous(
            gt_masks, pred, amb, overlap_thresh=overlap_thresh_amb
        )

    # -------------------------------------------------
    # 4. Apply boundary uncertainty ring
    # -------------------------------------------------
    if ring_width > 0:
        ring_mask = _build_ring_mask(gt_masks, ring_width=ring_width)
        if ring_mask is not None:
            gt_masks, pred = _apply_ring_mask(gt_masks, pred, ring_mask)

    # -------------------------------------------------
    # 5. Compute requested metrics
    # -------------------------------------------------
    results = {}

    if "dice" in metrics:
        results["dice"] = _get_dice(gt_masks, pred)

    if "aji" in metrics:
        results["aji"] = _get_aji(gt_masks, pred)

    need_pq = any(m in metrics for m in ["dq", "sq", "pq"])
    if need_pq:
        pq_stats, _ = _get_pq(gt_masks, pred, match_iou=match_iou)
        dq, sq, pq = pq_stats
        if "dq" in metrics:
            results["dq"] = dq
        if "sq" in metrics:
            results["sq"] = sq
        if "pq" in metrics:
            results["pq"] = pq

    if normalized:
        results["num_nuclei"] = len(gt_masks)

    return results


# =============================================================================
# Internal: ROI reading
# =============================================================================

def _read_imagej_roi(path):
    """Read a single ImageJ .roi file and return polygon coordinates."""
    with open(path, 'rb') as f:
        data = f.read()

    if data[0:4] != b'Iout':
        raise ValueError(f"Not a valid ImageJ ROI file: {path}")

    top = struct.unpack('>h', data[8:10])[0]
    left = struct.unpack('>h', data[10:12])[0]
    n_coords = struct.unpack('>h', data[16:18])[0]

    offset = 64
    x_coords = [struct.unpack('>h', data[offset + i*2: offset + i*2 + 2])[0] + left
                for i in range(n_coords)]

    offset2 = 64 + n_coords * 2
    y_coords = [struct.unpack('>h', data[offset2 + i*2: offset2 + i*2 + 2])[0] + top
                for i in range(n_coords)]

    return np.array(list(zip(x_coords, y_coords)), dtype=np.int32)


def _load_roi_masks(roi_dir, image_shape):
    """Load all .roi files from a directory as binary masks."""
    roi_files = sorted([f for f in os.listdir(roi_dir) if f.endswith('.roi')])
    masks = []
    for rf in roi_files:
        pts = _read_imagej_roi(os.path.join(roi_dir, rf))
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)
    return masks


def _label_map_to_masks(label_map):
    """Convert a label map to a list of binary masks."""
    masks = []
    for inst_id in np.unique(label_map):
        if inst_id == 0:
            continue
        masks.append((label_map == inst_id).astype(np.uint8))
    return masks


# =============================================================================
# Internal: Ring mask
# =============================================================================

def _build_ring_mask(gt_masks, ring_width=2):
    """Build boundary uncertainty ring mask from all GT instances."""
    if len(gt_masks) == 0:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * ring_width + 1, 2 * ring_width + 1))
    ring_mask = np.zeros_like(gt_masks[0], dtype=np.uint8)

    for m in gt_masks:
        dilated = cv2.dilate(m, kernel, iterations=1)
        eroded = cv2.erode(m, kernel, iterations=1)
        ring_mask = np.maximum(ring_mask, dilated - eroded)

    return ring_mask


def _apply_ring_mask(gt_masks, pred, ring_mask):
    """Zero out ring pixels from both GT masks and pred."""
    ring_bool = ring_mask > 0

    gt_out = []
    for m in gt_masks:
        m_clean = m.copy()
        m_clean[ring_bool] = 0
        if m_clean.sum() > 0:
            gt_out.append(m_clean)

    pred_out = np.copy(pred)
    pred_out[ring_bool] = 0
    pred_out = _remap_label(pred_out)

    return gt_out, pred_out


# =============================================================================
# Internal: Ambiguous region handling
# =============================================================================

def _remove_ambiguous(gt_masks, pred, amb, overlap_thresh=0.25):
    """Remove instances overlapping too much with ambiguous regions."""
    pred = np.copy(pred)

    stats = {
        "gt_original": len(gt_masks),
        "pred_original": len([p for p in np.unique(pred) if p != 0]),
        "gt_removed_total": 0, "pred_removed_total": 0,
        "gt_removed_inside": 0, "pred_removed_inside": 0,
        "gt_removed_border": 0, "pred_removed_border": 0,
    }

    if amb is None:
        return list(gt_masks), _remap_label(pred), stats

    amb_mask = amb > 0

    # Filter GT
    gt_clean = []
    for m in gt_masks:
        inst_area = m.sum()
        overlap_area = (m & amb_mask).sum()
        overlap_ratio = overlap_area / (inst_area + 1e-6)

        if overlap_ratio > overlap_thresh:
            stats["gt_removed_total"] += 1
            if overlap_area == inst_area:
                stats["gt_removed_inside"] += 1
            else:
                stats["gt_removed_border"] += 1
        else:
            m_clean = m.copy()
            m_clean[amb_mask] = 0
            if m_clean.sum() > 0:
                gt_clean.append(m_clean)

    # Filter pred
    for inst_id in np.unique(pred):
        if inst_id == 0:
            continue
        inst_mask = (pred == inst_id)
        inst_area = inst_mask.sum()
        overlap_area = (inst_mask & amb_mask).sum()
        overlap_ratio = overlap_area / (inst_area + 1e-6)

        if overlap_ratio > overlap_thresh:
            stats["pred_removed_total"] += 1
            if overlap_area == inst_area:
                stats["pred_removed_inside"] += 1
            else:
                stats["pred_removed_border"] += 1
            pred[inst_mask] = 0

    pred[amb_mask] = 0
    pred = _remap_label(pred)

    return gt_clean, pred, stats


# =============================================================================
# Internal: Metrics
# =============================================================================

def _get_dice(gt_masks, pred):
    """Dice coefficient. GT foreground = union of all masks."""
    if len(gt_masks) == 0:
        gt_bin = np.zeros(pred.shape, dtype=np.uint8)
    else:
        gt_bin = np.zeros_like(gt_masks[0], dtype=np.uint8)
        for m in gt_masks:
            gt_bin = np.maximum(gt_bin, m)

    pred_bin = (pred > 0).astype(np.uint8)
    inter = gt_bin * pred_bin
    denom = gt_bin.sum() + pred_bin.sum()

    if denom == 0:
        return 1.0
    return 2.0 * inter.sum() / (denom + 1e-6)


def _get_aji(gt_masks, pred):
    """Aggregated Jaccard Index."""
    pred = np.copy(pred)
    pred_ids = [pid for pid in np.unique(pred) if pid != 0]

    n_gt = len(gt_masks)
    n_pred = len(pred_ids)

    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0

    pred_masks = [(pred == p).astype(np.uint8) for p in pred_ids]

    pairwise_inter = np.zeros((n_gt, n_pred), dtype=np.float64)
    pairwise_union = np.zeros((n_gt, n_pred), dtype=np.float64)

    for gi, gt_m in enumerate(gt_masks):
        for pred_id in np.unique(pred[gt_m > 0]):
            if pred_id == 0:
                continue
            pi = pred_ids.index(pred_id)
            inter = (gt_m * pred_masks[pi]).sum()
            total = (gt_m + pred_masks[pi]).sum()
            pairwise_inter[gi, pi] = inter
            pairwise_union[gi, pi] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)

    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou_max = np.max(pairwise_iou, axis=1)
    paired_gt = np.nonzero(pairwise_iou_max > 0.0)[0]
    paired_pred = paired_pred[paired_gt]

    overall_inter = pairwise_inter[paired_gt, paired_pred].sum()
    overall_union = pairwise_union[paired_gt, paired_pred].sum()

    paired_gt_set = set(paired_gt.tolist())
    for gi in range(n_gt):
        if gi not in paired_gt_set:
            overall_union += gt_masks[gi].sum()

    paired_pred_set = set(paired_pred.tolist())
    for pi in range(n_pred):
        if pi not in paired_pred_set:
            overall_union += pred_masks[pi].sum()

    return overall_inter / (overall_union + 1e-6)


def _get_pq(gt_masks, pred, match_iou=0.5):
    """Panoptic Quality (DQ, SQ, PQ)."""
    pred = np.copy(pred)
    pred_ids = [pid for pid in np.unique(pred) if pid != 0]

    n_gt = len(gt_masks)
    n_pred = len(pred_ids)

    if n_gt == 0 and n_pred == 0:
        return [1, 1, 1], [[], [], [], []]
    if n_gt == 0:
        return [0, 0, 0], [[], [], [], list(range(n_pred))]
    if n_pred == 0:
        return [0, 0, 0], [[], [], list(range(n_gt)), []]

    pred_masks = [(pred == p).astype(np.uint8) for p in pred_ids]

    pairwise_iou = np.zeros((n_gt, n_pred), dtype=np.float64)

    for gi, gt_m in enumerate(gt_masks):
        for pred_id in np.unique(pred[gt_m > 0]):
            if pred_id == 0:
                continue
            pi = pred_ids.index(pred_id)
            inter = (gt_m * pred_masks[pi]).sum()
            total = (gt_m + pred_masks[pi]).sum()
            pairwise_iou[gi, pi] = inter / (total - inter)

    if match_iou >= 0.5:
        iou_copy = pairwise_iou.copy()
        iou_copy[iou_copy <= match_iou] = 0.0
        paired_gt, paired_pred = np.nonzero(iou_copy)
        paired_iou = iou_copy[paired_gt, paired_pred]
    else:
        paired_gt, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_gt, paired_pred]
        valid = paired_iou > match_iou
        paired_gt = paired_gt[valid]
        paired_pred = paired_pred[valid]
        paired_iou = paired_iou[valid]

    unpaired_gt = [i for i in range(n_gt) if i not in paired_gt]
    unpaired_pred = [i for i in range(n_pred) if i not in paired_pred]

    tp = len(paired_gt)
    fp = len(unpaired_pred)
    fn = len(unpaired_gt)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = paired_iou.sum() / (tp + 1e-6)

    return [dq, sq, dq * sq], [paired_gt.tolist(), paired_pred.tolist(),
                                unpaired_gt, unpaired_pred]


# =============================================================================
# Internal: Utilities
# =============================================================================

def _remap_label(pred, by_size=False):
    """Remap labels to contiguous IDs."""
    pred_id = [p for p in np.unique(pred) if p != 0]
    if len(pred_id) == 0:
        return pred
    if by_size:
        pred_id = sorted(pred_id, key=lambda x: (pred == x).sum(), reverse=True)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred
