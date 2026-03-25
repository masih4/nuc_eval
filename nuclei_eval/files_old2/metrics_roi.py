from scipy.optimize import linear_sum_assignment
import os
import struct
import numpy as np
import cv2


# =============================================================================
# ROI reading utilities
# =============================================================================

def read_imagej_roi(path):
    """
    Read a single ImageJ .roi file and return polygon coordinates
    as an Nx2 numpy array of (x, y) i.e. (col, row) points.
    Supports type 7 (freehand/polygon).
    """
    with open(path, 'rb') as f:
        data = f.read()

    # Verify magic bytes
    magic = data[0:4]
    if magic != b'Iout':
        raise ValueError(f"Not a valid ImageJ ROI file: {path}")

    roi_type = struct.unpack('>B', data[6:7])[0]
    top = struct.unpack('>h', data[8:10])[0]
    left = struct.unpack('>h', data[10:12])[0]
    n_coords = struct.unpack('>h', data[16:18])[0]

    # Coordinates start at byte offset 64
    offset = 64
    x_coords = []
    for i in range(n_coords):
        x = struct.unpack('>h', data[offset + i * 2: offset + i * 2 + 2])[0]
        x_coords.append(x + left)

    offset2 = 64 + n_coords * 2
    y_coords = []
    for i in range(n_coords):
        y = struct.unpack('>h', data[offset2 + i * 2: offset2 + i * 2 + 2])[0]
        y_coords.append(y + top)

    return np.array(list(zip(x_coords, y_coords)), dtype=np.int32)


def load_roi_masks(roi_dir, image_shape):
    """
    Load all .roi files from a directory and convert each to a binary mask.

    Args:
        roi_dir   : path to directory containing .roi files
        image_shape : (H, W) tuple for the output masks

    Returns:
        list of binary np.uint8 masks, one per ROI
    """
    roi_files = sorted([f for f in os.listdir(roi_dir) if f.endswith('.roi')])
    masks = []
    for rf in roi_files:
        pts = read_imagej_roi(os.path.join(roi_dir, rf))
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)
    return masks


# =============================================================================
# Ring (boundary uncertainty) mask generation
# =============================================================================

def build_ring_mask(gt_masks, ring_width=2):
    """
    Build a binary ring mask that marks the uncertain boundary zone
    around each GT ROI instance.

    For each GT mask:
        - Erode by ring_width pixels  -> confident interior
        - Dilate by ring_width pixels -> outer boundary
        - Ring = dilated minus eroded  (covers uncertainty in both directions)

    The final ring mask is the union of all per-instance rings.

    Args:
        gt_masks   : list of binary np.uint8 masks (one per ROI)
        ring_width : radius in pixels for erosion and dilation

    Returns:
        ring_mask : binary np.uint8 mask (H, W), 1 = uncertain ring zone
    """
    if len(gt_masks) == 0:
        return np.zeros(gt_masks[0].shape, dtype=np.uint8) if gt_masks else None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * ring_width + 1, 2 * ring_width + 1))
    ring_mask = np.zeros_like(gt_masks[0], dtype=np.uint8)

    for m in gt_masks:
        dilated = cv2.dilate(m, kernel, iterations=1)
        eroded = cv2.erode(m, kernel, iterations=1)
        ring = dilated - eroded  # pixels in dilated but not in eroded
        ring_mask = np.maximum(ring_mask, ring)

    return ring_mask


def apply_ring_mask(gt_masks, pred, ring_mask):
    """
    Zero out ring pixels from both GT masks and pred label map.

    Args:
        gt_masks  : list of binary np.uint8 masks
        pred      : instance label map (H, W) of np.int32
        ring_mask : binary mask (H, W), 1 = ring zone to exclude

    Returns:
        gt_masks_masked : list of GT masks with ring pixels zeroed
        pred_masked     : pred label map with ring pixels zeroed and remapped
    """
    ring_bool = ring_mask > 0

    gt_masks_masked = []
    for m in gt_masks:
        m_clean = m.copy()
        m_clean[ring_bool] = 0
        if m_clean.sum() > 0:
            gt_masks_masked.append(m_clean)

    pred_masked = np.copy(pred)
    pred_masked[ring_bool] = 0
    pred_masked = remap_label(pred_masked)

    return gt_masks_masked, pred_masked


# =============================================================================
# Metric functions (ROI-aware: GT is a list of binary masks)
# =============================================================================

def get_dice_roi(gt_masks, pred):
    """
    Dice coefficient between GT (list of binary masks) and pred (label map).

    GT foreground = union of all ROI masks (logical OR, no double-counting).
    Pred foreground = pred > 0.
    """
    # Build GT binary by logical OR across all masks
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


def get_fast_aji_roi(gt_masks, pred):
    """
    AJI (Aggregated Jaccard Index) with GT as a list of binary masks.

    Each GT mask is independent and can overlap with other GT masks.
    Pred is a standard label map (no overlaps).
    """
    pred = np.copy(pred)
    pred_id_list = list(np.unique(pred))

    n_gt = len(gt_masks)
    n_pred = len(pred_id_list) - 1  # exclude background 0

    # Both empty
    if n_gt == 0 and n_pred == 0:
        return 1.0

    # Prediction empty but GT not
    if n_pred == 0:
        return 0.0

    # Build pred binary masks
    pred_masks = []
    pred_ids = [pid for pid in pred_id_list if pid != 0]
    for p in pred_ids:
        pred_masks.append((pred == p).astype(np.uint8))

    # Pairwise intersection and union: gt_masks x pred_masks
    pairwise_inter = np.zeros((n_gt, n_pred), dtype=np.float64)
    pairwise_union = np.zeros((n_gt, n_pred), dtype=np.float64)

    for gi, gt_m in enumerate(gt_masks):
        # Find which pred IDs overlap with this GT mask
        pred_in_gt = pred[gt_m > 0]
        overlapping_pred_ids = np.unique(pred_in_gt)

        for pred_id in overlapping_pred_ids:
            if pred_id == 0:
                continue
            pi = pred_ids.index(pred_id)
            p_mask = pred_masks[pi]
            inter = (gt_m * p_mask).sum()
            total = (gt_m + p_mask).sum()
            pairwise_inter[gi, pi] = inter
            pairwise_union[gi, pi] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)

    # For each GT, find the pred with highest IoU (greedy, as in original AJI)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou_max = np.max(pairwise_iou, axis=1)

    # Only keep pairs with some intersection
    paired_gt = np.nonzero(pairwise_iou_max > 0.0)[0]
    paired_pred = paired_pred[paired_gt]

    overall_inter = pairwise_inter[paired_gt, paired_pred].sum()
    overall_union = pairwise_union[paired_gt, paired_pred].sum()

    # Add unpaired GT areas to union
    paired_gt_set = set(paired_gt.tolist())
    for gi in range(n_gt):
        if gi not in paired_gt_set:
            overall_union += gt_masks[gi].sum()

    # Add unpaired pred areas to union
    paired_pred_set = set(paired_pred.tolist())
    for pi in range(n_pred):
        if pi not in paired_pred_set:
            overall_union += pred_masks[pi].sum()

    aji_score = overall_inter / (overall_union + 1e-6)
    return aji_score


def get_fast_pq_roi(gt_masks, pred, match_iou=0.5):
    """
    Panoptic Quality with GT as a list of binary masks.

    Each GT mask is independent and can overlap with other GT masks.
    Pred is a standard label map (no overlaps).

    Returns:
        [dq, sq, pq], [paired_gt, paired_pred, unpaired_gt, unpaired_pred]
    """
    pred = np.copy(pred)
    pred_id_list = list(np.unique(pred))

    n_gt = len(gt_masks)
    n_pred = len(pred_id_list) - 1  # exclude background

    # Both empty
    if n_gt == 0 and n_pred == 0:
        return [1, 1, 1], [[], [], [], []]

    # Prediction empty but GT not
    if n_pred == 0:
        return [0, 0, 0], [0, 0, 0, 0]

    # Build pred binary masks
    pred_ids = [pid for pid in pred_id_list if pid != 0]
    pred_masks = []
    for p in pred_ids:
        pred_masks.append((pred == p).astype(np.uint8))

    # Pairwise IoU: gt_masks x pred_masks
    pairwise_iou = np.zeros((n_gt, n_pred), dtype=np.float64)

    for gi, gt_m in enumerate(gt_masks):
        pred_in_gt = pred[gt_m > 0]
        overlapping_pred_ids = np.unique(pred_in_gt)

        for pred_id in overlapping_pred_ids:
            if pred_id == 0:
                continue
            pi = pred_ids.index(pred_id)
            p_mask = pred_masks[pi]
            inter = (gt_m * p_mask).sum()
            total = (gt_m + p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[gi, pi] = iou

    # Matching
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou_copy = pairwise_iou.copy()
        pairwise_iou_copy[pairwise_iou_copy <= match_iou] = 0.0
        paired_gt, paired_pred = np.nonzero(pairwise_iou_copy)
        paired_iou = pairwise_iou_copy[paired_gt, paired_pred]
    else:
        # Hungarian matching
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
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_gt.tolist(), paired_pred.tolist(),
                                unpaired_gt, unpaired_pred]


# =============================================================================
# Ambiguous region handling for ROI masks
# =============================================================================

def remove_ambiguous_roi_masks(gt_masks, pred, amb, overlap_thresh=0.25):
    """
    Remove GT ROI masks and pred instances that overlap too much with
    the ambiguous region. Also zero out ambiguous pixels in pred.

    Args:
        gt_masks       : list of binary np.uint8 masks (one per GT instance)
        pred           : instance label map (H, W) of np.int32
        amb            : ambiguous mask (H, W), non-zero = ambiguous, or None
        overlap_thresh : fraction threshold for removal

    Returns:
        gt_masks_clean : filtered list of GT masks
        pred_clean     : cleaned pred label map (remapped)
        stats          : dict of removal statistics
    """
    pred = np.copy(pred)

    stats = {
        "gt_original": len(gt_masks),
        "pred_original": len(np.unique(pred)) - 1,
        "gt_removed_total": 0,
        "pred_removed_total": 0,
        "gt_removed_inside": 0,
        "pred_removed_inside": 0,
        "gt_removed_border": 0,
        "pred_removed_border": 0,
    }

    if amb is None:
        pred = remap_label(pred)
        return list(gt_masks), pred, stats

    amb_mask = amb > 0

    # --- Filter GT ROI masks ---
    gt_masks_clean = []
    for m in gt_masks:
        inst_area = m.sum()
        overlap_area = (m & amb_mask).sum()

        if overlap_area == 0:
            # Zero out ambiguous pixels from this mask too
            m_clean = m.copy()
            m_clean[amb_mask] = 0
            if m_clean.sum() > 0:
                gt_masks_clean.append(m_clean)
            continue

        overlap_ratio = overlap_area / (inst_area + 1e-6)

        if overlap_ratio > overlap_thresh:
            stats["gt_removed_total"] += 1
            if overlap_area == inst_area:
                stats["gt_removed_inside"] += 1
            else:
                stats["gt_removed_border"] += 1
        else:
            # Keep it but zero out ambiguous pixels
            m_clean = m.copy()
            m_clean[amb_mask] = 0
            if m_clean.sum() > 0:
                gt_masks_clean.append(m_clean)

    # --- Filter pred instances ---
    for inst_id in np.unique(pred):
        if inst_id == 0:
            continue
        inst_mask = (pred == inst_id)
        inst_area = inst_mask.sum()
        overlap_area = (inst_mask & amb_mask).sum()

        if overlap_area == 0:
            continue

        overlap_ratio = overlap_area / (inst_area + 1e-6)

        if overlap_ratio > overlap_thresh:
            stats["pred_removed_total"] += 1
            if overlap_area == inst_area:
                stats["pred_removed_inside"] += 1
            else:
                stats["pred_removed_border"] += 1
            pred[inst_mask] = 0

    # Zero out ambiguous pixels in pred
    pred[amb_mask] = 0
    pred = remap_label(pred)

    return gt_masks_clean, pred, stats


# =============================================================================
# Original helper (kept for pred side)
# =============================================================================

def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]."""
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred
