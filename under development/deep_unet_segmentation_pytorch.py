from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.io import imsave
from skimage.morphology import label as sk_label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =========================
# Configuration
# =========================
@dataclass
class Config:
    base_path: str = "../datasets/NuInsSeg/"
    num_channels: int = 3
    threshold: float = 0.5
    epochs: int = 100
    quick_run: int = 1
    batch_size: int = 16
    random_seed: int = 19
    k_fold: int = 5
    save_val_results: bool = False
    init_lr: float = 1e-3
    lr_decay_factor: float = 0.5
    lr_drop_after_nth_epoch: int = 20
    crop_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    min_object_size: int = 50
    peak_footprint: int = 15
    result_save_path: str = "./prediction_image"
    model_save_path: str = "./output_model"
    metrics_csv_dir: str = "."
    use_amp: bool = True
    model_type: str = "shallow"  # shallow | deep


# =========================
# Reproducibility helpers
# =========================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# File discovery
# =========================
def discover_paths(base_path: str) -> Dict[str, List[str]]:
    img_path = sorted(glob(os.path.join(base_path, "*", "tissue images", "*.png")))
    binary_mask_path = sorted(glob(os.path.join(base_path, "*", "mask binary", "*.png")))
    distance_mask_path = sorted(glob(os.path.join(base_path, "*", "distance maps", "*.png")))
    label_mask_path = sorted(glob(os.path.join(base_path, "*", "label masks modify", "*.tif")))
    vague_mask_path = sorted(glob(os.path.join(base_path, "*", "vague areas", "mask binary_vague", "*.png")))

    counts = {
        "images": len(img_path),
        "binary_masks": len(binary_mask_path),
        "distance_masks": len(distance_mask_path),
        "label_masks": len(label_mask_path),
        "vague_masks": len(vague_mask_path),
    }
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Path count mismatch: {counts}")

    return {
        "img": img_path,
        "binary": binary_mask_path,
        "distance": distance_mask_path,
        "label": label_mask_path,
        "vague": vague_mask_path,
    }


# =========================
# Losses and metrics
# =========================
def dice_coef_torch(y_true: torch.Tensor, y_pred_prob: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    y_true = y_true.contiguous().view(y_true.size(0), -1)
    y_pred_prob = y_pred_prob.contiguous().view(y_pred_prob.size(0), -1)
    intersection = (y_true * y_pred_prob).sum(dim=1)
    denom = y_true.sum(dim=1) + y_pred_prob.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice.mean()


def bce_dice_loss_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y_true)
    probs = torch.sigmoid(logits)
    dice = dice_coef_torch(y_true, probs)
    return 0.5 * bce - dice


def get_fast_aji(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    if len(pred_id_list) == 1:
        return 0.0

    true_masks = [None]
    for t in true_id_list[1:]:
        true_masks.append(np.array(true == t, np.uint8))

    pred_masks = [None]
    for p in pred_id_list[1:]:
        pred_masks.append(np.array(pred == p, np.uint8))

    pairwise_inter = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(np.unique(pred_true_overlap))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]

    overall_inter = pairwise_inter[paired_true, paired_pred].sum()
    overall_union = pairwise_union[paired_true, paired_pred].sum()

    paired_true = list(paired_true + 1)
    paired_pred = list(paired_pred + 1)
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])

    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    return float(overall_inter / (overall_union + 1.0e-6))


def get_fast_pq(true: np.ndarray, pred: np.ndarray, match_iou: float = 0.5):
    assert match_iou >= 0.0

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    if len(pred_id_list) == 1:
        return [0.0, 0.0, 0.0], [0, 0, 0, 0]

    true_masks = [None]
    for t in true_id_list[1:]:
        true_masks.append(np.array(true == t, np.uint8))

    pred_masks = [None]
    for p in pred_id_list[1:]:
        pred_masks.append(np.array(pred == p, np.uint8))

    pairwise_iou = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(np.unique(pred_true_overlap))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter + 1.0e-6)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    if match_iou >= 0.5:
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1
        paired_pred += 1
    else:
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)
    sq = paired_iou.sum() / (tp + 1.0e-6)
    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def get_dice_1(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    dice_score = 2.0 * np.sum(inter) / (np.sum(denom) + 1.0e-4)
    if np.sum(inter) == 0 and np.sum(denom) == 0:
        dice_score = 1.0
    return float(dice_score)


def remap_label(pred: np.ndarray, by_size: bool = False) -> np.ndarray:
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred

    if by_size:
        pred_size = []
        for inst_id in pred_id:
            pred_size.append((pred == inst_id).sum())
        pair_list = sorted(zip(pred_id, pred_size), key=lambda x: x[1], reverse=True)
        pred_id, _ = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


# =========================
# Data pipeline
# =========================
def get_id_from_file_path(file_path: str, indicator: str) -> str:
    return file_path.split(os.path.sep)[-1].replace(indicator, "")


class NuInsSegDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[str],
        mask_paths: Sequence[str],
        crop_size: int,
        augment: bool = False,
        distance_unet_flag: bool = False,
    ) -> None:
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.crop_size = crop_size
        self.augment = augment
        self.distance_unet_flag = distance_unet_flag
        self.transform = self._build_transform(crop_size) if augment else None

    @staticmethod
    def _build_transform(crop_size: int):
        return A.Compose(
            [
                A.RandomCrop(crop_size, crop_size, p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    brightness_by_max=True,
                    p=0.4,
                ),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=20,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.1,
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(self.mask_paths[idx])

        if not self.distance_unet_flag:
            mask = (mask == 255).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1.0e-7)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)


# =========================
# Model definition
# =========================
def center_crop_to_match(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _, _, h, w = source.shape
    _, _, th, tw = target.shape
    dh = h - th
    dw = w - tw
    if dh == 0 and dw == 0:
        return source
    top = max(dh // 2, 0)
    left = max(dw // 2, 0)
    return source[:, :, top : top + th, left : left + tw]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = center_crop_to_match(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ShallowUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.enc4 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up3 = UpBlock(64, 32, 32)
        self.up4 = UpBlock(32, 16, 16)
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        p1 = self.pool(c1)
        c2 = self.enc2(p1)
        p2 = self.pool(c2)
        c3 = self.enc3(p2)
        p3 = self.pool(c3)
        c4 = self.enc4(p3)
        p4 = self.pool(c4)
        c5 = self.bottleneck(p4)
        x = self.up1(c5, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        return self.final(x)


class DeepUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.enc4 = DoubleConv(64, 128)
        self.enc5 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up0 = UpBlock(512, 256, 256)
        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up3 = UpBlock(64, 32, 32)
        self.up4 = UpBlock(32, 16, 16)
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        p1 = self.pool(c1)
        c2 = self.enc2(p1)
        p2 = self.pool(c2)
        c3 = self.enc3(p2)
        p3 = self.pool(c3)
        c4 = self.enc4(p3)
        p4 = self.pool(c4)
        c5 = self.enc5(p4)
        p5 = self.pool(c5)
        c6 = self.bottleneck(p5)
        x = self.up0(c6, c5)
        x = self.up1(x, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        return self.final(x)


# =========================
# Training and evaluation
# =========================
def build_loaders(
    train_img: Sequence[str],
    train_mask: Sequence[str],
    val_img: Sequence[str],
    val_mask: Sequence[str],
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = NuInsSegDataset(train_img, train_mask, crop_size=config.crop_size, augment=True)
    val_ds = NuInsSegDataset(val_img, val_mask, crop_size=config.crop_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = bce_dice_loss_from_logits(logits, masks)
            dice = dice_coef_torch(masks, torch.sigmoid(logits))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_dice += dice.item()
        num_batches += 1

    return running_loss / max(num_batches, 1), running_dice / max(num_batches, 1)


def validate_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            loss = bce_dice_loss_from_logits(logits, masks)
            dice = dice_coef_torch(masks, probs)
            running_loss += loss.item()
            running_dice += dice.item()
            num_batches += 1

    return running_loss / max(num_batches, 1), running_dice / max(num_batches, 1)


def load_label_image(path: str) -> np.ndarray:
    label_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if label_img is None:
        raise FileNotFoundError(path)
    return label_img


def load_rgb_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gray_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def predict_single_image(model: nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    image_tensor = torch.from_numpy(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs


def watershed_postprocess(prob_map: np.ndarray, threshold: float, min_object_size: int, peak_footprint: int) -> Tuple[np.ndarray, np.ndarray]:
    pred_bin = (prob_map > threshold).astype(np.uint8)

    local_maxi = peak_local_max(
        prob_map,
        exclude_border=False,
        footprint=np.ones((peak_footprint, peak_footprint), dtype=np.uint8),
        labels=pred_bin,
    )
    marker_mask = np.zeros_like(prob_map, dtype=bool)
    if local_maxi.size > 0:
        marker_mask[tuple(local_maxi.T)] = True
    markers = ndi.label(marker_mask)[0]

    output_watershed = watershed(-prob_map, markers, mask=pred_bin)
    output_watershed[pred_bin == 0] = 0
    output_watershed = remove_small_objects(output_watershed, min_size=min_object_size, connectivity=2)

    output_raw = sk_label(pred_bin)
    output_raw = remove_small_objects(output_raw, min_size=min_object_size, connectivity=2)

    return remap_label(output_raw), remap_label(output_watershed)


def save_instance_png(path: str, arr: np.ndarray) -> None:
    arr_to_save = arr.astype(np.uint16)
    imsave(path, arr_to_save)


def ensure_dirs(config: Config) -> None:
    Path(config.result_save_path).mkdir(parents=True, exist_ok=True)
    Path(config.model_save_path).mkdir(parents=True, exist_ok=True)
    Path(config.metrics_csv_dir).mkdir(parents=True, exist_ok=True)
    Path(config.result_save_path, "validation", "unet").mkdir(parents=True, exist_ok=True)
    Path(config.result_save_path, "validation", "watershed_unet").mkdir(parents=True, exist_ok=True)


def instantiate_model(config: Config) -> nn.Module:
    if config.model_type.lower() == "shallow":
        return ShallowUNet(in_channels=config.num_channels, out_channels=1)
    return DeepUNet(in_channels=config.num_channels, out_channels=1)


def write_history_csv(history_path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_cross_validation(config: Config) -> None:
    seed_everything(config.random_seed)
    ensure_dirs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = discover_paths(config.base_path)
    img_path = paths["img"]
    binary_mask_path = paths["binary"]
    label_mask_path = paths["label"]
    vague_mask_path = paths["vague"]

    kf = KFold(n_splits=config.k_fold, random_state=config.random_seed, shuffle=True)
    start_time = time.time()

    dice_mean, aji_mean, pq_mean = [], [], []
    dice_watershed_mean, aji_watershed_mean, pq_watershed_mean = [], [], []
    dice_watershed_wovague_mean, aji_watershed_wovague_mean, pq_watershed_wovague_mean = [], [], []

    fold_names = []

    for current_fold, (train_index, test_index) in enumerate(kf.split(img_path), start=1):
        print(f"\n{'=' * 20} Fold {current_fold}/{config.k_fold} {'=' * 20}")

        train_img = [img_path[i] for i in train_index]
        train_mask = [binary_mask_path[i] for i in train_index]
        test_img = [img_path[i] for i in test_index]
        test_mask = [binary_mask_path[i] for i in test_index]
        test_label = [label_mask_path[i] for i in test_index]
        test_vague = [vague_mask_path[i] for i in test_index]

        train_loader, val_loader = build_loaders(train_img, train_mask, test_img, test_mask, config)

        model = instantiate_model(config).to(device)
        optimizer = Adam(model.parameters(), lr=config.init_lr)
        scheduler = StepLR(
            optimizer,
            step_size=config.lr_drop_after_nth_epoch,
            gamma=config.lr_decay_factor,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")

        best_val_dice = -np.inf
        model_path = os.path.join(config.model_save_path, f"unet_{current_fold}.pth")
        history_path = os.path.join(config.model_save_path, f"unet_{current_fold}.log.csv")
        history_rows: List[Dict[str, float]] = []

        effective_steps = len(train_loader)
        if config.quick_run > 1:
            effective_steps = max(1, len(train_loader) // config.quick_run)

        for epoch in range(1, config.epochs + 1):
            if effective_steps < len(train_loader):
                limited_batches = []
                for batch_idx, batch in enumerate(train_loader):
                    limited_batches.append(batch)
                    if batch_idx + 1 >= effective_steps:
                        break
                train_loss, train_dice = train_one_epoch_from_batches(model, limited_batches, optimizer, scaler, device, config.use_amp)
            else:
                train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, scaler, device, config.use_amp)

            val_loss, val_dice = validate_one_epoch(model, val_loader, device)
            scheduler.step()

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            print(
                f"Epoch {epoch:03d}/{config.epochs} | "
                f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
            )

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), model_path)

        write_history_csv(history_path, history_rows)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_dice_unet = []
        fold_aji_unet = []
        fold_pq_unet = []
        fold_dice_ws = []
        fold_aji_ws = []
        fold_pq_ws = []
        fold_dice_ws_wovague = []
        fold_aji_ws_wovague = []
        fold_pq_ws_wovague = []

        for img_p, label_p, vague_p in tqdm(list(zip(test_img, test_label, test_vague)), desc=f"Fold {current_fold} inference"):
            image = load_rgb_image(img_p)
            gt_label = remap_label(load_label_image(label_p).astype(np.int32))
            vague_mask = load_gray_image(vague_p)
            prob_map = predict_single_image(model, image, device)
            output_raw, output_watershed = watershed_postprocess(
                prob_map,
                threshold=config.threshold,
                min_object_size=config.min_object_size,
                peak_footprint=config.peak_footprint,
            )

            test_name = get_id_from_file_path(img_p, ".png")
            if config.save_val_results:
                save_instance_png(os.path.join(config.result_save_path, "validation", "unet", f"{test_name}.png"), output_raw)
                save_instance_png(os.path.join(config.result_save_path, "validation", "watershed_unet", f"{test_name}.png"), output_watershed)

            fold_dice_unet.append(get_dice_1(gt_label, output_raw) * 100)
            fold_aji_unet.append(get_fast_aji(gt_label, output_raw) * 100)
            fold_pq_unet.append(get_fast_pq(gt_label, output_raw)[0][2] * 100)

            fold_dice_ws.append(get_dice_1(gt_label, output_watershed) * 100)
            fold_aji_ws.append(get_fast_aji(gt_label, output_watershed) * 100)
            fold_pq_ws.append(get_fast_pq(gt_label, output_watershed)[0][2] * 100)

            output_watershed_wo_vague = np.copy(output_watershed)
            output_watershed_wo_vague[vague_mask == 255] = 0
            output_watershed_wo_vague = remove_small_objects(output_watershed_wo_vague, min_size=config.min_object_size, connectivity=2)
            output_watershed_wo_vague = remap_label(output_watershed_wo_vague)

            gt_label_wo_vague = np.copy(gt_label)
            gt_label_wo_vague[vague_mask == 255] = 0
            gt_label_wo_vague = remove_small_objects(gt_label_wo_vague, min_size=config.min_object_size, connectivity=2)
            gt_label_wo_vague = remap_label(gt_label_wo_vague)

            fold_dice_ws_wovague.append(get_dice_1(gt_label_wo_vague, output_watershed_wo_vague) * 100)
            fold_aji_ws_wovague.append(get_fast_aji(gt_label_wo_vague, output_watershed_wo_vague) * 100)
            fold_pq_ws_wovague.append(get_fast_pq(gt_label_wo_vague, output_watershed_wo_vague)[0][2] * 100)

        fold_names.append(f"fold{current_fold}")
        dice_mean.append(float(np.mean(fold_dice_unet)))
        aji_mean.append(float(np.mean(fold_aji_unet)))
        pq_mean.append(float(np.mean(fold_pq_unet)))
        dice_watershed_mean.append(float(np.mean(fold_dice_ws)))
        aji_watershed_mean.append(float(np.mean(fold_aji_ws)))
        pq_watershed_mean.append(float(np.mean(fold_pq_ws)))
        dice_watershed_wovague_mean.append(float(np.mean(fold_dice_ws_wovague)))
        aji_watershed_wovague_mean.append(float(np.mean(fold_aji_ws_wovague)))
        pq_watershed_wovague_mean.append(float(np.mean(fold_pq_ws_wovague)))

        print("==========")
        print(f"average Dice pure Unet for fold{current_fold}: {dice_mean[-1]:.2f}")
        print(f"average AJI pure Unet for fold{current_fold}: {aji_mean[-1]:.2f}")
        print(f"average PQ pure Unet for fold{current_fold}: {pq_mean[-1]:.2f}")
        print("==========")
        print(f"average Dice Unet watershed for fold{current_fold}: {dice_watershed_mean[-1]:.2f}")
        print(f"average AJI Unet watershed for fold{current_fold}: {aji_watershed_mean[-1]:.2f}")
        print(f"average PQ Unet watershed for fold{current_fold}: {pq_watershed_mean[-1]:.2f}")
        print("==========")
        print(f"average Dice Unet watershed wo vague for fold{current_fold}: {dice_watershed_wovague_mean[-1]:.2f}")
        print(f"average AJI Unet watershed wo vague for fold{current_fold}: {aji_watershed_wovague_mean[-1]:.2f}")
        print(f"average PQ Unet watershed wo vague for fold{current_fold}: {pq_watershed_wovague_mean[-1]:.2f}")
        print("==========")

    df_dice = pd.DataFrame(
        {
            "fold num": fold_names,
            "dice unet": dice_mean,
            "dice unet watershed": dice_watershed_mean,
            "dice unet watershed wo vague": dice_watershed_wovague_mean,
        }
    )
    df_aji = pd.DataFrame(
        {
            "fold num": fold_names,
            "AJI unet": aji_mean,
            "AJI unet watershed": aji_watershed_mean,
            "AJI unet watershed wo vague": aji_watershed_wovague_mean,
        }
    )
    df_pq = pd.DataFrame(
        {
            "fold num": fold_names,
            "PQ unet": pq_mean,
            "PQ unet watershed": pq_watershed_mean,
            "PQ unet watershed wo vague": pq_watershed_wovague_mean,
        }
    )

    df_dice.to_csv(os.path.join(config.metrics_csv_dir, "dice.csv"), index=False)
    df_aji.to_csv(os.path.join(config.metrics_csv_dir, "aji.csv"), index=False)
    df_pq.to_csv(os.path.join(config.metrics_csv_dir, "pq.csv"), index=False)

    print(df_dice)
    print("============================================================")
    print(df_aji)
    print("============================================================")
    print(df_pq)
    print("============================================================")
    finish_time = time.time()
    print(f"total training time (all {config.k_fold} folds): {(finish_time - start_time) / 60:.2f} minutes")


def train_one_epoch_from_batches(
    model: nn.Module,
    batches,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0
    for images, masks in batches:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = bce_dice_loss_from_logits(logits, masks)
            dice = dice_coef_torch(masks, torch.sigmoid(logits))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        running_dice += dice.item()
        num_batches += 1
    return running_loss / max(num_batches, 1), running_dice / max(num_batches, 1)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="PyTorch conversion of deep-u-net-segmentation.ipynb")
    parser.add_argument("--base-path", type=str, default=Config.base_path)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--k-fold", type=int, default=Config.k_fold)
    parser.add_argument("--crop-size", type=int, default=Config.crop_size)
    parser.add_argument("--init-lr", type=float, default=Config.init_lr)
    parser.add_argument("--threshold", type=float, default=Config.threshold)
    parser.add_argument("--random-seed", type=int, default=Config.random_seed)
    parser.add_argument("--quick-run", type=int, default=Config.quick_run)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--result-save-path", type=str, default=Config.result_save_path)
    parser.add_argument("--model-save-path", type=str, default=Config.model_save_path)
    parser.add_argument("--metrics-csv-dir", type=str, default=Config.metrics_csv_dir)
    parser.add_argument("--model-type", type=str, default=Config.model_type, choices=["shallow", "deep"])
    parser.add_argument("--save-val-results", action="store_true")
    parser.add_argument("--no-save-val-results", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    config = Config(
        base_path=args.base_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k_fold=args.k_fold,
        crop_size=args.crop_size,
        init_lr=args.init_lr,
        threshold=args.threshold,
        random_seed=args.random_seed,
        quick_run=args.quick_run,
        num_workers=args.num_workers,
        result_save_path=args.result_save_path,
        model_save_path=args.model_save_path,
        metrics_csv_dir=args.metrics_csv_dir,
        model_type=args.model_type,
        save_val_results=False if args.no_save_val_results else True,
        use_amp=not args.disable_amp,
    )
    if args.save_val_results:
        config.save_val_results = True
    return config


if __name__ == "__main__":
    cfg = parse_args()
    print("Running with config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    run_cross_validation(cfg)
