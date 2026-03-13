import os
import cv2
import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import ndimage as ndi
from sklearn.model_selection import KFold
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models


# =========================================================
# SETTINGS
# =========================================================

image_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\tissue images"
mask_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\label masks modify"
save_root = r"C:\Users\amahbod\projects\fulbright\results\dualdecoder_unet"

num_epochs = 80
batch_size = 16
lr = 1e-4
num_folds = 5
early_stop_patience = 10

binary_threshold = 0.4
dist_threshold_rel = 0.25   # relative threshold for local maxima from distance map

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(save_root, exist_ok=True)


# =========================================================
# UTILITIES
# =========================================================

def build_distance_map(inst_map):
    """
    inst_map: labeled instance mask, shape HxW, 0=background, 1..N=instances
    returns normalized distance map in [0,1]
    """
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids != 0]

    dist_map = np.zeros(inst_map.shape, dtype=np.float32)

    for inst_id in inst_ids:
        nucleus = (inst_map == inst_id).astype(np.uint8)
        if nucleus.sum() == 0:
            continue

        dist = ndi.distance_transform_edt(nucleus)
        if dist.max() > 0:
            dist = dist / dist.max()

        dist_map[nucleus > 0] = np.maximum(dist_map[nucleus > 0], dist[nucleus > 0])

    return dist_map.astype(np.float32)


def dice_score(gt, pred):
    gt_bin = gt > 0
    pred_bin = pred > 0

    inter = np.logical_and(gt_bin, pred_bin).sum()
    denom = gt_bin.sum() + pred_bin.sum()

    return (2.0 * inter) / (denom + 1e-8)


def pq_score(gt, pred, iou_thresh=0.5):
    """
    GT and pred are labeled instance masks.
    """
    gt_lab = label(gt > 0) if gt.dtype == bool else gt
    pred_lab = label(pred > 0) if pred.dtype == bool else pred

    gt_ids = np.unique(gt_lab)
    pred_ids = np.unique(pred_lab)

    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    tp = 0
    fn = 0
    sum_iou = 0.0
    matched_pred = set()

    for gid in gt_ids:
        g = (gt_lab == gid)

        best_iou = 0.0
        best_pid = None

        for pid in pred_ids:
            if pid in matched_pred:
                continue

            p = (pred_lab == pid)

            inter = np.logical_and(g, p).sum()
            if inter == 0:
                continue

            union = np.logical_or(g, p).sum()
            iou = inter / (union + 1e-8)

            if iou > best_iou:
                best_iou = iou
                best_pid = pid

        if best_iou > iou_thresh:
            tp += 1
            sum_iou += best_iou
            matched_pred.add(best_pid)
        else:
            fn += 1

    fp = len(pred_ids) - len(matched_pred)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
    sq = sum_iou / (tp + 1e-8)
    pq = dq * sq

    return pq


def save_visualization(img, gt, pred, save_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img)
    ax[0].set_title("Raw image")

    ax[1].imshow(gt, cmap="nipy_spectral")
    ax[1].set_title("Ground truth")

    ax[2].imshow(pred, cmap="nipy_spectral")
    ax[2].set_title("Prediction")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# DATASET
# =========================================================

class NucleiDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = tiff.imread(mask_path)

        if mask.ndim > 2:
            mask = np.squeeze(mask)

        mask = mask.astype(np.int32)

        mask_bin = (mask > 0).astype(np.float32)
        dist_map = build_distance_map(mask)

        img_norm = img.astype(np.float32) / 255.0
        img_norm = np.transpose(img_norm, (2, 0, 1))

        return (
            torch.tensor(img_norm, dtype=torch.float32),                     # [3,H,W]
            torch.tensor(mask_bin, dtype=torch.float32).unsqueeze(0),       # [1,H,W]
            torch.tensor(dist_map, dtype=torch.float32).unsqueeze(0),       # [1,H,W]
            img,                                                            # raw image
            mask,                                                           # GT labeled mask
            os.path.basename(img_path)
        )


# =========================================================
# MODEL
# =========================================================

class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvRelu(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class EncoderResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet34(weights=weights)

        self.initial = nn.Sequential(
            base.conv1,   # /2
            base.bn1,
            base.relu
        )
        self.maxpool = base.maxpool  # /4
        self.layer1 = base.layer1    # /4
        self.layer2 = base.layer2    # /8
        self.layer3 = base.layer3    # /16
        self.layer4 = base.layer4    # /32

    def forward(self, x):
        x0 = self.initial(x)           # 64, /2
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)           # 64, /4
        x2 = self.layer2(x1)           # 128, /8
        x3 = self.layer3(x2)           # 256, /16
        x4 = self.layer4(x3)           # 512, /32
        return x0, x1, x2, x3, x4


class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        self.center = ConvRelu(512, 512)

        self.dec4 = DecoderBlock(512, 256, 256)  # x4 + x3
        self.dec3 = DecoderBlock(256, 128, 128)  # + x2
        self.dec2 = DecoderBlock(128, 64, 64)    # + x1
        self.dec1 = DecoderBlock(64, 64, 64)     # + x0

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, feats):
        x0, x1, x2, x3, x4 = feats

        x = self.center(x4)
        x = self.dec4(x, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, x0)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.final_conv(x)
        return x


class DualDecoderUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = EncoderResNet34(pretrained=pretrained)
        self.binary_decoder = UNetDecoder(out_channels=1)
        self.dist_decoder = UNetDecoder(out_channels=1)

    def forward(self, x):
        feats = self.encoder(x)
        binary_logits = self.binary_decoder(feats)
        dist_pred = self.dist_decoder(feats)
        return binary_logits, dist_pred


# =========================================================
# WATERSHED POSTPROCESS USING PREDICTED DISTANCE MAP
# =========================================================

def watershed_postprocess(binary_prob, dist_pred, bin_thresh=0.4, dist_rel_thresh=0.25):
    """
    binary_prob: [H,W] after sigmoid
    dist_pred:   [H,W] raw or regressed distance map
    """
    binary = binary_prob > bin_thresh

    if binary.sum() == 0:
        return np.zeros(binary.shape, dtype=np.uint16)

    dist_pred = np.clip(dist_pred, 0, None)

    max_val = dist_pred.max()
    if max_val > 0:
        dist_norm = dist_pred / max_val
    else:
        dist_norm = dist_pred.copy()

    # local maxima based on predicted distance map
    coords = peak_local_max(
        dist_norm,
        min_distance=3,
        threshold_abs=dist_rel_thresh,
        labels=binary
    )

    if len(coords) == 0:
        # fallback: connected components on binary mask
        markers, _ = ndi.label(binary)
        labels = watershed(-dist_norm, markers, mask=binary)
        return labels.astype(np.uint16)

    marker_mask = np.zeros(binary.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(marker_mask)

    labels = watershed(-dist_norm, markers, mask=binary)
    return labels.astype(np.uint16)


# =========================================================
# LOSSES
# =========================================================

class DiceLossBinary(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLossBinary()
mse_loss = nn.MSELoss()


def compute_total_loss(binary_logits, dist_pred, mask_bin, dist_gt,
                       w_bce=1.0, w_dice=1.0, w_mse=1.0):
    loss_bce = bce_loss(binary_logits, mask_bin)
    loss_dice = dice_loss(binary_logits, mask_bin)
    loss_mse = mse_loss(dist_pred, dist_gt)

    total = w_bce * loss_bce + w_dice * loss_dice + w_mse * loss_mse
    return total, loss_bce.item(), loss_dice.item(), loss_mse.item()


# =========================================================
# LOAD FILES
# =========================================================

image_files = sorted([
    os.path.join(image_dir, x)
    for x in os.listdir(image_dir)
    if x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
])

mask_files = sorted([
    os.path.join(mask_dir, x)
    for x in os.listdir(mask_dir)
    if x.lower().endswith((".tif", ".tiff", ".png"))
])

assert len(image_files) == len(mask_files), "Number of images and masks must match."

# optional strong filename check
for img_p, mask_p in zip(image_files, mask_files):
    img_name = os.path.splitext(os.path.basename(img_p))[0]
    mask_name = os.path.splitext(os.path.basename(mask_p))[0]
    assert img_name == mask_name, f"Mismatch: {img_name} vs {mask_name}"

print("Total samples:", len(image_files))


# =========================================================
# CROSS VALIDATION
# =========================================================

kf = KFold(n_splits=num_folds, shuffle=True, random_state=19)

all_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(image_files), start=1):
    print("\n" + "=" * 50)
    print(f"FOLD {fold}")
    print("=" * 50)

    train_images = [image_files[i] for i in train_idx]
    train_masks = [mask_files[i] for i in train_idx]

    val_images = [image_files[i] for i in val_idx]
    val_masks = [mask_files[i] for i in val_idx]

    train_dataset = NucleiDataset(train_images, train_masks)
    val_dataset = NucleiDataset(val_images, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = DualDecoderUNet(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fold_dir = os.path.join(save_root, f"fold_{fold}")
    mask_save_dir = os.path.join(fold_dir, "masks")
    vis_save_dir = os.path.join(fold_dir, "visualizations")
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    best_val_dice = -1
    patience_counter = 0
    best_ckpt_path = os.path.join(fold_dir, "best_model.pth")

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------
    for epoch in range(num_epochs):
        model.train()

        epoch_total = 0.0
        epoch_bce = 0.0
        epoch_dice = 0.0
        epoch_mse = 0.0

        train_bar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{num_epochs}")

        for imgs, mask_bin, dist_gt, _, _, _ in train_bar:
            imgs = imgs.to(device)
            mask_bin = mask_bin.to(device)
            dist_gt = dist_gt.to(device)

            binary_logits, dist_pred = model(imgs)

            loss, l_bce, l_dice, l_mse = compute_total_loss(
                binary_logits, dist_pred, mask_bin, dist_gt,
                w_bce=1.0, w_dice=1.0, w_mse=1.0
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_bce += l_bce
            epoch_dice += l_dice
            epoch_mse += l_mse

            train_bar.set_postfix(
                total=f"{loss.item():.4f}",
                bce=f"{l_bce:.4f}",
                dice=f"{l_dice:.4f}",
                mse=f"{l_mse:.4f}"
            )

        epoch_total /= len(train_loader)
        epoch_bce /= len(train_loader)
        epoch_dice /= len(train_loader)
        epoch_mse /= len(train_loader)

        # -------------------------------------------------
        # VALIDATION DICE PER EPOCH
        # Use binary branch only for fast validation
        # -------------------------------------------------
        model.eval()
        val_dice_list = []

        with torch.no_grad():
            for imgs, _, _, _, gt_mask, _ in val_loader:
                imgs = imgs.to(device)

                binary_logits, _ = model(imgs)
                binary_prob = torch.sigmoid(binary_logits).cpu().numpy()[0, 0]

                pred_bin = (binary_prob > binary_threshold).astype(np.uint8)
                gt_mask_np = gt_mask.numpy()[0]

                d = dice_score(gt_mask_np, pred_bin)
                val_dice_list.append(d)

        val_dice = float(np.mean(val_dice_list))

        print(f"\nFold {fold} | Epoch {epoch+1}/{num_epochs}")
        print(f"Train total loss: {epoch_total:.4f}")
        print(f"  BCE loss:       {epoch_bce:.4f}")
        print(f"  Dice loss:      {epoch_dice:.4f}")
        print(f"  MSE loss:       {epoch_mse:.4f}")
        print(f"Validation Dice:  {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print("Best model updated.")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    # -----------------------------------------------------
    # LOAD BEST MODEL FOR FINAL VALIDATION PREDICTIONS
    # -----------------------------------------------------
    print(f"\nLoading best model for fold {fold}...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    fold_results = []

    with torch.no_grad():
        for imgs, _, _, raw_img, gt_mask, name in tqdm(val_loader, desc=f"Saving fold {fold} results"):
            imgs = imgs.to(device)

            binary_logits, dist_pred = model(imgs)

            binary_prob = torch.sigmoid(binary_logits).cpu().numpy()[0, 0]
            dist_pred_np = dist_pred.cpu().numpy()[0, 0]

            pred_instances = watershed_postprocess(
                binary_prob=binary_prob,
                dist_pred=dist_pred_np,
                bin_thresh=binary_threshold,
                dist_rel_thresh=dist_threshold_rel
            )

            gt_mask_np = gt_mask.numpy()[0]
            raw_img_np = raw_img[0].numpy()

            # save predicted labeled mask as tif
            base_name = os.path.splitext(name[0])[0] + ".tif"
            pred_mask_path = os.path.join(mask_save_dir, base_name)
            tiff.imwrite(pred_mask_path, pred_instances.astype(np.uint16))

            # save visualization
            vis_name = os.path.splitext(name[0])[0] + ".png"
            vis_path = os.path.join(vis_save_dir, vis_name)
            save_visualization(raw_img_np, gt_mask_np, pred_instances, vis_path)

            # metrics
            d = dice_score(gt_mask_np, pred_instances)
            p = pq_score(gt_mask_np, pred_instances)

            result_row = {
                "image": name[0],
                "fold": fold,
                "dice": d,
                "pq": p
            }

            fold_results.append(result_row)
            all_results.append(result_row)

    # save fold csv
    fold_df = pd.DataFrame(fold_results)
    fold_csv_path = os.path.join(fold_dir, f"metrics_fold_{fold}.csv")
    fold_df.to_csv(fold_csv_path, index=False)

    print(f"\nFold {fold} average Dice: {fold_df['dice'].mean():.4f}")
    print(f"Fold {fold} average PQ:   {fold_df['pq'].mean():.4f}")


# =========================================================
# FINAL CSV AND AVERAGES
# =========================================================

all_df = pd.DataFrame(all_results)
all_csv_path = os.path.join(save_root, "metrics_all_folds.csv")
all_df.to_csv(all_csv_path, index=False)

print("\n" + "=" * 50)
print("FINAL RESULTS ACROSS ALL 3 FOLDS")
print("=" * 50)
print(f"Average Dice: {all_df['dice'].mean():.4f}")
print(f"Average PQ:   {all_df['pq'].mean():.4f}")
print(f"CSV saved to: {all_csv_path}")