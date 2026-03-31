import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import tifffile
import pandas as pd
#from hovernet_post_proc import process
from models.hovernet.post_proc import process



#######################################################################################################################
CONFIG = {
    # -----------------------------
    # DATASET
    # -----------------------------
    "images_dir": r"..\..\..\..\datasets\NuFuseRank\custom_split\CryoNuSeg\org_format\tissue images",
    "masks_dir": r"..\..\..\..\datasets\NuFuseRank\custom_split\CryoNuSeg\org_format\Annotator 1 (biologist second round of manual marks up)\label masks modify",
    "num_folds": 5,
    "seed": 19,
    # -----------------------------
    # DATALOADER
    # -----------------------------
    "batch_size": 4,
    # "num_workers": 4,
    # -----------------------------
    # TRAINING
    # -----------------------------
    "epochs": 2
    # "learning_rate": 1e-4,
    # "optimizer": "adam",
    # -----------------------------
    # LOSS WEIGHTS
    # -----------------------------
    # "bce_weight": 1.0,
    # "dice_weight": 1.0,
    # "hv_weight": 2.0,
    # -----------------------------
    # POSTPROCESSING
    # -----------------------------
    # "np_threshold": 0.5,
    # "distance_threshold": 0.4,
    # -----------------------------
    # EVALUATION
    # -----------------------------
    # "pq_iou_threshold": 0.5,
    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    # "figure_dpi": 200,
    # "figure_size": (12,4),
    # -----------------------------
    # FILE OUTPUT
    # -----------------------------
    # "results_dir": "results",
    # "mask_dtype": "uint16",
}



#######################################################################################################################
def list_common_files(images_dir, masks_dir):
    """
    Find matching image and mask files based on filename stem.
    Example:
        images/human_bladder_01.png
        masks/human_bladder_01.tiff
    Both share the stem: human_bladder_01
    """

    cfg = CONFIG
    img_dir = Path(images_dir)
    msk_dir = Path(masks_dir)
    if not img_dir.exists():
        raise RuntimeError(f"Images directory not found: {images_dir}")
    if not msk_dir.exists():
        raise RuntimeError(f"Masks directory not found: {masks_dir}")
    # Collect image files
    img_files = {}
    for p in img_dir.iterdir():
        if p.is_file():
            img_files[p.stem] = p
    # Collect mask files
    msk_files = {}
    for p in msk_dir.iterdir():
        if p.is_file():
            msk_files[p.stem] = p
    # Find common filenames
    common_keys = sorted(list(set(img_files.keys()) & set(msk_files.keys())))

    if len(common_keys) == 0:
        raise RuntimeError(
            "No matching image-mask pairs found. Check filenames."
        )
    pairs = []
    for k in common_keys:
        img_path = str(img_files[k])
        mask_path = str(msk_files[k])
        pairs.append((img_path, mask_path, k))
    return pairs

#######################################################################################################################
def read_image(path):
    """
    Load RGB histology image
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    # OpenCV loads BGR → convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

#######################################################################################################################
def read_instance_mask(path):
    """
    Load instance segmentation mask from TIFF or NPY
    """
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        mask = np.load(path)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {path}")
    else:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {path}")
    # Sometimes TIFF loads as 3 channels
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.int32)
    return mask
#######################################################################################################################
def generate_np_map(inst_map):
    np_map = (inst_map > 0).astype(np.float32)
    return np_map
#######################################################################################################################
def generate_hv_map(inst_map):
    h, w = inst_map.shape
    hv_map = np.zeros((2, h, w), dtype=np.float32)
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]
    for inst_id in inst_ids:
        ys, xs = np.where(inst_map == inst_id)
        if len(xs) == 0:
            continue
        x_center = xs.mean()
        y_center = ys.mean()
        x_rel = xs.astype(np.float32) - x_center
        y_rel = ys.astype(np.float32) - y_center
        x_norm = np.max(np.abs(x_rel))
        y_norm = np.max(np.abs(y_rel))
        if x_norm < 1e-6:
            x_norm = 1.0
        if y_norm < 1e-6:
            y_norm = 1.0
        hv_map[0, ys, xs] = x_rel / x_norm
        hv_map[1, ys, xs] = y_rel / y_norm
    return hv_map

#######################################################################################################################
def visualize_hover_targets(image, inst_map):
    np_map = generate_np_map(inst_map)
    hv_map = generate_hv_map(inst_map)
    h_map = hv_map[0]
    v_map = hv_map[1]
    plt.figure(figsize=(12,6))
    plt.subplot(2,3,1)
    plt.title("RGB image")
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(2,3,2)
    plt.title("Instance mask")
    plt.imshow(inst_map)
    plt.axis("off")
    plt.subplot(2,3,3)
    plt.title("NP map")
    plt.imshow(np_map, cmap="gray")
    plt.axis("off")
    plt.subplot(2,3,4)
    plt.title("Horizontal map")
    plt.imshow(h_map, cmap="coolwarm")
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.title("Vertical map")
    plt.imshow(v_map, cmap="coolwarm")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


#######################################################################################################################
class NuInsSegHoverDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):

        img_path, mask_path, name = self.samples[idx]
        # load image
        img = read_image(img_path)
        # load instance mask
        inst_map = read_instance_mask(mask_path)
        # generate targets
        np_map = generate_np_map(inst_map)
        hv_map = generate_hv_map(inst_map)
        # normalize image
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # convert to channel-first
        img = np.transpose(img, (2, 0, 1))
        # add channel dimension to NP
        np_map = np.expand_dims(np_map, axis=0)
        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "np_map": torch.tensor(np_map, dtype=torch.float32),
            "hv_map": torch.tensor(hv_map, dtype=torch.float32),
            "name": name,
            "inst_map": torch.tensor(inst_map, dtype=torch.int64),

        }
#######################################################################################################################
def create_folds(samples, n_folds=5, seed=19):
    kf = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )
    folds = []
    for train_idx, val_idx in kf.split(samples):
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        folds.append((train_samples, val_samples))
    return folds
#######################################################################################################################
def create_dataloaders(train_samples, val_samples, batch_size=4):
    train_dataset = NuInsSegHoverDataset(train_samples)
    val_dataset = NuInsSegHoverDataset(val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader
#######################################################################################################################
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)

        )
    def forward(self, x):

        return self.block(x)
#######################################################################################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)
#######################################################################################################################
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(3,64),
            ResidualBlock(64,64)
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(64,128)
        )

        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(128,256)
        )

        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(256,512)
        )
    def forward(self,x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x1,x2,x3,x4

#######################################################################################################################
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(out_ch + skip_ch, out_ch),
            ResidualBlock(out_ch,out_ch)
        )

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat([x,skip],dim=1)
        return self.conv(x)
#######################################################################################################################
class DecoderBranch(nn.Module):

    def __init__(self, out_channels):

        super().__init__()

        self.dec3 = DecoderBlock(512,256,256)
        self.dec2 = DecoderBlock(256,128,128)
        self.dec1 = DecoderBlock(128,64,64)

        self.head = nn.Conv2d(64,out_channels,1)

    def forward(self,x1,x2,x3,x4):

        x = self.dec3(x4,x3)
        x = self.dec2(x,x2)
        x = self.dec1(x,x1)

        return self.head(x)
#######################################################################################################################
class HoverNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.np_branch = DecoderBranch(1)
        self.hv_branch = DecoderBranch(2)
    def forward(self,x):
        x1,x2,x3,x4 = self.encoder(x)
        np_out = self.np_branch(x1,x2,x3,x4)
        hv_out = self.hv_branch(x1,x2,x3,x4)
        return {
            "np":np_out,
            "hv":hv_out
        }
#######################################################################################################################
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice.mean()
        return loss
#######################################################################################################################
class HoverNetLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, hv_weight=2.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.mse = nn.MSELoss(reduction="none")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.hv_weight = hv_weight

    def forward(self, outputs, target_np, target_hv):
        pred_np = outputs["np"]
        pred_hv = outputs["hv"]
        # NP losses
        loss_bce = self.bce(pred_np, target_np)
        loss_dice = self.dice(pred_np, target_np)
        # HV loss only inside nuclei
        fg_mask = (target_np > 0.5).float()
        hv_loss_map = self.mse(pred_hv, target_hv)
        hv_loss_map = hv_loss_map * fg_mask
        loss_hv = hv_loss_map.sum() / (fg_mask.sum() * 2.0 + 1e-6)
        total_loss = (
            self.bce_weight * loss_bce +
            self.dice_weight * loss_dice +
            self.hv_weight * loss_hv
        )
        loss_dict = {
            "loss_total": total_loss,
            "loss_bce": loss_bce.item(),
            "loss_dice": loss_dice.item(),
            "loss_hv": loss_hv.item()
        }
        return total_loss, loss_dict
#######################################################################################################################
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import numpy as np
import cv2


def hover_postprocess(np_pred, hv_pred, thresh=0.5):

    np_prob = torch.sigmoid(np_pred).cpu().numpy()[0,0]

    hv = hv_pred.cpu().numpy()[0]

    h_dir = hv[0]
    v_dir = hv[1]

    # binary nuclei mask
    nuclei = np_prob > thresh

    # gradient magnitude from HV maps
    sobel_h = cv2.Sobel(h_dir, cv2.CV_64F,1,0,ksize=5)
    sobel_v = cv2.Sobel(v_dir, cv2.CV_64F,0,1,ksize=5)

    grad = np.sqrt(sobel_h**2 + sobel_v**2)

    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

    distance = ndi.distance_transform_edt(nuclei)

    markers = ndi.label(distance > 0.4 * distance.max())[0]

    inst_map = watershed(grad, markers, mask=nuclei)

    return inst_map.astype(np.uint16)
#######################################################################################################################
def dice_score(pred, gt):

    pred_bin = pred > 0
    gt_bin = gt > 0

    inter = (pred_bin & gt_bin).sum()

    denom = pred_bin.sum() + gt_bin.sum()

    if denom == 0:
        return 1.0

    return 2 * inter / denom
#######################################################################################################################
def compute_pq(gt, pred, iou_thresh=0.5):

    gt_ids = np.unique(gt)
    pred_ids = np.unique(pred)

    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = pred_ids[pred_ids > 0]

    tp = 0
    fp = 0
    fn = 0
    sum_iou = 0

    matched_pred = set()

    for gid in gt_ids:

        g_mask = gt == gid

        best_iou = 0
        best_pid = None

        for pid in pred_ids:

            if pid in matched_pred:
                continue

            p_mask = pred == pid

            inter = np.logical_and(g_mask, p_mask).sum()

            union = np.logical_or(g_mask, p_mask).sum()

            if union == 0:
                continue

            iou = inter / union

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

    if tp == 0:
        return 0

    pq = sum_iou / (tp + 0.5 * fp + 0.5 * fn)

    return pq
#######################################################################################################################
# def validate(model, loader, device, save_dir):
#
#     model.eval()
#
#     dice_scores = []
#     pq_scores = []
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     fig_dir = os.path.join(save_dir, "figures")
#     os.makedirs(fig_dir, exist_ok=True)
#
#     with torch.no_grad():
#
#         for batch in tqdm(loader):
#
#             images = batch["image"].to(device)
#             names = batch["name"]
#
#             inst_maps = batch["inst_map"].cpu().numpy()
#
#             outputs = model(images)
#
#             for i in range(images.shape[0]):
#
#                 pred_inst = hover_postprocess(
#                     outputs["np"][i:i+1],
#                     outputs["hv"][i:i+1]
#                 )
#
#                 gt_inst = inst_maps[i]
#
#                 dice = dice_score(pred_inst, gt_inst)
#                 pq = compute_pq(gt_inst, pred_inst)
#
#                 dice_scores.append(dice)
#                 pq_scores.append(pq)
#
#                 name = names[i]
#
#                 # save predicted mask
#                 mask_path = os.path.join(save_dir, name + ".tiff")
#                 tifffile.imwrite(mask_path, pred_inst.astype(np.uint16))
#
#                 # recover original image for visualization
#                 img = images[i].cpu().numpy().transpose(1,2,0)
#
#                 # undo normalization
#                 mean = np.array([0.485,0.456,0.406])
#                 std = np.array([0.229,0.224,0.225])
#
#                 img = std * img + mean
#                 img = np.clip(img,0,1)
#
#                 fig_path = os.path.join(fig_dir, name + ".png")
#
#                 save_validation_figure(img, gt_inst, pred_inst, fig_path)
#
#     mean_dice = np.mean(dice_scores)
#     mean_pq = np.mean(pq_scores)
#
#     return mean_dice, mean_pq

#######################################################################################################################
def validate(model, loader, device, save_dir):

    model.eval()

    dice_scores = []
    pq_scores = []

    os.makedirs(save_dir, exist_ok=True)

    fig_dir = os.path.join(save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    with torch.no_grad():

        for batch in tqdm(loader):

            images = batch["image"].to(device)
            names = batch["name"]
            inst_maps = batch["inst_map"].cpu().numpy()

            outputs = model(images)

            np_pred = outputs["np"]
            hv_pred = outputs["hv"]

            for i in range(images.shape[0]):

                # convert prediction to numpy
                np_map = torch.sigmoid(np_pred[i]).cpu().numpy()[0]
                hv_map = hv_pred[i].cpu().numpy()

                # combine NP + HV exactly as official code expects
                pred_map = np.stack(
                    [np_map, hv_map[0], hv_map[1]],
                    axis=-1
                )

                # official HoVer-Net post-processing
                pred_inst, _ = process(pred_map)

                gt_inst = inst_maps[i]

                dice = dice_score(pred_inst, gt_inst)
                pq = compute_pq(gt_inst, pred_inst)

                dice_scores.append(dice)
                pq_scores.append(pq)

                name = names[i]

                # save instance mask
                mask_path = os.path.join(save_dir, name + ".tiff")
                tifffile.imwrite(mask_path, pred_inst.astype(np.uint16))

                # recover RGB image for visualization
                img = images[i].cpu().numpy().transpose(1, 2, 0)

                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])

                img = std * img + mean
                img = np.clip(img, 0, 1)

                fig_path = os.path.join(fig_dir, name + ".png")

                save_validation_figure(
                    img,
                    gt_inst,
                    pred_inst,
                    fig_path
                )

    mean_dice = np.mean(dice_scores)
    mean_pq = np.mean(pq_scores)

    return mean_dice, mean_pq
#######################################################################################################################
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0

    for batch in tqdm(loader):

        images = batch["image"].to(device)
        np_map = batch["np_map"].to(device)
        hv_map = batch["hv_map"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss, loss_dict = criterion(outputs, np_map, hv_map)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)
#######################################################################################################################
def compute_validation_loss(model, loader, criterion, device):

    model.eval()

    running_loss = 0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)
            np_map = batch["np_map"].to(device)
            hv_map = batch["hv_map"].to(device)

            outputs = model(images)

            loss, _ = criterion(outputs, np_map, hv_map)

            running_loss += loss.item()

    return running_loss / len(loader)
#######################################################################################################################
def run_training(folds, device, epochs=3, batch_size_num=4):

    results = []

    for fold, (train_samples, val_samples) in enumerate(folds):

        print("\n========================")
        print("Starting Fold", fold + 1)
        print("========================")

        train_loader, val_loader = create_dataloaders(
            train_samples,
            val_samples,
            batch_size=batch_size_num
        )

        model = HoverNet().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        criterion = HoverNetLoss()

        best_val_loss = float("inf")

        os.makedirs("results/checkpoints", exist_ok=True)

        for epoch in range(epochs):

            print("\nEpoch", epoch + 1)

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )

            print("Train loss:", train_loss)

            # -------- validation loss --------
            val_loss = compute_validation_loss(
                model,
                val_loader,
                criterion,
                device
            )

            print("Validation loss:", val_loss)

            # -------- save best model --------
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                save_path = f"./results/checkpoints/best_model_fold_{fold+1}.pth"

                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss
                }, save_path)

                print("Saved new best model:", save_path)

        # load best model before final evaluation
        checkpoint = torch.load(f"./results/checkpoints/best_model_fold_{fold+1}.pth")

        model.load_state_dict(checkpoint["model_state_dict"])

        print("Loaded best model from epoch:", checkpoint["epoch"])

        save_dir = f"results/fold_{fold + 1}"

        mean_dice, mean_pq = validate(
            model,
            val_loader,
            device,
            save_dir
        )



        print("\nFold", fold + 1, "results")
        print("Mean Dice:", mean_dice)
        print("Mean PQ:", mean_pq)

        results.append({
            "fold": fold + 1,
            "dice": mean_dice,
            "pq": mean_pq
        })

    return results
#######################################################################################################################
# def run_training(folds, device, epochs  = 3, batch_size_num = 4):
#
#     results = []
#
#     for fold, (train_samples, val_samples) in enumerate(folds):
#
#         print("\n========================")
#         print("Starting Fold", fold + 1)
#         print("========================")
#
#         train_loader, val_loader = create_dataloaders(
#             train_samples,
#             val_samples,
#             batch_size= batch_size_num
#         )
#
#         model = HoverNet().to(device)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#
#         criterion = HoverNetLoss()
#
#
#         for epoch in range(epochs):
#
#             print("\nEpoch", epoch + 1)
#
#             train_loss = train_one_epoch(
#                 model,
#                 train_loader,
#                 optimizer,
#                 criterion,
#                 device
#             )
#
#             print("Train loss:", train_loss)
#
#         save_dir = f"results/fold_{fold+1}"
#
#         mean_dice, mean_pq = validate(
#             model,
#             val_loader,
#             device,
#             save_dir
#         )
#
#         print("\nFold", fold + 1, "results")
#         print("Mean Dice:", mean_dice)
#         print("Mean PQ:", mean_pq)
#
#         results.append({
#             "fold": fold + 1,
#             "dice": mean_dice,
#             "pq": mean_pq
#         })
#
#     return results
#######################################################################################################################
def save_results(results):

    df = pd.DataFrame(results)

    mean_dice = df["dice"].mean()
    mean_pq = df["pq"].mean()

    df.loc["mean"] = ["-", mean_dice, mean_pq]

    df.to_csv("cross_validation_results.csv", index=False)

    print("\n==========================")
    print("FINAL RESULTS")
    print("==========================")

    print("Mean Dice:", mean_dice)
    print("Mean PQ:", mean_pq)
#######################################################################################################################
def save_validation_figure(image, gt_mask, pred_mask, save_path):

    gt_color = colorize_instances(gt_mask)
    pred_color = colorize_instances(pred_mask)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(gt_color)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(pred_color)
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(save_path, dpi=200)

    plt.close()
#######################################################################################################################
def colorize_instances(inst_map):

    h, w = inst_map.shape

    colored = np.zeros((h, w, 3), dtype=np.uint8)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    rng = np.random.default_rng(123)

    for inst_id in inst_ids:

        mask = inst_map == inst_id

        color = rng.integers(50, 255, size=3)

        colored[mask] = color

    return colored
#######################################################################################################################

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # CHANGE THESE PATHS
    images_dir = cfg["images_dir"]
    masks_dir = cfg["masks_dir"]

    pairs = list_common_files(images_dir, masks_dir)
    print("Total matched samples:", len(pairs))
    print()

    folds = create_folds(pairs, n_folds=cfg["num_folds"], seed=cfg["seed"])
    print("Total folds:", len(folds))

    results = run_training(folds, device, epochs = cfg["epochs"], batch_size_num = cfg["batch_size"])

    save_results(results)

    # model = HoverNet()
    # x = torch.randn(1, 3, 512, 512)
    # out = model(x)
    # print(out["np"].shape)
    # print(out["hv"].shape)



    # train_samples, val_samples = folds[0]
    #
    # train_loader, val_loader = create_dataloaders(
    #     train_samples,
    #     val_samples,
    #     batch_size=4
    # )
    #
    # batch = next(iter(train_loader))
    # print(batch["image"].shape)
    # print(batch["np_map"].shape)
    # print(batch["hv_map"].shape)
    # print("Train batches:", len(train_loader))
    # print("Val batches:", len(val_loader))

    # for i, (train_samples, val_samples) in enumerate(folds):
    #     print("Fold", i + 1)
    #     print("Train:", len(train_samples))
    #     print("Val:", len(val_samples))
    #     print()

    # # show first 10 pairs
    # for i in range(min(10, len(pairs))):
    #     img, mask, name = pairs[i]
    #
    #     print("Sample:", name)
    #     print("Image:", img)
    #     print("Mask :", mask)
    #     print()
    #
    #     img_path, mask_path, name = pairs[0]
    #
    #     img = read_image(img_path)
    #     mask = read_instance_mask(mask_path)
    #
    #     print("Image shape:", img.shape)
    #     print("Mask shape:", mask.shape)
    #     print("Mask dtype:", mask.dtype)
    #
    #     print("First 20 unique labels:")
    #     print(np.unique(mask)[:20])
    #
    #     np_map = generate_np_map(mask)
    #
    #     hv_map = generate_hv_map(mask)
    #
    #     print("NP map shape:", np_map.shape)
    #     print("HV map shape:", hv_map.shape)
    #
    #     print("HV min:", hv_map.min())
    #     print("HV max:", hv_map.max())
    #
    #     img_path, mask_path, name = pairs[i]
    #
    #     image = read_image(img_path)
    #     inst_map = read_instance_mask(mask_path)
    #
    #     visualize_hover_targets(image, inst_map)



if __name__ == "__main__":
    main(CONFIG)