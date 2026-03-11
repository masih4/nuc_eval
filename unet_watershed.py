import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

import segmentation_models_pytorch as smp

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.measure import label


# =============================
# SETTINGS
# =============================

image_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\tissue images"
mask_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\label masks modify"
save_root = r"C:\Users\amahbod\projects\fulbright\results\unet_watershed"

num_epochs = 10
batch_size = 32
lr = 1e-4
num_folds = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(save_root, exist_ok=True)


# =============================
# DATASET
# =============================

class NucleiDataset(Dataset):

    def __init__(self, images, masks):

        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = tiff.imread(self.masks[idx])

        img_norm = img.astype(np.float32) / 255.0
        img_norm = np.transpose(img_norm, (2, 0, 1))

        mask_bin = (mask > 0).astype(np.float32)

        return (
            torch.tensor(img_norm).float(),
            torch.tensor(mask_bin).unsqueeze(0).float(),
            img,
            mask,
            os.path.basename(self.images[idx])
        )

# =============================
# WATERSHED
# =============================

def watershed_postprocess(prob):

    binary = prob > 0.5

    distance = ndi.distance_transform_edt(binary)

    # coords = peak_local_max(distance, footprint=np.ones((3,3)), labels=binary)
    # mask = np.zeros(distance.shape, dtype=bool)
    # mask[tuple(coords.T)] = True
    # markers,_ = ndi.label(mask)

    # fast marker detection
    local_max = distance > (0.5 * distance.max())
    markers, _ = ndi.label(local_max)

    labels = watershed(-distance, markers, mask=binary)

    return labels.astype(np.uint16)


# =============================
# METRICS
# =============================

def dice_score(gt, pred):

    gt_bin = gt > 0
    pred_bin = pred > 0

    intersection = np.logical_and(gt_bin, pred_bin).sum()

    return 2*intersection / (gt_bin.sum() + pred_bin.sum() + 1e-8)


def pq_score(gt, pred, iou_thresh=0.5):

    gt_instances = label(gt)
    pred_instances = label(pred)

    gt_ids = np.unique(gt_instances)[1:]
    pred_ids = np.unique(pred_instances)[1:]

    TP = 0
    FP = 0
    FN = 0
    sum_iou = 0

    matched_pred = set()

    for gid in gt_ids:

        g = gt_instances == gid
        best_iou = 0
        best_pid = None

        for pid in pred_ids:

            if pid in matched_pred:
                continue

            p = pred_instances == pid

            inter = np.logical_and(g,p).sum()
            union = np.logical_or(g,p).sum()

            iou = inter / (union + 1e-8)

            if iou > best_iou:
                best_iou = iou
                best_pid = pid

        if best_iou > iou_thresh:
            TP += 1
            sum_iou += best_iou
            matched_pred.add(best_pid)
        else:
            FN += 1

    FP = len(pred_ids) - len(matched_pred)

    dq = TP / (TP + 0.5*FP + 0.5*FN + 1e-8)

    sq = sum_iou / (TP + 1e-8)

    pq = dq * sq

    return pq


# =============================
# VISUALIZATION
# =============================

def save_visualization(img, gt, pred, save_path):

    fig,ax = plt.subplots(1,3,figsize=(12,4))

    ax[0].imshow(img)
    ax[0].set_title("Image")

    ax[1].imshow(gt, cmap="nipy_spectral")
    ax[1].set_title("Ground Truth")

    ax[2].imshow(pred, cmap="nipy_spectral")
    ax[2].set_title("Prediction")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =============================
# LOAD DATA
# =============================

images = sorted([os.path.join(image_dir,x) for x in os.listdir(image_dir)])
masks = sorted([os.path.join(mask_dir,x) for x in os.listdir(mask_dir)])

print("Total images:",len(images))


# =============================
# MODEL
# =============================

def build_model():

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )



#criterion = smp.losses.DiceLoss(mode="binary") + torch.nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = torch.nn.BCEWithLogitsLoss()

# =============================
# CROSS VALIDATION
# =============================

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

all_results = []


for fold,(train_idx,val_idx) in enumerate(kf.split(images)):

    print("\nFOLD",fold+1)

    train_images = [images[i] for i in train_idx]
    train_masks = [masks[i] for i in train_idx]

    val_images = [images[i] for i in val_idx]
    val_masks = [masks[i] for i in val_idx]

    train_dataset = NucleiDataset(train_images,train_masks)
    val_dataset = NucleiDataset(val_images,val_masks)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)

    model = build_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    # =============================
    # TRAIN
    # =============================
    best_val_dice = 0
    patience_counter = 0
    early_stop_patience = 10

    checkpoint_dir = os.path.join(save_root, f"fold_{fold + 1}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):

        # --------------------
        # TRAIN
        # --------------------
        model.train()

        train_loss = 0

        loop = tqdm(train_loader, desc=f"Fold {fold + 1} Epoch {epoch + 1}/{num_epochs}")

        for imgs, masks_bin, _, _, _ in loop:
            imgs = imgs.to(device)
            masks_bin = masks_bin.to(device)

            preds = model(imgs)

            #loss = criterion(preds, masks_bin)
            loss = dice_loss(preds, masks_bin) + bce_loss(preds, masks_bin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # --------------------
        # VALIDATION
        # --------------------
        model.eval()

        dice_list = []

        with torch.no_grad():

            for imgs, _, _, gt_mask, _ in val_loader:
                imgs = imgs.to(device)

                logits = model(imgs)

                prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

                pred_instances = watershed_postprocess(prob)

                gt_mask = gt_mask.numpy()[0]

                d = dice_score(gt_mask, pred_instances)

                dice_list.append(d)

        val_dice = np.mean(dice_list)

        print(f"\nFold {fold + 1} | Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Dice: {val_dice:.4f}")

        # --------------------
        # CHECKPOINT
        # --------------------
        if val_dice > best_val_dice:

            best_val_dice = val_dice
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, "best_model.pth")
            )

            print("Best model updated")

        else:

            patience_counter += 1

            if patience_counter >= early_stop_patience:
                print("Early stopping triggered")
                break


    # =============================
    # VALIDATION
    # =============================

    fold_mask_dir = os.path.join(save_root,f"fold_{fold+1}","masks")
    fold_vis_dir = os.path.join(save_root,f"fold_{fold+1}","visualizations")

    os.makedirs(fold_mask_dir,exist_ok=True)
    os.makedirs(fold_vis_dir,exist_ok=True)

    model.eval()

    with torch.no_grad():

        for imgs,_,raw_img,gt_mask,name in tqdm(val_loader):

            imgs = imgs.to(device)

            logits = model(imgs)

            prob = torch.sigmoid(logits).cpu().numpy()[0,0]

            pred_instances = watershed_postprocess(prob)

            gt_mask = gt_mask.numpy()[0]

            # save predicted mask
            # cv2.imwrite(
            #     os.path.join(fold_mask_dir,name[0]),
            #     pred_instances.astype(np.uint16)
            # )

            base = os.path.splitext(name[0])[0] + ".tif"

            tiff.imwrite(
                os.path.join(fold_mask_dir, base),
                pred_instances.astype(np.uint16)
            )

            # save visualization
            save_visualization(
                raw_img[0].numpy(),
                gt_mask,
                pred_instances,
                os.path.join(fold_vis_dir,name[0].replace(".png",".png"))
            )

            # metrics
            dice = dice_score(gt_mask,pred_instances)
            pq = pq_score(gt_mask,pred_instances)

            all_results.append({
                "image":name[0],
                "fold":fold+1,
                "dice":dice,
                "pq":pq
            })


# =============================
# SAVE CSV
# =============================

df = pd.DataFrame(all_results)

csv_path = os.path.join(save_root,"metrics.csv")

df.to_csv(csv_path,index=False)

print("\nMetrics saved:",csv_path)


# =============================
# FINAL RESULTS
# =============================

print("\nAverage Dice:",df["dice"].mean())
print("Average PQ:",df["pq"].mean())