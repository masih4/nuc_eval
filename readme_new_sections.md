### Synthetic example

The [examples/synthetic](./examples/synthetic) folder contains a small synthetic image with four circular instances. Each instance isolates the effect of one modification, as described in the paper:

| Instance | Setup | Modification shown |
|----------|-------|--------------------|
| 1 | Lies fully within an ambiguous region; GT and prediction do not overlap | #1 (ambiguous regions) |
| 2 and 3 | Overlap; the shared region is assigned to opposite instances in GT and prediction | #3 (overlapping instances) |
| 4 | Small circle whose prediction differs from the GT by a one-pixel boundary | #4 (border uncertainty) |

Modification #2 (score normalization) is not included, as it is an aggregation step across images and does not apply to a single image.

The folder contains the merged GT label map (`gt_labelmap.tif`), the overlap-preserving GT as one binary mask per instance (`Overlap-preserving GT/inst_*.tif`), the prediction (`pred_labelmap.tif`) and the ambiguous mask (`amb_mask.png`).

Run it from the repository root:

```bash
python examples/synthetic/run_synthetic_eval.py
```

Expected output (values in %, as reported in the paper):

```
Using MERGED label-map GT (cannot show Mod #3 — overlaps are merged)
Baseline                         PQ= 63.33  AJI= 71.41  Dice= 91.53  DQ= 75.00  SQ= 84.45
Modification #1 (vague)          PQ= 84.45  AJI= 83.08  Dice= 99.70  DQ=100.00  SQ= 84.45
Modification #4 (boundary)       PQ= 66.55  AJI= 72.04  Dice= 91.85  DQ= 75.00  SQ= 88.74

Using OVERLAP-PRESERVING GT (list of masks) — enables Mod #3
Baseline + Modification #3       PQ= 67.62  AJI= 78.44  Dice= 91.53  DQ= 75.00  SQ= 90.16
Modification #1 + #3             PQ= 90.16  AJI= 91.26  Dice= 99.70  DQ=100.00  SQ= 90.16
Modification #1 + #3 + #4        PQ= 94.61  AJI= 92.25  Dice=100.00  DQ=100.00  SQ= 94.61
```

Even with all applicable modifications, PQ, AJI and SQ do not reach 100%. This is expected: the prediction assigns the overlapping region to a single instance, which is a genuine difference from the overlap-preserving GT rather than an evaluation artifact.

---

## Datasets

We used four public datasets in this study. For NuInsSeg, CryoNuSeg and PCNS, please download the data from the original publications:

| Dataset | Description | Reference |
|---------|-------------|-----------|
| **NuInsSeg** | H&E images from 31 human and mouse organs, with ambiguous region masks and ImageJ ROI files (overlap-preserving) | Mahbod et al., *Scientific Data*, 2024. [doi:10.1038/s41597-024-03117-2](https://doi.org/10.1038/s41597-024-03117-2) |
| **CryoNuSeg** | 30 cryosectioned H&E images from 10 human organs, annotated by multiple annotators | Mahbod et al., *Computers in Biology and Medicine*, 2021. [doi:10.1016/j.compbiomed.2021.104349](https://doi.org/10.1016/j.compbiomed.2021.104349) |
| **PCNS** | 1,356 H&E images (256x256) from ten cancer types | Hou et al., *Scientific Data*, 2020. [doi:10.1038/s41597-020-0528-1](https://doi.org/10.1038/s41597-020-0528-1) |
| **MoNuSAC** | Multi-organ nuclei segmentation and classification challenge data | Verma et al., *IEEE Transactions on Medical Imaging*, 2021. [doi:10.1109/TMI.2021.3085712](https://doi.org/10.1109/TMI.2021.3085712) |

### Revised MoNuSAC test data

For the ranking experiment in the paper, we converted the MoNuSAC test set into class-agnostic instance label maps (all nucleus classes merged) with the corresponding ambiguous region masks. The revised data are available here:

[Google Drive: revised MoNuSAC test data](https://drive.google.com/drive/folders/1UvXm-1I8g7zqEdxqd0-xfdIn6LG0Z-mL?usp=sharing)

If you use these data, please also cite the original MoNuSAC publication (Verma et al., 2021).
