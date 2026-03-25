
########################################################################################################################
#merging samples from original NuInsSeg dataset (downloaded from kaggle)

# import os
# import shutil
#
# # root directory that contains all folders
# root_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg"
#
# # destination folder where all images will be merged
# output_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\disk10"
#
# os.makedirs(output_dir, exist_ok=True)
#
# for root, dirs, files in os.walk(root_dir):
#
#
#     if os.path.basename(root) == "disk10":
#         for file in files:
#             src = os.path.join(root, file)
#             dst = os.path.join(output_dir, file)
#
#             # avoid overwriting files with the same name
#             base, ext = os.path.splitext(file)
#             counter = 1
#             while os.path.exists(dst):
#                 dst = os.path.join(output_dir, f"{base}_{counter}{ext}")
#                 counter += 1
#
#             shutil.copy2(src, dst)
#
# print("All images merged successfully.")
########################################################################################################################
# creating labeled mask with vague areas removed
# import os
# import numpy as np
# import pandas as pd
# from tifffile import imread, imwrite
# from PIL import Image
#
# # Paths
# label_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\label masks"
# vague_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\mask binary_vague"
# output_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\label masks vague removed"
#
# os.makedirs(output_dir, exist_ok=True)
#
# results = []
#
# for file in os.listdir(label_dir):
#
#     if not file.endswith(".tif"):
#         continue
#
#     name = os.path.splitext(file)[0]
#
#     label_path = os.path.join(label_dir, file)
#     vague_path = os.path.join(vague_dir, name + ".png")
#
#     label = imread(label_path)
#
#     vague = np.array(Image.open(vague_path))
#     vague = vague > 0
#
#     # -------------------------
#     # BEFORE vague removal
#     # -------------------------
#     labels_before = np.unique(label)
#     labels_before = labels_before[labels_before != 0]
#     num_instances_before = len(labels_before)
#
#     total_pixels = label.size
#     instance_pixels_before = np.count_nonzero(label)
#     area_before = instance_pixels_before / total_pixels * 100
#
#     # -------------------------
#     # Remove vague regions
#     # -------------------------
#     modified = label.copy()
#
#     # instances touching vague area
#     touching_instances = np.unique(label[vague])
#     touching_instances = touching_instances[touching_instances != 0]
#
#     # remove them completely
#     for inst in touching_instances:
#         modified[modified == inst] = 0
#
#     # also remove any pixel inside vague mask
#     modified[vague] = 0
#
#     # -------------------------
#     # AFTER vague removal
#     # -------------------------
#     labels_after = np.unique(modified)
#     labels_after = labels_after[labels_after != 0]
#     num_instances_after = len(labels_after)
#
#     instance_pixels_after = np.count_nonzero(modified)
#     area_after = instance_pixels_after / total_pixels * 100
#
#     # save modified mask
#     out_path = os.path.join(output_dir, file)
#     imwrite(out_path, modified.astype(label.dtype))
#
#     results.append([
#         file,
#         num_instances_before,
#         num_instances_after,
#         area_before,
#         area_after
#     ])
#
#
# # -------------------------
# # Save CSV
# # -------------------------
# df = pd.DataFrame(results, columns=[
#     "sample_name",
#     "instances_before",
#     "instances_after",
#     "instance_area_percent_before",
#     "instance_area_percent_after"
# ])
#
# csv_path = os.path.join(output_dir, "instance_statistics_before_after_vague.csv")
# df.to_csv(csv_path, index=False)
#
#
# # -------------------------
# # Dataset statistics
# # -------------------------
# print("\n==============================")
# print("INSTANCES BEFORE VAGUE REMOVAL")
# print("==============================")
#
# print("Min:", df["instances_before"].min())
# print("Max:", df["instances_before"].max())
# print("Mean:", round(df["instances_before"].mean(), 2))
# print("Total:", df["instances_before"].sum())
#
# print("\n==============================")
# print("INSTANCES AFTER VAGUE REMOVAL")
# print("==============================")
#
# print("Min:", df["instances_after"].min())
# print("Max:", df["instances_after"].max())
# print("Mean:", round(df["instances_after"].mean(), 2))
# print("Total:", df["instances_after"].sum())
#
#
# print("\n==============================")
# print("AREA % BEFORE")
# print("==============================")
#
# print("Min:", round(df["instance_area_percent_before"].min(), 2))
# print("Max:", round(df["instance_area_percent_before"].max(), 2))
# print("Mean:", round(df["instance_area_percent_before"].mean(), 2))
#
#
# print("\n==============================")
# print("AREA % AFTER")
# print("==============================")
#
# print("Min:", round(df["instance_area_percent_after"].min(), 2))
# print("Max:", round(df["instance_area_percent_after"].max(), 2))
# print("Mean:", round(df["instance_area_percent_after"].mean(), 2))
#
# print("\nCSV saved to:", csv_path)
# print("Modified masks saved to:", output_dir)
#######################################################################################################################
# # box plots and bar charts
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Path to CSV
# csv_path = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\stats\v1-v3-v3-v4.csv"
#
# # Load CSV
# df = pd.read_csv(csv_path)
#
# # -----------------------------
# # Columns
# # -----------------------------
# instance_cols = [
#     "instances_v1",
#     "instances_v2",
#     "instances_v3",
#     "instances_v4"
# ]
#
# area_cols = [
#     "instance_area_v1",
#     "instance_area_v2",
#     "instance_area_v3",
#     "instance_area_v4"
# ]
#
# labels = ["V1", "V2", "V3", "V4"]
#
#
# # =============================
# # BOX PLOT — INSTANCES
# # =============================
# plt.figure(figsize=(6,5))
#
# plt.boxplot([df[c] for c in instance_cols], labels=labels)
#
# plt.ylabel("Number of Instances", fontsize=16)
# plt.xlabel("Dataset Version", fontsize=16)
# plt.title("Distribution of Instances per Image", fontsize=16)
# plt.tick_params(axis='x', labelsize=14)   # increases v1,v2,v3,v4
# plt.tick_params(axis='y', labelsize=14)   # increases y-axis numbers
#
# plt.tight_layout()
#
# plt.savefig("boxplot_instances.png", dpi=600)
#
# plt.show()
#
#
# # =============================
# # BOX PLOT — AREA
# # =============================
# plt.figure(figsize=(6,5))
#
# plt.boxplot([df[c] for c in area_cols], labels=labels)
#
# plt.ylabel("Instance Area (%)", fontsize=16)
# plt.xlabel("Dataset Version", fontsize=16)
# plt.title("Distribution of Instance Area Percentage", fontsize=16)
# plt.tick_params(axis='x', labelsize=14)   # increases v1,v2,v3,v4
# plt.tick_params(axis='y', labelsize=14)   # increases y-axis numbers
#
# plt.tight_layout()
#
# plt.savefig("boxplot_instance_area.png", dpi=600)
#
# plt.show()
#
#
# # =============================
# # BAR CHART — MEAN INSTANCES
# # =============================
# means_instances = [df[c].mean() for c in instance_cols]
#
# plt.figure(figsize=(6,5))
#
# plt.bar(labels, means_instances)
# plt.ylabel("Mean Number of Instances")
# plt.xlabel("Dataset Version")
# plt.title("Average Instances per Image")
#
# plt.tight_layout()
#
# plt.savefig("barchart_instances_mean.png", dpi=600)
#
# plt.show()
#
#
# # =============================
# # BAR CHART — MEAN AREA
# # =============================
# means_area = [df[c].mean() for c in area_cols]
#
# plt.figure(figsize=(6,5))
#
# plt.bar(labels, means_area)
#
# plt.ylabel("Mean Instance Area (%)")
# plt.xlabel("Dataset Version")
# plt.title("Average Instance Area")
#
# plt.tight_layout()
#
# plt.savefig("barchart_instance_area_mean.png", dpi=600)
#
# plt.show()

#######################################################################################################################
# # # plot PQ, AJI vs. number of instances usin baseline hovernext model
# import os
# import numpy as np
# import pandas as pd
# import cv2
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# # =========================
# # PATHS
# # =========================
# mask_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\label masks modify"
# csv_path = r"C:\Users\amahbod\projects\fulbright\results\results_hovernext_54.17\metrics_results_54.17.csv"  # update if needed
#
# # =========================
# # LOAD CSV
# # =========================
# df = pd.read_csv(csv_path)
#
# # assume column "name" contains file names
# # adjust if needed
# names = df["image"].tolist()
#
#
# min_instances = float("inf")
# max_instances = 0
#
# min_sample = None
# max_sample = None
#
#
# # =========================
# # FUNCTION: count nuclei
# # =========================
# def count_instances(mask):
#     # assuming instance masks: 0=background, 1,2,... = nuclei
#     unique_ids = np.unique(mask)
#     unique_ids = unique_ids[unique_ids != 0]  # remove background
#     return len(unique_ids)
#
#
# # =========================
# # COMPUTE INSTANCE COUNTS
# # =========================
# num_nuclei_list = []
#
# for name in tqdm(names):
#     # handle different extensions if needed
#     base = os.path.splitext(name)[0]
#
#     # try possible formats
#     possible_paths = [
#         os.path.join(mask_dir, base + ".tif"),
#         os.path.join(mask_dir, base + ".tiff"),
#         os.path.join(mask_dir, base + ".png"),
#         os.path.join(mask_dir, base + ".npy"),
#     ]
#
#     mask_path = None
#     for p in possible_paths:
#         if os.path.exists(p):
#             mask_path = p
#             break
#
#     if mask_path is None:
#         print(f"Mask not found for {name}")
#         num_nuclei_list.append(np.nan)
#         continue
#
#     # load mask
#     if mask_path.endswith(".npy"):
#         mask = np.load(mask_path)
#     else:
#         mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#
#     # count nuclei
#     n_inst = count_instances(mask)
#     num_nuclei_list.append(n_inst)
#
#
#     # track min and max
#     if n_inst < min_instances:
#         min_instances = n_inst
#         min_sample = name
#
#     if n_inst > max_instances:
#         max_instances = n_inst
#         max_sample = name
#
# # add to dataframe
# df["num_nuclei"] = num_nuclei_list
#
# # remove missing
# df = df.dropna(subset=["num_nuclei"])
#
#
#
# # =========================
# # OUTPUT PATH
# # =========================
# save_path = r"C:\Users\amahbod\projects\fulbright\results\pq_aji_vs_nuclei.png"
#
# # =========================
# # PLOT
# # =========================
# plt.figure(figsize=(8, 6))
#
# plt.scatter(df["num_nuclei"], df["pq"], color="red", alpha=0.6, label="PQ")
# plt.scatter(df["num_nuclei"], df["aji"], color="blue", alpha=0.6, label="AJI")
#
# plt.xlabel("Number of Nuclei per Image", fontsize=19)
# plt.ylabel("Score (%)", fontsize=19)
# plt.title("PQ and AJI vs Number of Nuclei", fontsize=19)
#
# plt.legend(fontsize=12)
# plt.grid(True)
#
# # =========================
# # SAVE FIGURE (HIGH QUALITY)
# # =========================
# plt.savefig(save_path, dpi=300, bbox_inches="tight")
#
# print(f"Figure saved at: {save_path}")
#
# plt.show()
#
# # =========================
# # SAVE CSV WITH NUM_NUCLEI
# # =========================
# csv_save_path = r"C:\Users\amahbod\projects\fulbright\results\metrics_with_nuclei_count.csv"
# df.to_csv(csv_save_path, index=False)
# print(f"CSV saved at: {csv_save_path}")
#
# print(f"Minimum number of nuclei: {min_instances} (sample: {min_sample})")
# print(f"Maximum number of nuclei: {max_instances} (sample: {max_sample})")
#

######################################################################################################################
# # solve the problem of names in the original nuissseg ImageJZip files
# (_1_ to _01_ and add "roiset" at the end of folder names)
# for other files go to the next section
#
# import os
# import re
#
# base_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\tissue images"
#
# for name in os.listdir(base_dir):
#     full_path = os.path.join(base_dir, name)
#     if not os.path.isdir(full_path):
#         continue
#
#     new_name = name
#
#     # 1) Zero-pad single-digit numbers: _1_ → _01_, _3_ → _03_
#     #    Also handles end of string: _1 → _01
#     new_name = re.sub(r'_(\d)(?=_|$)', lambda m: f'_{m.group(1).zfill(2)}', new_name)
#
#     # # 2) Add _roiset suffix if missing
#     # if not new_name.endswith('_roiset'):
#     #     new_name = new_name + '_roiset'
#
#     # Rename only if something changed
#     if new_name != name:
#         new_path = os.path.join(base_dir, new_name)
#         print(f"Renaming: {name}  →  {new_name}")
#         os.rename(full_path, new_path)
#
# print("Done!")

#######################################################################################################################
# # solve the problem of names in the original nuissseg all files beside imagejzip (_1 to _01 )
# import os
# import re
#
# dir_path = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\weighted_maps_erode"
#
# for name in os.listdir(dir_path):
#     # Match pattern: everything up to the last underscore, then a number (and optional extension)
#     match = re.match(r'^(.+_)(\d+)(\..*)?$', name)
#     if match:
#         prefix, num, ext = match.groups()
#         ext = ext or ""
#         new_name = f"{prefix}{int(num):02d}{ext}"
#         if new_name != name:
#             old_path = os.path.join(dir_path, name)
#             new_path = os.path.join(dir_path, new_name)
#             print(f"Renaming: {name} -> {new_name}")
#             os.rename(old_path, new_path)
#
# print("Done!")
#######################################################################################################################


# import os
#
# dir_1 = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\Imagj_zips"
# dir_2 = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\tissue images"
#
# # Get basenames without extensions and strip "_roiset" suffix for dir_1
# names_1 = set()
# for f in os.listdir(dir_1):
#     name = os.path.splitext(f)[0]
#     if name.endswith("_roiset"):
#         name = name[:-len("_roiset")]
#     names_1.add(name)
#
# names_2 = {os.path.splitext(f)[0] for f in os.listdir(dir_2)}
#
# extra = names_1 - names_2
# print(f"dir_1 has {len(names_1)} files, dir_2 has {len(names_2)} files")
# print(f"Extra in dir_1: {extra}")
#######################################################################################################################
# # # # renaming files with this correction" _1 to _01
# # import os
# # import re
# #
# # folder = r"C:\Users\amahbod\projects\fulbright\results\NuInsSeg_entire\results_cellvit"  # <-- change this
# #
# # for filename in os.listdir(folder):
# #     new_name = re.sub(r'_(\d)(\.\w+)$', r'_0\1\2', filename)
# #     if new_name != filename:
# #         os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))
# #         print(f"{filename} -> {new_name}")
# #######################################################################################################################
# # """
# # Copy 21 selected NuInsSeg samples (tissue images + prediction maps)
# # into a single output folder for further analysis.
# #
# # Group 1 (PQ > 70%, nuclei > 80):   3 samples (all)
# # Group 2 (PQ < 30%, nuclei < 15):  18 samples (all)
# # """
# #
# import os
# import shutil
#
# # ── Selected sample names (without extension) ──────────────────────────
# selected_samples = [
#     # Group 1: PQ > 70%, nuclei > 80 (all 3)
#     "human_cerebellum_11",
#     "human_cerebellum_02",
#     "human_salivory_44",
#     # Group 2: PQ < 30%, nuclei < 15 (all 18)
#     "human_epiglottis_11",
#     "human_lung_02",
#     "human_muscle_02",
#     "human_muscle_05",
#     "human_muscle_09",
#     "human_spleen_04",
#     "human_spleen_23",
#     "human_umbilical_cord_04",
#     "human_umbilical_cord_08",
#     "human_umbilical_cord_09",
#     "mouse_muscle_tibia_12",
#     "mouse_muscle_tibia_18",
#     "mouse_subscapula_01",
#     "mouse_subscapula_05",
#     "mouse_subscapula_13",
#     "mouse_subscapula_25",
#     "mouse_subscapula_26",
#     "mouse_subscapula_38",
# ]
#
# # ── Paths ───────────────────────────────────────────────────────────────
# tissue_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\tissue images"
# npy_dir    = r"C:\Users\amahbod\projects\fulbright\results\NuInsSeg_entire\cellvit\preds_cellvit_56.44\nuinsseg"
# output_dir = r"C:\Users\amahbod\projects\fulbright\results\NuInsSeg_sub\cellvit\sub_NuInsSeg_data_cellvit_preds"
#
# # ── Create output directory ─────────────────────────────────────────────
# os.makedirs(output_dir, exist_ok=True)
#
# # ── Copy files ──────────────────────────────────────────────────────────
# copied = 0
# missing = []
#
# for name in selected_samples:
#     png_file = os.path.join(tissue_dir, f"{name}.png")
#     npy_file = os.path.join(npy_dir, f"{name}.npy")
#
#     for src in [png_file, npy_file]:
#         if os.path.isfile(src):
#             shutil.copy2(src, output_dir)
#             copied += 1
#             print(f"  Copied: {os.path.basename(src)}")
#         else:
#             missing.append(src)
#             print(f"  NOT FOUND: {src}")
#
# # ── Summary ─────────────────────────────────────────────────────────────
# print(f"\nDone! Copied {copied} files to:\n  {output_dir}")
# if missing:
#     print(f"\n{len(missing)} file(s) not found:")
#     for m in missing:
#         print(f"  - {m}")

#######################################################################################################################
# # nuber of instnces per image and save them in csv file
# import os
# import numpy as np
# import pandas as pd
# import tifffile as tiff
# from tqdm import tqdm
#
# # Path to label masks
# mask_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\label masks modify"
#
# # Output CSV file
# output_csv = r"C:\Users\amahbod\projects\fulbright\results\results_data_analysis\instance_counts.csv"
#
# results = []
#
# # Get all .tif / .tiff files
# mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith((".tif", ".tiff"))]
#
# for fname in tqdm(mask_files):
#     path = os.path.join(mask_dir, fname)
#
#     # Read mask
#     mask = tiff.imread(path)
#
#     # Unique labels
#     unique_ids = np.unique(mask)
#
#     # Remove background (0)
#     instance_ids = unique_ids[unique_ids != 0]
#
#     # Count instances
#     num_instances = len(instance_ids)
#
#     results.append({
#         "filename": fname,
#         "num_instances": num_instances
#     })
#
# # Save to CSV
# df = pd.DataFrame(results)
# df.to_csv(output_csv, index=False)
#
# print(f"Saved CSV to: {output_csv}")



#######################################################################################################################
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Categories
# metrics = ["PQ", "AJI", "DICE", "DQ", "SQ"]
# labels = ["baseline", "#1", "#1,2", "#1,2,3", "#1,2,3,4"]
#
# # -------------------------
# # Data
# # -------------------------
#
# hovernet = np.array([
#     [53.40, 56.17, 57.05, 58.18, 64.07],
#     [57.50, 60.75, 59.63, 60.77, 64.91],
#     [80.29, 81.31, 83.73, 83.75, 86.60],
#     [70.38, 73.28, 74.15, 76.61, 78.64],
#     [75.17, 75.82, 76.64, 75.68, 81.24],
# ])
#
# hovernext = np.array([
#     [54.17, 56.80, 58.29, 58.45, 64.92],
#     [58.45, 61.06, 61.52, 61.75, 66.41],
#     [82.59, 83.81, 85.73, 85.66, 88.76],
#     [72.14, 74.83, 76.51, 77.35, 79.90],
#     [74.40, 75.17, 75.88, 75.31, 81.00],
# ])
#
# cellvit = np.array([
#     [56.44, 59.72, 60.92, 61.93, 68.44],
#     [60.03, 63.31, 63.08, 64.02, 68.87],
#     [81.92, 83.17, 85.32, 85.25, 88.40],
#     [74.50, 77.95, 79.33, 81.12, 83.27],
#     [74.83, 75.58, 76.51, 76.08, 81.96],
# ])
#
# # -------------------------
# # Plot function
# # -------------------------
#
# def plot_model(data, title, save_name):
#     x = np.arange(len(metrics))
#     width = 0.15
#
#     plt.figure(figsize=(10, 6))
#
#     for i in range(data.shape[1]):
#         plt.bar(x + i * width, data[:, i], width, label=labels[i])
#
#     plt.xticks(x + width * 2, metrics, fontsize=12)
#     plt.ylabel("Score (%)", fontsize=12)
#     plt.title(title, fontsize=14)
#     plt.legend()
#     plt.grid(axis="y", linestyle="--", alpha=0.5)
#
#     plt.tight_layout()
#     plt.savefig(save_name, dpi=300)
#     plt.close()
#
#
# # -------------------------
# # Generate plots
# # -------------------------
#
# plot_model(hovernet, "Hover-Net Performance", "hovernet_bar.png")
# plot_model(hovernext, "Hover-Next Performance", "hovernext_bar.png")
# plot_model(cellvit, "CellViT Performance", "cellvit_bar.png")

#######################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Data
# -------------------------
labels = ["baseline", "#1", "#1,2", "#1,2,3", "#1,2,3,4"]

# entire NuInsSeg
hovernet  = [53.40, 56.17, 57.05, 58.18, 64.07]
hovernext = [54.17, 56.80, 58.29, 58.45, 64.92]
cellvit   = [56.44, 59.72, 60.92, 61.93, 68.44]

# # sub NuInsSeg
# hovernet  = [28.68, 40.03, 60.21, 60.90, 67.13]
# hovernext = [28.00, 40.35, 64.95, 64.18, 70.61]
# cellvit   = [29.07, 43.45, 65.66, 66.15, 72.26]

models = ["Hover-Net", "Hover-Next", "CellViT"]
data = [hovernet, hovernext, cellvit]

# -------------------------
# Colors (consistent across models)
# -------------------------
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# -------------------------
# Layout settings
# -------------------------
n_models = len(models)
n_bars = len(labels)

bar_width = 0.15
group_spacing = 0.6

# Compute x positions
x_positions = []
for i in range(n_models):
    start = i * (n_bars * bar_width + group_spacing)
    x_positions.append([start + j * bar_width for j in range(n_bars)])

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(10, 6))

for i, model_data in enumerate(data):
    for j, value in enumerate(model_data):
        plt.bar(
            x_positions[i][j],
            value,
            bar_width,
            color=colors[j]
        )

# X-axis: model names centered under each group
group_centers = [np.mean(pos) for pos in x_positions]
plt.xticks(group_centers, models, fontsize=16)

# Labels and title
plt.ylabel("PQ (%)", fontsize=16)
#plt.title("PQ Comparison Across Models and Modifications", fontsize=16)

# Y-axis start from 40
plt.ylim(25, 80)

# Grid
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Legend (modifications)
handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(n_bars)
]
#plt.legend(handles, labels, title="Modifications")
plt.legend(
    handles,
    labels,
    title="Modifications",
    fontsize=13,
    title_fontsize=14
)

# Layout & save
plt.tight_layout()
plt.savefig("pq_grouped_models.png", dpi=300)
plt.show()

