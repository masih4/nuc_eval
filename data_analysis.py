
########################################################################################################################
#merging samples from original NuInsSeg dataset (downloaded from kaggle)

# import os
# import shutil
#
# # root directory that contains all folders
# root_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg"
#
# # destination folder where all images will be merged
# output_dir = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\overlay_save_path_vague"
#
# os.makedirs(output_dir, exist_ok=True)
#
# for root, dirs, files in os.walk(root_dir):
#
#
#     if os.path.basename(root) == "overlay_save_path_vague":
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
# box plots and bar charts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to CSV
csv_path = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\stats\v1-v3-v3-v4.csv"

# Load CSV
df = pd.read_csv(csv_path)

# -----------------------------
# Columns
# -----------------------------
instance_cols = [
    "instances_v1",
    "instances_v2",
    "instances_v3",
    "instances_v4"
]

area_cols = [
    "instance_area_v1",
    "instance_area_v2",
    "instance_area_v3",
    "instance_area_v4"
]

labels = ["V1", "V2", "V3", "V4"]


# =============================
# BOX PLOT — INSTANCES
# =============================
plt.figure(figsize=(6,5))

plt.boxplot([df[c] for c in instance_cols], labels=labels)

plt.ylabel("Number of Instances", fontsize=16)
plt.xlabel("Dataset Version", fontsize=16)
plt.title("Distribution of Instances per Image", fontsize=16)
plt.tick_params(axis='x', labelsize=14)   # increases v1,v2,v3,v4
plt.tick_params(axis='y', labelsize=14)   # increases y-axis numbers

plt.tight_layout()

plt.savefig("boxplot_instances.png", dpi=600)

plt.show()


# =============================
# BOX PLOT — AREA
# =============================
plt.figure(figsize=(6,5))

plt.boxplot([df[c] for c in area_cols], labels=labels)

plt.ylabel("Instance Area (%)", fontsize=16)
plt.xlabel("Dataset Version", fontsize=16)
plt.title("Distribution of Instance Area Percentage", fontsize=16)
plt.tick_params(axis='x', labelsize=14)   # increases v1,v2,v3,v4
plt.tick_params(axis='y', labelsize=14)   # increases y-axis numbers

plt.tight_layout()

plt.savefig("boxplot_instance_area.png", dpi=600)

plt.show()


# =============================
# BAR CHART — MEAN INSTANCES
# =============================
means_instances = [df[c].mean() for c in instance_cols]

plt.figure(figsize=(6,5))

plt.bar(labels, means_instances)
plt.ylabel("Mean Number of Instances")
plt.xlabel("Dataset Version")
plt.title("Average Instances per Image")

plt.tight_layout()

plt.savefig("barchart_instances_mean.png", dpi=600)

plt.show()


# =============================
# BAR CHART — MEAN AREA
# =============================
means_area = [df[c].mean() for c in area_cols]

plt.figure(figsize=(6,5))

plt.bar(labels, means_area)

plt.ylabel("Mean Instance Area (%)")
plt.xlabel("Dataset Version")
plt.title("Average Instance Area")

plt.tight_layout()

plt.savefig("barchart_instance_area_mean.png", dpi=600)

plt.show()