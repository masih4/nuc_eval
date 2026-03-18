import os

folder_path_1 = r"C:\Users\amahbod\projects\fulbright\datasets\NuInsSeg_merged\tissue images"

png_names_1 = [f for f in os.listdir(folder_path_1) if f.endswith(".png")]

folder_path_2 = r"C:\Users\amahbod\projects\fulbright\datasets\NuFuseRank\custom_split\NuInsSeg\train"
png_names_2 = [f for f in os.listdir(folder_path_2) if f.endswith(".tif")]


print(png_names_1)



list1 = png_names_1   # 665 png files
list2 = png_names_2   # 651 tif files

# remove extensions
base1 = {os.path.splitext(f)[0] for f in list1}
base2 = {os.path.splitext(f)[0] for f in list2}

# find samples in list1 but not in list2
missing = base1 - base2

print("Number of missing samples:", len(missing))
print("Missing samples:")
print(sorted(missing))