import os
import shutil

old_path = r"/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/lky/diffresult/polytif"
new_path = r"/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/lky/diffresult/polytiff"

files = os.listdir(old_path)
index = 0
#
# for file in files:
#     # if file.find(".shp") != -1:
#     image_files = os.listdir(old_path)
#     old_name = os.path.join(old_path, file)
#     # file1 = "qj_" + file
#     new_name = os.path.join(new_path, file)
#     # shutil.copy(old_name,new_name)
#     new_name2 = old_name.replace("_line.tif", ".tif")
#     # new_name2 = old_name.replace(".init_png", ".tif")
#     os.rename(old_name, new_name2)
#     # print(new_name2)
#     print(old_name)
#     print(new_name2)



image_files = os.listdir(old_path)
for image_file in image_files:

        old_name = os.path.join(old_path, image_file)
        print(old_name)

        new_name = old_name.replace("_poly_line","")
        print(new_name)
        os.rename(old_name, new_name)
        # print(new_name)




# for file in files:
#     # if file.find(".shp") != -1:
#         old_name = os.path.join(old_path, file)
#         file1 = "water_" + file
#         new_name = os.path.join(new_path, file1)
#         os.renam  e(old_name, new_name)
#         print(new_name)