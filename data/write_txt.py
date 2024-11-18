import os

# image_path = r"Z:\wjh\dataset_9.26\tif"
# poly_path = r"Z:\wjh\dataset_9.26\label"
# line_path = r'Z:\Project_Data\Suzhou\0810\samples\gengdi1_labels'
# txt_path = r"Z:\wjh\dataset_9.26\train_line.txt"

image_path = r"/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/chongzuo/tif"
poly_path = r"/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/chongzuo/line"
# line_path = r'Z:\Project_Data\Suzhou\0810\samples\gengdi1_labels'
txt_path = r"/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/SegNeXt-main/data/gengdi/ImageSets/train_chongzuo.txt"

# 写训练集
f = open(txt_path, "w")
files = os.listdir(image_path)
print(files)
for file in files:
    if file[-4:] == ".tif":
        # if file.find(".tif") != -1 and file.find("tif.") == -1:
        image_name = os.path.join(image_path, file)
        poly_name = os.path.join(poly_path, file)
        print(image_name)
        print(poly_name)
        # line_name = os.path.join(line_path, file)
        # f.write(image_name + "\n")
        f.write(image_name + " " + poly_name + "\n")
        # f.write(image_name + " " + line_name + "\n")
        # f.write(image_name + " " + poly_name + " " + line_name + "\n")

f.close()

# # 写测试集
# f = open(txt_path, "w")
# files = os.listdir(image_path)
# for file in files:
#     if file.find(".tif") != -1 and file.find("tif.") == -1:
#         image_name = os.path.join(image_path, file)
#         f.write(image_name + " " + "\n")
#
# f.close()
