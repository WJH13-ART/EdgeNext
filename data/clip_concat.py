import glob

import ipdb
from PIL import Image

def image_concat(image_names):
    """ image_names: list, 存放的是图片的绝对路径 """
    # 1.创建一块背景布
    image = Image.open(image_names[0])
    width, height = image.size
    target_shape = (3*width, 3*height)
    background = Image.new('RGBA', target_shape, (0,0,0,0,))

    # 2.依次将图片放入背景中(注意图片尺寸规整、mode规整、放置位置)
    for ind, image_name in enumerate(image_names):
        img = Image.open(image_name)
        img = img.resize((width, height))  # 尺寸规整
        if img.mode != "RGBA":             # mode规整
            img = img.convert("RGBA")
        row, col = ind//3, ind%3
        location = (col*width, row*height) # 放置位置
        background.paste(img, location)
    background.save("/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/gengdi_Data/concat_res/5.png")

img_dir = "/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/gengdi_Data/toConcat/"
image_names = sorted(glob.glob(img_dir+"*"))
image_concat(image_names)


