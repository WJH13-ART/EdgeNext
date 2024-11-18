# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 下午6:41

import cv2
import numpy as np
import os

def tif_to_png(image_path,save_path):
    """
    :param image_path: *.tif image path
    :param save_path: *.png image path
    :return:
    """
    img = cv2.imread(image_path, -1)
    if(img is None):
        print("null")
        return
    # print(img)
    # print(img.dtype)
    # filename = image_path.split('/')[-1].split('_')[0]
    # print(filename)
    # save_path = save_path + '/' + filename + '.png'
    cv2.imwrite(save_path, img)

if __name__ == '__main__':
    # root_path = r'/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/yangdz/data/yangyuan/test_img_3band/'
    # save_path = r'/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/yangdz/data/yangyuan/test_img_png/'
    root_path = r'/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/DiffusionEdge-main/data_root/youyang/edge/'
    save_path = r'/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/DiffusionEdge-main/data_root/youyang/edge/'
    image_files = os.listdir(root_path)
    print(image_files[1][:-4])
    for image_file in image_files:
        tif_to_png(root_path + image_file,save_path+image_file[:-4]+'.png')
