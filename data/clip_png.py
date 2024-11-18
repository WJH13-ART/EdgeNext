import os
import cv2
import numpy as np
from tqdm import tqdm

def crop(img, label, label_g, label_r, save_dirs, save_name,
         crop_size=(50, 50), gap=(50, 50), ratio=0.7, isshow=False):
    h, w, _ = img.shape
    gp_w, gp_h = gap
    cp_w, cp_h = crop_size
    num = 0
    for j in range(0, h, gp_h):
        if j + cp_h > h: continue
        for i in range(0, w, gp_w):
            if i + cp_w > w: continue
            # print(j, i, j*gap_h, j*gap_h+cp_h, i*gap_w, i*gp_w+cp_w)
            cp_img = img[j:j+cp_h, i:i+cp_w, :]
            a_img = label_r[j:j+cp_h, i:i+cp_w]
            b_img = label_g[j:j+cp_h, i:i+cp_w]
            if np.sum(a_img.flatten()) > cp_w * cp_h * 255 * ratio:
                cv2.imwrite(os.path.join(save_dirs[0], save_name.replace('.jpg', f'_{num}.jpg')), cp_img)
                if isshow:
                    cv2.imwrite(os.path.join(save_dirs[0], save_name.replace('.jpg', f'_{num}_show.jpg')), label[j:j+cp_h, i:i+cp_w, :])

            elif np.sum(b_img.flatten()) > cp_w * cp_h * 255 * ratio:
                cv2.imwrite(os.path.join(save_dirs[1], save_name.replace('.jpg', f'_{num}.jpg')), cp_img)
                if isshow:
                    cv2.imwrite(os.path.join(save_dirs[1], save_name.replace('.jpg', f'_{num}_show.jpg')), label[j:j+cp_h, i:i+cp_w, :])
            num += 1

if __name__ == '__main__':
    label_dir = r'/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/gengdi_Data/train/zhenan_test/png//lab'
    img_dir = r'/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/gengdi_Data/train/zhenan_test/png/tif'
    save_img_dir = r'/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/gengdi_Data/train/zhenan_test/320/tif'
    save_label_dir = r'/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/gengdi_Data/train/zhenan_test/320/lab'
    if not os.path.isdir(save_img_dir) : os.makedirs(save_img_dir)
    if not os.path.isdir(save_label_dir) : os.makedirs(save_label_dir)

    crop_w , crop_h = 320,320
    gap_w, gap_h = 100,100
    ratio = 0.7
    for label_name  in tqdm(os.listdir(label_dir)):
        img_path = os.path.join(img_dir,label_name)
        label_path = os.path.join(label_dir,label_name)
        label = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        crop(img,label)
