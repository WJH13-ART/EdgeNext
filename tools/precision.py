import cv2
import ipdb
import numpy as np
import os

# 预测结果路径
pred_path = r'Z:\WJH\edge_res\SEG_ZHEJIANG\zhenan+zhebei\exe_res_tif'
#标签路径
lab_path = r'Z:\WJH\gengdi_Data\test_zhejiang\mask_tif'
# Precision
txt_path = r'Z:\WJH\Seg_BIoU.txt'

#预测结果路径
# pred_path = r'D:\hxm\res\beijing\BDCN\90polyline'
# #标签路径
# lab_path = r'Z:\hxm\train\0417\test_line'
# # Precision
# txt_path = r'D:\hxm\res\beijing\BDCN\90pbiou.txt'

def count(imgp,imgl):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgp[i, j] == 255 and imgl[i, j] == 255:
                tp = tp+1
            elif imgl[i,j] == 255 and imgp[i,j] == 0:
                fn = fn+1
            elif imgl[i, j] == 0 and imgp[i, j] == 255:
                fp = fp+1
            elif imgl[i, j] == 0 and imgp[i, j] == 0:
                tn = tn+1
    return tp, fn, fp, tn

# def tpcount(imgp,imgl):
#     n = 0
#     for i in range(WIDTH):
#         for j in range(HIGTH):
#             if imgp[i,j] == 255 and imgl[i,j] == 255:
#                 n = n+1
#     return n

# def fncount (imgp,imgl):
#     n = 0
#     for i in range(WIDTH):
#         for j in range(HIGTH):
#             if imgl[i,j] == 255 and imgp[i,j] == 0:
#                 n = n+1
#     return n
#
# def fpcount(imgp,imgl):
#     n = 0
#     for i in range(WIDTH):
#         for j in range(HIGTH):
#             if imgl[i,j] == 0 and imgp[i,j] == 255:
#                 n+=1
#     return n
#
# def tncount(imgp,imgl):
#     n=0
#     for i in range(WIDTH):
#         for j in range(HIGTH):
#             if imgl[i,j] == 0 and imgp[i,j] == 0:
#                 n += 1
#     return n




imgs = os.listdir(pred_path)
a = len(imgs)
TP = 0
FN = 0
FP = 0
TN = 0
c = 0
ker = [3, 5, 7]
for name in imgs:
    if name[-4:] == ".tif":
        imgp = cv2.imread(pred_path + '\\' + name, -1)
        imgp = np.array(imgp)

        imgl = cv2.imread(lab_path + '\\' + name, -1)
        imgl = np.array(imgl)
        ipdb.set_trace()

        # kernel3 = np.ones((ker[0], ker[0]), np.uint8)#膨胀
        # imgp = cv2.dilate(imgp, kernel3)
        # imgl = cv2.dilate(imgl, kernel3)

        kernel5 = np.ones((ker[1], ker[1]), np.uint8)  # 膨胀
        imgp = cv2.dilate(imgp, kernel5)
        imgl = cv2.dilate(imgl, kernel5)

        # kernel7 = np.ones((ker[2], ker[2]), np.uint8)  # 膨胀
        # imgp = cv2.dilate(imgp, kernel7)
        # imgl = cv2.dilate(imgl, kernel7)

        WIDTH = imgl.shape[0]
        HIGTH = imgl.shape[1]
        ipdb.set_trace()
        tp, fn, fp, tn = count(imgp, imgl)
        TP += tp
        FN += fn
        FP += fp
        TN += tn

        c += 1
        print('已经计算：'+str(c) + ',剩余数目：'+str(a-c))

print('TP:'+str(TP))
print('FN:'+str(FN))
print('FP:'+str(FP))
print('TN:'+str(TN))


#准确率
# zq = (int(TN)+int(TP))/(int(WIDTH)*int(HIGTH)*int(len(imgs)))
zq = (int(TN)+int(TP))/(int(TP)+int(TN)+int(FP)+int(FN))
#精确率
jq = int(TP)/(int(TP)+int(FP))
#召回率
zh = int(TP)/(int(TP)+int(FN))
#F1
f1 = int(TP)*2/(int(TP)*2+int(FN)+int(FP))
#IOU
IOU = int(TP)/(int(TP)+int(FP)+int(FN))

print('Accuracy：'+ str(zq))
print('Precision：'+ str(jq))
print('Recall：'+ str(zh))
print('F1：'+ str(f1))
print('IOU：'+ str(IOU))

txt = open(txt_path, "w")
txt.write("Accuracy :" + str(zq) + "\n")
txt.write("Precision :" + str(jq) + "\n")
txt.write("Recall :" + str(zh) + "\n")
txt.write("F1 :" + str(f1) + "\n")
txt.write("IOU :" + str(IOU) + "\n")
