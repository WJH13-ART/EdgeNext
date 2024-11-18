import torch
import torch.nn as nn
import torch.utils.data as data
from cv2 import cv2
from skimage.morphology import skeletonize
from torch.autograd import Variable as V
from torch.nn import functional as F
# from mylogger import logger
# import config
import os, sys
import numpy as np
# import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from osgeo import gdal, ogr, osr
import imageio
import ipdb
from metric_sem import MeanIoUMetric
# from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool, my_DinkNet34, \
#     my_DinkNet34_3plus, my_DinkNet34_3plus_first7, my_DinkNet34_3plus_first7_new,HEDinkNet34_41,\
#     my_DinkNet34_3plus_Edge, my_DinkNet34_3plus_Edge_bot2top

# from dinknet import HEDinkNetBatch, HEDinkNetGroup, HEDinkNet34_6
# from networks.RCF_models import RCF
# gFlags = config.parse()
cards = list(map(int, [0]))


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=cards)
        self.net = self.net.eval()
        # self.x_width = config.Flags.predict_crop
        self.x_width = 320 # 960
        # self.pad = config.Flags.predict_buf
        self.pad = 32

    def block_gdal_input(self, img, img_size, crop, pad=0):
        [img_width, img_height] = img_size
        x_height = self.x_width
        x_width = self.x_width
        crop_width = self.x_width - 2 * pad
        crop_height = x_height - 2 * pad

        numBand = 3
        numBand = img.RasterCount
        num_Xblock = img_width // crop_width
        x_start, x_end = [], []
        x_start.append(0)
        for i in range(num_Xblock):
            xs = crop_width * (i + 1) - pad
            xe = crop_width * i + x_width - pad
            if (i == num_Xblock - 1):
                xs = img_width - crop_width - pad
                xe = min(xe, img_width)
            x_start.append(xs)
            x_end.append(xe)
        x_end.append(img_width)

        num_Yblock = img_height // crop_height
        y_start, y_end = [], []
        y_start.append(0)
        for i in range(num_Yblock):
            ys = crop_height * (i + 1) - pad
            ye = crop_height * i + x_height - pad
            if (i == num_Yblock - 1):
                ys = img_height - crop_height - pad
                ye = min(ye, img_height)
            y_start.append(ys)
            y_end.append(ye)
        y_end.append(img_height)

        if img_width % crop_width > 0:
            num_Xblock = num_Xblock + 1
        if img_height % crop_height > 0:
            num_Yblock = num_Yblock + 1
        for i in range(num_Yblock):
            for j in range(num_Xblock):
                [x0, x1, y0, y1] = [x_start[j], x_end[j], y_start[i], y_end[i]]

                feature = np.zeros(np.append([y1 - y0, x1 - x0], numBand), np.float32)
                for ii in range(numBand):
                    floatData = np.array(img.GetRasterBand(ii + 1).ReadAsArray(x0, y0, x1 - x0, y1 - y0))
                    # floatData = np.array(img.GetRasterBand(4-ii).ReadAsArray(x0,y0,x1-x0,y1-y0))
                    feature[..., ii] = floatData

                if (i == 0):
                    feature_pad = cv2.copyMakeBorder(feature,
                                                     pad, x_height - pad - feature.shape[0],
                                                     0, 0, cv2.BORDER_REFLECT_101)
                else:
                    feature_pad = cv2.copyMakeBorder(feature,
                                                     0, x_height - feature.shape[0],
                                                     0, 0, cv2.BORDER_REFLECT_101)
                if (j == 0):
                    feature_pad = cv2.copyMakeBorder(feature_pad,
                                                     0, 0, pad, x_width - pad - feature_pad.shape[1],
                                                     cv2.BORDER_REFLECT_101)
                else:
                    feature_pad = cv2.copyMakeBorder(feature_pad,
                                                     0, 0, 0, x_width - feature_pad.shape[1],
                                                     cv2.BORDER_REFLECT_101)

                yield feature_pad, [x0, x1, y0, y1]

    def pred_gdal_blocks_write_multiclass(self, img_path, out_path='', repeats=4):
        self.num_classes = 8
        # logger.info('predicting %s' % img_path)
        self.net.eval()
        datasetname = gdal.Open(img_path, gdal.GA_ReadOnly)
        # datasetname = reproject_dataset(img_path,5500,5500)
        if datasetname is None:
            print('Could not open %s' % img_path)
        img_width = datasetname.RasterXSize
        img_height = datasetname.RasterYSize
        imageSize = [img_width, img_height]
        nBand = datasetname.RasterCount

        if out_path == '':
            out_path = img_path.rsplit('.', 1)[0] + '_res.tif'
        outRaster = None
        mask = None
        bigImg = False
        temp_path = out_path.rsplit('.', 1)[0] + '_temp.tif'
        if img_height * img_width > 9e8:
            # outRaster = gdal.Open(out_path,1)
            outRaster = gdal.GetDriverByName('GTiff').Create(out_path, img_width, img_height, 1, gdal.GDT_Byte)
            outRaster.SetGeoTransform(datasetname.GetGeoTransform())
            outRaster.SetProjection(datasetname.GetProjection())
            bigImg = True
            repeats = 1
        else:
            # mask = np.zeros([repeats, self.num_classes, img_height, img_width], np.float)
            mask = np.zeros([repeats, img_height, img_width], dtype=np.float32)

        crops = [self.x_width + 64, self.x_width, self.x_width - 64, self.x_width - 128]
        steps = [[1, 1], [-1, -1], [-1, 1], [1, -1]]
        for k in range(repeats):
            pad = self.pad
            crop_data = self.x_width
            crop_width = crop_data - 2 * pad
            crop_height = crop_data - 2 * pad
            num_Xblock = img_width // crop_width
            if img_width % crop_width > 0:
                num_Xblock += 1
            num_Yblock = img_height // crop_height
            if img_height % crop_height > 0:
                num_Yblock += 1
            i = 0
            blocks = num_Xblock * num_Yblock
            # mask = np.zeros([batch_size, img_height, img_width],dtype=np.float32)
            input_gen = self.block_gdal_input(datasetname, imageSize, crop_data, pad)
            for i in tqdm(range(blocks)):
                img, xy = next(input_gen)
                if xy[0] > 0:
                    xs = xy[0] + pad
                else:
                    xs = xy[0]

                if xy[2] > 0:
                    ys = xy[2] + pad
                else:
                    ys = xy[2]
                if np.max(img[pad: pad + crop_height, pad: pad + crop_width]) < 5:
                    print("错误")
                    continue
                    prediction = np.zeros([1, self.x_height, self.x_width])
                else:
                    with torch.no_grad():
                        [x_step, y_step] = steps[k]
                        img_tta = img[::x_step, ::y_step, :].transpose(2, 0, 1) / 255.0 * 3.2 - 1.6

                        imgs = []
                        imgs.append(img_tta)
                        imgs = np.array(imgs)
                        # imgs = imgs.transpose(0, 3, 1, 2)
                        # print(imgs.shape)
                        imgs = V(torch.Tensor(np.array(imgs, np.float32)).cuda())
                        # predictions = self.net.forward(imgs)
                        # predictions = predictions[-1].squeeze().cpu().data.numpy()
                        # 多分类
                        res = self.net.forward(imgs)
                        res = F.softmax(res[-1], dim=1)  # 1 6 256 256
                        # res = F.softmax(res, dim=1)  # 1 6 256 256
                        # print(res.shape)
                        predictions = torch.argmax(res, dim=1)  # 1 1 256 256
                        predictions = predictions.squeeze().cpu().data.numpy()  # 1 256 256
                        if bigImg:
                            # predictions = np.asarray(np.argmax(predictions, axis=0), dtype=np.uint8)
                            prediction = predictions[pad: pad + crop_height, pad: pad + crop_width]
                            prediction = prediction[::x_step, ::y_step]
                        else:
                            # prediction = predictions[..., pad: pad + crop_height, pad: pad + crop_width]
                            # prediction = prediction[..., ::x_step, ::y_step]
                            prediction = predictions[pad: pad + crop_height, pad: pad + crop_width]
                            prediction = prediction[::x_step, ::y_step]

                    # print(prediction.shape)

                if bigImg:
                    outRaster.GetRasterBand(k + 1).WriteArray(prediction.astype(np.float32), xs, ys)
                else:
                    # mask[k, :, ys: ys + crop_height, xs: xs + crop_width] = prediction.astype(np.float32)
                    mask[k, ys: ys + crop_height, xs: xs + crop_width] = prediction.astype(np.float32)

                    # datasetname = None
        if not bigImg:
            outRaster = gdal.GetDriverByName('GTiff').Create(out_path, img_width, img_height, 1, gdal.GDT_Byte)
            outRaster.SetGeoTransform(datasetname.GetGeoTransform())
            outRaster.SetProjection(datasetname.GetProjection())
            # output = np.mean(mask, axis=0)
            output = np.squeeze(np.mean(mask, axis=0)).astype(np.int)
            # print(output.shape)
            # output = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
            outRaster.GetRasterBand(1).WriteArray(output)

        outRaster = None

        return  # np.squeeze(mask*255).astype(np.int)

    def pred_gdal_blocks_write_multi(self, img_path, out_path='', repeats=4):
        # logger.info('predicting %s' % img_path)
        self.net.eval()
        datasetname = gdal.Open(img_path, gdal.GA_ReadOnly)
        # datasetname = reproject_dataset(img_path,5500,5500)
        if datasetname is None:
            print('Could not open %s' % img_path)
        img_width = datasetname.RasterXSize
        img_height = datasetname.RasterYSize
        imageSize = [img_width, img_height]
        nBand = datasetname.RasterCount

        if out_path == '':
            out_path = img_path.rsplit('.', 1)[0] + '_res.tif'
        outRaster = None
        mask = None
        bigImg = False
        temp_path = out_path.rsplit('.', 1)[0] + '_temp.tif'
        if img_height * img_width > 9e8:
            # outRaster = gdal.Open(out_path,1)
            tempRaster = gdal.GetDriverByName('GTiff').Create(temp_path, img_width, img_height, 4, gdal.GDT_Float32)
            bigImg = True
        else:
            mask = np.zeros([repeats, img_height, img_width], float)

        crops = [self.x_width + 128, self.x_width, self.x_width - 64, self.x_width - 128]
        steps = [[1, 1], [-1, -1], [-1, 1], [1, -1]]
        for k in range(repeats):
            # crop_data = crops[k]
            # pad = crop_data // 16
            pad = self.pad
            crop_data = self.x_width
            crop_width = crop_data - 2 * pad
            crop_height = crop_data - 2 * pad
            num_Xblock = img_width // crop_width
            if img_width % crop_width > 0:
                num_Xblock += 1
            num_Yblock = img_height // crop_height
            if img_height % crop_height > 0:
                num_Yblock += 1
            i = 0
            blocks = num_Xblock * num_Yblock
            # mask = np.zeros([batch_size, img_height, img_width],dtype=np.float32)
            input_gen = self.block_gdal_input(datasetname, imageSize, crop_data, pad)
            for i in tqdm(range(blocks)):
                img, xy = next(input_gen)
                if (xy[0] > 0):
                    xs = xy[0] + pad
                else:
                    xs = xy[0]

                if (xy[2] > 0):
                    ys = xy[2] + pad
                else:
                    ys = xy[2]
                if np.max(img[pad: pad + crop_height, pad: pad + crop_width]) < 5:
                    continue
                    prediction = np.zeros([1, self.x_height, self.x_width])
                else:
                    with torch.no_grad():
                        [x_step, y_step] = steps[k]
                        img_tta = img[::x_step, ::y_step, :]
                        imgs = []
                        imgs.append(img_tta)
                        imgs = np.array(imgs)
                        imgs = imgs.transpose(0, 3, 1, 2) / 255.0 * 3.2 - 1.6
                        # print(imgs.shape)
                        imgs = V(torch.Tensor(np.array(imgs, np.float32)).cuda())
                        # prediction = self.net.forward(imgs).squeeze().cpu().data.numpy()
                        # res,line = self.net.forward(imgs)
                        # prediction = line.squeeze().cpu().data.numpy()
                        res = self.net.forward(imgs)

                        prediction = res[-1].squeeze().cpu().data.numpy()
                        prediction = prediction[pad: pad + crop_height, pad: pad + crop_width]
                        prediction = prediction[::x_step, ::y_step]

                if bigImg:
                    # print(bigImg.shape)
                    tempRaster.GetRasterBand(k + 1).WriteArray(prediction.astype(np.float32), xs, ys)
                else:
                    # print(prediction.shape)
                    mask[k, ys: ys + crop_height, xs: xs + crop_width] = prediction.astype(np.float32)

        outRaster = gdal.GetDriverByName('GTiff').Create(out_path, img_width, img_height, 1, gdal.GDT_Byte)
        outRaster.SetGeoTransform(datasetname.GetGeoTransform())
        outRaster.SetProjection(datasetname.GetProjection())
        datasetname = None
        if bigImg:
            rows = int(4e8 // img_width)
            numRows = img_height // rows
            mask = np.zeros([repeats, rows, img_width])
            for i in tqdm(range(numRows)):
                for j in range(repeats):
                    mask[j, ...] = np.array(tempRaster.GetRasterBand(j + 1).ReadAsArray(0, i * rows, img_width, rows))
                outRaster.GetRasterBand(1).WriteArray((np.mean(mask, axis=0) * 255).astype(np.int), 0, i * rows)

            leftrows = int(img_height - numRows * rows)
            mask1 = np.zeros([repeats, leftrows, img_width])
            for j in range(repeats):
                mask1[j, ...] = np.array(
                    tempRaster.GetRasterBand(j + 1).ReadAsArray(0, numRows * rows, img_width, leftrows))
            outRaster.GetRasterBand(1).WriteArray((np.mean(mask1, axis=0) * 255).astype(np.int), 0, numRows * rows)

            tempRaster = None
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            outRaster.GetRasterBand(1).WriteArray((np.mean(mask, axis=0) * 255).astype(int))

        outRaster = None
        return  np.squeeze(mask*255).astype(np.int)
        # return np.squeeze(np.mean(mask, axis=0)).astype(int)

    def pred_gdal_blocks(self, img_path):
        # logger.info('predicting %s' % img_path)
        self.net.eval()
        batch_size = 4
        pad = self.pad
        x_height = x_width = self.x_width
        crop_width = x_width - 2 * pad
        crop_height = x_height - 2 * pad
        datasetname = gdal.Open(img_path, gdal.GA_ReadOnly)
        # datasetname = reproject_dataset(img_path,5500,5500)
        if datasetname is None:
            print('Could not open %s' % img_path)
        img_width = datasetname.RasterXSize
        img_height = datasetname.RasterYSize
        imageSize = [img_width, img_height]
        nBand = datasetname.RasterCount

        num_Xblock = img_width // crop_width
        if img_width % crop_width > 0:
            num_Xblock += 1
        num_Yblock = img_height // crop_height
        if img_height % crop_height > 0:
            num_Yblock += 1
        i = 0
        blocks = num_Xblock * num_Yblock
        mask = np.zeros([batch_size, img_height, img_width], dtype=np.float32)
        add = np.zeros([1, img_height, img_width], dtype=np.uint8)
        input_gen = self.block_gdal_input(datasetname, imageSize, x_width, pad)
        for i in tqdm(range(blocks)):
            img, xy = next(input_gen)
            if xy[0] > 0:
                xs = xy[0] + pad
            else:
                xs = xy[0]

            if xy[2] > 0:
                ys = xy[2] + pad
            else:
                ys = xy[2]
            if np.max(img) < 0.1:
                predictions = np.zeros([batch_size, x_height, x_width])
            else:
                with torch.no_grad():
                    imgs = []
                    # for feat_trans in [img, np.rollaxis(img, 1, 0)]:
                    for [x_step, y_step] in [[1, 1], [-1, 1], [1, -1], [-1, -1]]:
                        # print(img[::x_step, ::y_step, :].shape)
                        feature_w_padding = img[::x_step, ::y_step, :] / 255.0 * 3.2 - 1.6
                        # np.squeeze(feature_w_padding)
                        # np.expand_dims(feature_w_padding,axis=2)
                        # feature_w_padding = feature_w_padding.transpose(2,0,1)
                        imgs.append(feature_w_padding)
                    imgs = np.array(imgs)
                    # imgs = imgs[:,np.newaxis]
                    # np.squeeze(imgs)
                    # np.expand_dims(imgs,axis=1)
                    imgs = imgs.transpose(0, 3, 1, 2)
                    # print(imgs.shape)
                    imgs = V(torch.Tensor(np.array(imgs, np.float32)).cuda())

                    # predictions = self.net.forward(imgs).squeeze().cpu().data.numpy()

                    res = self.net.forward(imgs)
                    predictions = res[-1].squeeze().cpu().data.numpy()
                    # out=self.net.forward(imgs)
                    # predictions = F.sigmoid(out[-1]).cpu().data.numpy()
            idx = 0
            # for feat_trans in [0,1]:
            for [x_step, y_step] in [[1, 1], [-1, 1], [1, -1], [-1, -1]]:
                prediction = predictions[idx, pad: pad + crop_height,
                             pad: pad + crop_width]

                prediction = prediction[::x_step, ::y_step]

                # if idx > 3:
                #    prediction  = np.rollaxis(prediction , 1, 0)

                mask[idx, ys: ys + crop_height, xs: xs + crop_width] = prediction.astype(np.float32)
                idx += 1

        datasetname = None
        return np.squeeze(np.mean(mask, axis=0) * 255).astype(np.int)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


def array2raster(newRasterfn, originRasterfn, array):
    cols = array.shape[1]
    rows = array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)

    originDS = gdal.Open(originRasterfn, 0)
    geoTrans = originDS.GetGeoTransform()
    outRaster.SetGeoTransform(geoTrans)
    outRaster.SetProjection(originDS.GetProjection())
    outband.FlushCache()


def test():
    # multiprocessing.freeze_support()
    # inImg = gFlags.inImg
    # weights = gFlags.save_weights
    # model = gFlags.model
    # solver = TTAFrame(eval(model))
    # solver.load(weights)
    # inImg = r'Z:\Project_Data\chongqing\work_task\yubei_2020'
    # weights = r"weights/chongqing_poly/chongqing_poly_800epoachs_my_DinkNet34_3plus_first7_20211212_crop960_bt4_lab3_700.th"
    # model = "my_DinkNet34_3plus_first7"
    test_path = r"E:\lry\zhejiang\train_edge_1175\test.txt"
    save_path = r"Z:\lry\data\edter_test\pred"
    with open(test_path, 'r') as file:
        lines = file.readlines()

    pred_list=[]
    lab_list=[]
    no_img = len(lines)
    metric = MeanIoUMetric(70)
    for i in range(no_img):
        test_img = lines[i].strip().split(" ")[0]
        lab_img = lines[i].strip().split(" ")[1]
        save_img = os.path.join(save_path, os.path.basename(test_img))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if os.path.exists(save_img):
            resam = test_img  # "d:\\temp.tif"
            save = save_img  # "d:\\temp.tif"
            # mask = solver.pred_gdal_blocks(resam)
            print(resam)
            pred_data = cv2.imread(save, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            pred_list.append(pred_data)

            seg_data = cv2.imread(lab_img, cv2.IMREAD_GRAYSCALE)
            _, seg_data_binary=cv2.threshold(seg_data, 0, 1, cv2.THRESH_BINARY)
            # 分割标签统一膨胀为5像素
            # 使用 skimage 的 skeletonize 函数
            skeleton = skeletonize(seg_data_binary)
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)
            _, dilated = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)
            lab_list.append(dilated)
            # solver.pred_gdal_blocks_write_multiclass(resam, save, repeats=1)
            # array2raster(all_saveImg[i], resam, mask)
    metric.update(lab_list, pred_list)

    metric_score = metric.compute()
    print("precision: ", metric_score['precision'])
    print("recall: ", metric_score['recall'])
    print("accuracy: ", metric_score['accuracy'])
    print("miou: ", metric_score['miou'])
    print("ods_f1: ", metric_score['ods_f1'])
    print("ois_f1: ", metric_score['ois_f1'])
    print("ap: ", metric_score['ap'])

    return

def test2():
    save_path = r"/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/lky/diffresult/polytif"     #预测结果
    lab_path=r"/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/lky/diffresult/edge"

    pred_list=[]
    lab_list=[]
    metric = MeanIoUMetric(70)
    for lab_name in tqdm(os.listdir(lab_path)):
        # labs = lab_name.split("_")
        # save = labs[0]+"_"+labs[1]+"_data.png"
        # save = labs[0] + "_" + labs[1] + "_data.tif"
        labs = lab_name.split(".")
        save = labs[0]+".tif"
        save_img = os.path.join(save_path, save)

        if os.path.exists(save_img):
            lab = os.path.join(lab_path, lab_name)
            save = save_img  # "d:\\temp.tif"

            pred_data = cv2.imread(save, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            pred_list.append(pred_data)

            seg_data = cv2.imread(lab, cv2.IMREAD_GRAYSCALE)
            _, seg_data_binary=cv2.threshold(seg_data, 0, 1, cv2.THRESH_BINARY)
            # 分割标签统一膨胀为5像素
            # 使用 skimage 的 skeletonize 函数
            skeleton = skeletonize(seg_data_binary)
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)
            _, dilated = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)
            lab_list.append(dilated)
            # solver.pred_gdal_blocks_write_multiclass(resam, save, repeats=1)
            # array2raster(all_saveImg[i], resam, mask)
    print("开始计算分数！")
    metric.update(lab_list, pred_list)

    metric_score = metric.compute()
    print("precision: ", metric_score['precision'])
    print("recall: ", metric_score['recall'])
    print("accuracy: ", metric_score['accuracy'])
    print("miou: ", metric_score['miou'])
    print("ods_f1: ", metric_score['ods_f1'])
    print("ois_f1: ", metric_score['ois_f1'])
    print("ap: ", metric_score['ap'])

    return


if __name__ == '__main__':
    # test()
    test2()