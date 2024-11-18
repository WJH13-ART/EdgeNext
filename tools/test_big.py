import argparse

import ipdb
from mmcv.parallel import DataContainer
import mmcv
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
import os.path as osp
from mmseg.models import build_segmentor
import torch
from torch.autograd import Variable as V
import cv2
import os, sys
import numpy as np
from tensorflow import float32
from tqdm import tqdm
from osgeo import gdal, ogr, osr
import paddle
from numpy import array
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from torchvision import transforms
os.environ['RANK']='0'
os.environ['WORLD_SIZE']='1'
os.environ['MASTER_ADDR']='127.0.3.10'
os.environ['MASTER_PORT']='21000'




def block_gdal_input(x_width,img, img_size, crop, pad=0):
    [img_width, img_height] = img_size
    x_height = x_width
    x_width = x_width
    crop_width =x_width - 2 * pad
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
def pred_gdal_blocks_write_multi(img_path, out_path='', repeats=1):
    sys.setrecursionlimit(10000)
    # pad = 128
    # x_width = 1000
    pad = 32
    x_width = 320
    datasetname = gdal.Open(img_path, gdal.GA_ReadOnly)  # 打开图像
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
    if img_height * img_width > 9e8:  # 判断能否放入内存
        # outRaster = gdal.Open(out_path,1)
        # 写GTiff文件。GetDriverByName读取GTiff的数据，需要先载入数据驱动，初始化一个对象。Create创建空文件，并确定开辟多大内存；每个像素都有一个对应的值，这个值得类型用数据类型指定。这里的数据类型是gdal数据类型。
        driver = gdal.GetDriverByName('GTiff')
        # tempRaster = driver.Create(temp_path, img_width, img_height, 4, gdal.GDT_Float32)
        tempRaster = driver.Create(temp_path, img_width, img_height, nBand, gdal.GDT_Byte)
        bigImg = True

    else:
        mask = np.zeros([repeats, img_height, img_width], float)

    crops = [x_width + 128, x_width, x_width - 64, x_width - 128]
    steps = [[1, 1], [-1, -1], [-1, 1], [1, -1]]

    img_metas = [[{
        'filename': '/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/gengdi_Data/500013_clip6.png',
        'ori_filename': '/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/gengdi_Data/500013_clip6.png',
        'ori_shape': (1000, 1000, 3), 'img_shape': (1000, 1000, 3),
        'pad_shape': (1000, 1000, 3), 'scale_factor': 1.0,
        'img_norm_cfg': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                         'to_rgb': True}, 'img_id': '500013_clip6'}]]
    mean = img_metas[-1][-1]['img_norm_cfg']['mean']
    std = img_metas[-1][-1]['img_norm_cfg']['std']
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    # 加载模型
    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:  # 启动TTA
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [  # 数据增强
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # global_cfg = mmcv.Config.fromfile(args.globalconfig)
    # global_cfg.work_dir = osp.join('../work_dirs', osp.splitext(osp.basename(args.globalconfig))[0])
    # global_cfg.global_model_path = args.global_checkpoint
    # model = build_segmentor_local8x8(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg,
    #                                  global_cfg=global_cfg)
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    print("Load Local Checkpoint from   =======>>>>   " + args.checkpoint)
    checkpoint_dict = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    model_dict = model.state_dict()
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    model.eval()
    model = MMDataParallel(model, device_ids=[0])
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    for k in range(repeats):
        # crop_data = crops[k]
        # pad = crop_data // 16
        crop_data = x_width
        crop_width = crop_data - 2 * pad
        crop_height = crop_data - 2 * pad
        num_Xblock = img_width // crop_width

        if img_width % crop_width > 0:
            num_Xblock += 1
        num_Yblock = img_height // crop_height
        if img_height % crop_height > 0:
            num_Yblock += 1
        i = 0
        blocks = num_Xblock * num_Yblock  # 计算大图分块数量
        # mask = np.zeros([batch_size, img_height, img_width],dtype=np.float32)
        input_gen = block_gdal_input(x_width,datasetname, imageSize, crop_data, pad)  # 将分块读进来
        print(blocks)
        for i in tqdm(range(blocks)):  # tqdm显示进度条
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
                prediction = np.zeros([1, x_height, x_width])
            else:
                with torch.no_grad():  # 关闭自动求导，节省内存
                    [x_step, y_step] = steps[k]
                    img_tta = img[::x_step, ::y_step, :]
                    imgs = []
                    imgs.append(img_tta)
                    imgs = np.array(imgs)
                    # imgs = imgs.transpose(0, 3, 1, 2)  # / 255.0 * 3.2 - 1.6        <class 'numpy.ndarray'>
                    imgs = imgs[-1]
                    imgs = imgs.astype('uint8')
                    imgs = transforms.ToTensor()(imgs)
                    imgs = transforms.Normalize(mean,std)(imgs)
                    imgs = imgs.unsqueeze(0)
                    # imgs = V(torch.Tensor(np.array(imgs1, np.float32)).cuda())   #<class 'torch.Tensor'>         1*3*1000*1000
                    data = {'img_metas':img_metas,'img':[imgs]}
                    print(data)
                    # ipdb.set_trace()
                    res = model(return_loss=False, rescale=True, **data)
                    res = torch.tensor(res.tolist())
                    #   进入预测
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
    outRaster.SetGeoTransform(datasetname.GetGeoTransform())  # 写入仿射变换参数
    outRaster.SetProjection(datasetname.GetProjection())  # 写入投影信息
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
    # return  # np.squeeze(mask*255).astype(np.int)
    return np.squeeze(np.mean(mask, axis=0)).astype(int)

def parse_args():
    parser = argparse.ArgumentParser(description='mmseg test (and eval) a model')
    # parser.add_argument('--globalconfig', type=str,
    #                     default='../configs/gengdi/EDTER_BIMLA_1000x1000_400_gengdiZJ.py',
    #                     # 第一阶段的config
    #                     help='train global config file path')
    # parser.add_argument('--global-checkpoint', type=str,
    #                     default='../work_dirs/EDTER_BIMLA_1000x1000_40k_gengdiZJ/latest.pth',
    #                     help='the dir of global model')

    parser.add_argument('--config', type=str,
                        default='/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/SegNeXt-main/configs_my/segnext/base/segnext.base.512x512.gengdi.80k.py',
                        # 第二阶段的config
                        help='train local config file path')
    parser.add_argument('--checkpoint', type=str,
                        default='/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/SegNeXt-main/tools/work_dirs/segnext.base.512x512.gengdi.80k/latest.pth',
                        help='the dir of local model')


    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        #default=True,
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved',
        type =str, default='')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir', type =str, default='/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/lky/data/chongzuo/res',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    args = parse_args()
    TifPath = r"/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/lky/data/chongzuo/merge_4_to_3.tif"
    resPath = r"/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/lky/data/chongzuo/segnet_result2.tif"
    # # build the model and load checkpoint
    print(TifPath)
    print(resPath)

    pred_gdal_blocks_write_multi(img_path=TifPath,out_path=resPath)
