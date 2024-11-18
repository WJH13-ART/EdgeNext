'''

'''
import os
import gdal
import numpy as np
##############xianyong clip
ori_path = r'/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/gengdi_Data/train/jiashan/line_lab'
out_path = r'/media/xia/40903229-3d5d-4272-81a0-93b4ad6abb57/WJH/gengdi_Data/train/jiashan/320/lab'
tif_name_list = os.listdir(ori_path)
for tif_name in tif_name_list:
    if tif_name[-4:] == '.tif':
        tif_path = os.path.join(ori_path, tif_name)
        tif_out = os.path.join(out_path, tif_name)

        # 面栅格
        dataset = gdal.Open(tif_path)
        width = dataset.RasterXSize  # 获取数据宽度
        height = dataset.RasterYSize  # 获取数据高度
        outbandsize = dataset.RasterCount  # 获取数据波段数
        im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
        im_proj = dataset.GetProjection()  # 获取投影信息
        datatype = dataset.GetRasterBand(1).DataType
        bands = dataset.RasterCount  # 波段数
        im_data = dataset.ReadAsArray()  # 获取数据  numpy.ndarray

        w = 0  # 开始裁剪
        step = 320
        while w < width -step:
            h = 0
            while h < height-step:
                output_filename = os.path.join(out_path, tif_name[:-4]+'_'+str(w//step)+'_'+str(h//step)+'.tif')
                driver = gdal.GetDriverByName("GTiff")
                dataset = driver.Create(output_filename, step, step, bands, gdal.GDT_Byte)
                if bands == 3:
                    im_blueBand = im_data[2, h:h+step, w:w+step]  # 蓝
                    im_greenBand = im_data[1, h:h+step, w:w+step]  # 绿
                    im_redBand = im_data[0, h:h+step, w:w+step]  # 红
                    dataset.GetRasterBand(1).WriteArray(im_redBand)
                    dataset.GetRasterBand(2).WriteArray(im_greenBand)
                    dataset.GetRasterBand(3).WriteArray(im_blueBand)
                if bands == 1:
                    temp_im_data = im_data[h:h+step, w:w+step]
                    dataset.GetRasterBand(1).WriteArray(temp_im_data)

                h += step
                # break
            w += step
            # break
        # break