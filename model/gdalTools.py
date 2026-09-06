import os, sys, time
import numpy as np
from osgeo import ogr, gdal, gdalconst


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def stretch_n(bands, img_min, img_max, lower_percent=0, higher_percent=100):
    """
    :param bands:  目标数据，numpy格式
    :param img_min:   目标位深的最小值，以8bit为例，最大值为255， 最小值为0
    :param img_max:    目标位深的最大值
    :return:
    """
    out = np.zeros_like(bands).astype(np.float32)
    a = img_min
    b = img_max
    c = np.percentile(bands[:, :], lower_percent)
    d = np.percentile(bands[:, :], higher_percent)
    t = a + (bands[:, :] - c) * (b - a) / (d - c)
    t[t < a] = a
    t[t > b] = b
    out[:, :] = t
    return out


def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset


def image_resampling(source_file, target_file, scale=5.):
    """
          image resampling
    :param source_file: the path of source file
    :param target_file: the path of target file
    :param scale: pixel scaling
    :return: None
    """
    dataset = gdal.Open(source_file, gdalconst.GA_ReadOnly)
    band_count = dataset.RasterCount  # 波段数

    if band_count == 0 or not scale > 0:
        print("参数异常")
        return

    cols = dataset.RasterXSize  # 列数
    rows = dataset.RasterYSize  # 行数
    cols = int(cols * scale)  # 计算新的行列数
    rows = int(rows * scale)

    geotrans = list(dataset.GetGeoTransform())
    print(dataset.GetGeoTransform())
    print(geotrans)
    geotrans[1] = geotrans[1] / scale  # 像元宽度变为原来的scale倍
    geotrans[5] = geotrans[5] / scale  # 像元高度变为原来的scale倍
    print(geotrans)

    if os.path.exists(target_file) and os.path.isfile(target_file):  # 如果已存在同名影像
        os.remove(target_file)  # 则删除之

    band1 = dataset.GetRasterBand(1)
    data_type = band1.DataType
    target = dataset.GetDriver().Create(target_file, xsize=cols, ysize=rows, bands=band_count,
                                        eType=data_type)
    target.SetProjection(dataset.GetProjection())  # 设置投影坐标
    target.SetGeoTransform(geotrans)  # 设置地理变换参数
    total = band_count + 1
    for index in range(1, total):
        # 读取波段数据
        print("正在写入" + str(index) + "波段")
        data = dataset.GetRasterBand(index).ReadAsArray(buf_xsize=cols, buf_ysize=rows)
        out_band = target.GetRasterBand(index)
        # out_band.SetNoDataValue(dataset.GetRasterBand(index).GetNoDataValue())
        out_band.WriteArray(data)  # 写入数据到新影像中
        out_band.FlushCache()
        out_band.ComputeBandStats(False)  # 计算统计信息
    print("正在写入完成")
    del dataset


def sample_clip(shp, tif, outputdir, sampletype, size, fieldName='cls', n=None):
    """
        according to sampling point, generating image slices
    :param shp: the path of shape file
    :param tif: the path of image
    :param outputdir: the directory of output
    :param sampletype: line or polygon
    :param size:  the size of images slices
    :param fieldName: the name of field
    :param n: the start number
    :return:
    """
    time1 = time.clock()

    gdal.AllRegister()
    lc = gdal.Open(tif)
    im_width = lc.RasterXSize
    im_height = lc.RasterYSize
    im_geotrans = lc.GetGeoTransform()
    bandscount = lc.RasterCount
    im_proj = lc.GetProjection()
    print(im_width, im_height)
    gdal.AllRegister()
    gdal.SetConfigOption("gdal_FILENAME_IS_UTF8", "YES")

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dsshp = driver.Open(shp, 0)
    if dsshp is None:
        print('Could not open ' + 'sites.shp')
        sys.exit(1)
    layer = dsshp.GetLayer()
    xValues = []
    yValues = []
    m = layer.GetFeatureCount()
    feature = layer.GetNextFeature()
    print("tif_bands:{0},samples_nums:{1},sample_type:{2},sample_size:{3}*{3}".format(bandscount, m, sampletype,
                                                                                      int(size)))

    if n is not None:
        pass
    else:
        n = 1
    while feature:
        if n < 10:
            dirname = "0000000" + str(n)
        elif n >= 10 and n < 100:
            dirname = "000000" + str(n)
        elif n >= 100 and n > 1000:
            dirname = "00000" + str(n)
        else:
            dirname = "0000" + str(n)

        # print dirname
        dirpath = os.path.join(outputdir, dirname + "_V1")
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        tifname = dirname + ".tif"
        if "poly" in sampletype or "POLY" in sampletype:
            shpname = dirname + "_V1_POLY.shp"
        if "line" in sampletype or "LINE" in sampletype:
            shpname = dirname + "_V1_LINE.shp"
        geometry = feature.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()
        print(x, y)
        print(im_geotrans)
        xValues.append(x)
        yValues.append(y)
        newform = []
        newform = list(im_geotrans)
        # print newform
        newform[0] = x - im_geotrans[1] * int(size) / 2.0
        newform[3] = y - im_geotrans[5] * int(size) / 2.0
        print(newform[0], newform[3])
        newformtuple = tuple(newform)
        x1 = x - int(size) / 2 * im_geotrans[1]
        y1 = y - int(size) / 2 * im_geotrans[5]
        x2 = x + int(size) / 2 * im_geotrans[1]
        y2 = y - int(size) / 2 * im_geotrans[5]
        x3 = x - int(size) / 2 * im_geotrans[1]
        y3 = y + int(size) / 2 * im_geotrans[5]
        x4 = x + int(size) / 2 * im_geotrans[1]
        y4 = y + int(size) / 2 * im_geotrans[5]
        Xpix = (x1 - im_geotrans[0]) / im_geotrans[1]
        # Xpix=(newform[0]-im_geotrans[0])

        Ypix = (newform[3] - im_geotrans[3]) / im_geotrans[5]
        # Ypix=abs(newform[3]-im_geotrans[3])
        print("#################")
        print(Xpix, Ypix)

        # **************create tif**********************
        # print"start creating {0}".format(tifname)
        pBuf = None
        pBuf = lc.ReadAsArray(int(Xpix), int(Ypix), int(size), int(size))
        # print pBuf.dtype.name
        driver = gdal.GetDriverByName("GTiff")
        create_option = []
        if 'int8' in pBuf.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in pBuf.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        outtif = os.path.join(dirpath, tifname)
        ds = driver.Create(outtif, int(size), int(size), int(bandscount), datatype, options=create_option)
        if ds == None:
            print("2222")
        ds.SetProjection(im_proj)
        ds.SetGeoTransform(newformtuple)
        ds.FlushCache()
        if bandscount > 1:
            for i in range(int(bandscount)):
                outBand = ds.GetRasterBand(i + 1)
                outBand.WriteArray(pBuf[i])
        else:
            outBand = ds.GetRasterBand(1)
            outBand.WriteArray(pBuf)
        ds.FlushCache()
        # print "creating {0} successfully".format(tifname)
        # **************create shp**********************
        # print"start creating shps"
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
        gdal.SetConfigOption("SHAPE_ENCODING", "")
        strVectorFile = os.path.join(dirpath, shpname)
        ogr.RegisterAll()
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.Open(shp)
        layer0 = ds.GetLayerByIndex(0)
        prosrs = layer0.GetSpatialRef()
        # geosrs = osr.SpatialReference()

        oDriver = ogr.GetDriverByName("ESRI Shapefile")
        if oDriver == None:
            print("1")
            return

        oDS = oDriver.CreateDataSource(strVectorFile)
        if oDS == None:
            print("2")
            return

        papszLCO = []
        if "line" in sampletype or "LINE" in sampletype:
            oLayer = oDS.CreateLayer("TestPolygon", prosrs, ogr.wkbLineString, papszLCO)
        if "poly" in sampletype or "POLY" in sampletype:
            oLayer = oDS.CreateLayer("TestPolygon", prosrs, ogr.wkbPolygon, papszLCO)
        if oLayer == None:
            print("3")
            return

        oFieldName = ogr.FieldDefn(fieldName, ogr.OFTString)
        oFieldName.SetWidth(50)
        oLayer.CreateField(oFieldName, 1)
        oDefn = oLayer.GetLayerDefn()
        oFeatureRectangle = ogr.Feature(oDefn)

        geomRectangle = ogr.CreateGeometryFromWkt(
            "POLYGON (({0} {1},{2} {3},{4} {5},{6} {7},{0} {1}))".format(x1, y1, x2, y2, x4, y4, x3, y3))
        oFeatureRectangle.SetGeometry(geomRectangle)
        oLayer.CreateFeature(oFeatureRectangle)
        print("{0} ok".format(dirname))
        n = n + 1
        feature = layer.GetNextFeature()
    time2 = time.clock()
    print('Process Running time: %s min' % ((time2 - time1) / 60))

    return n

# shp边界转换为二值边界
def load_shp_to_mask(shp_path, ref_tif_path, output_mask_path=None):
    """
    将SHP文件转换为与参考影像同范围、同分辨率的掩码
    :param shp_path: SHP文件路径
    :param ref_tif_path: 参考影像路径（用于获取地理范围和分辨率）
    :param output_mask_path: 输出掩码TIFF路径（可选）
    :return: 掩码数组、投影信息、地理变换参数
    """
    # 读取参考影像信息
    ref_proj, ref_geotrans, ref_width, ref_height, _ = read_img(ref_tif_path)
    x_min, x_res, _, y_max, _, y_res = ref_geotrans
    x_max = x_min + ref_width * x_res
    y_min = y_max + ref_height * y_res

    # 创建掩码数组
    mask = np.zeros((ref_height, ref_width), dtype=np.uint8)

    # 打开SHP文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(shp_path, 0)
    if ds is None:
        raise FileNotFoundError(f"无法打开SHP文件: {shp_path}")
    layer = ds.GetLayer()

    # 循环要素并绘制到掩码
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        # 将几何坐标转换为像素坐标
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for i in range(geom.GetGeometryCount()):
            sub_geom = geom.GetGeometryRef(i)
            for j in range(sub_geom.GetPointCount()):
                x, y, _ = sub_geom.GetPoint(j)
                # 计算像素坐标
                px = int((x - x_min) / x_res)
                py = int((y - y_max) / y_res)
                ring.AddPoint(px, py)
        ring.CloseRings()
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        # 填充多边形到掩码
        from osgeo import gdal_array
        gdal_array.Polygonize(poly, None, mask, 0, [], callback=None)

    ds.Destroy()

    # 保存掩码（如果指定输出路径）
    if output_mask_path:
        write_img(output_mask_path, ref_proj, ref_geotrans, mask)

    return mask, ref_proj, ref_geotrans


def shp_to_mask(shp_path, ref_tif_path, output_mask_path):
    """
    将SHP矢量标签转换为与参考TIFF对齐的二值掩码
    :param shp_path: SHP矢量标签路径
    :param ref_tif_path: 参考TIFF路径（用于获取尺寸、投影、地理变换）
    :param output_mask_path: 输出掩码TIFF路径
    :return: 二值掩码数组（0背景，1前景）
    """
    # 读取参考TIFF的信息（确保掩码与预测结果空间对齐）
    ref_proj, ref_geotrans, ref_width, ref_height, _ = read_img(ref_tif_path)

    # 创建输出掩码（全0背景）
    driver = gdal.GetDriverByName("GTiff")
    mask_ds = driver.Create(
        output_mask_path,
        ref_width,
        ref_height,
        1,
        gdal.GDT_Byte  # 8位无符号整数（0/1）
    )
    mask_ds.SetProjection(ref_proj)
    mask_ds.SetGeoTransform(ref_geotrans)
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.Fill(0)  # 初始化为背景

    # 栅格化SHP到掩码（将矢量区域设为1）
    ogr_ds = ogr.Open(shp_path)
    ogr_layer = ogr_ds.GetLayer()
    gdal.RasterizeLayer(
        mask_ds,
        [1],  # 目标波段
        ogr_layer,
        burn_values=[1]  # 将矢量区域烧录为1
    )

    # 读取掩码数组
    mask = mask_band.ReadAsArray()
    del mask_ds, ogr_ds
    return mask

# 计算植被指数
def compute_vegetation_indices(im_data):
    """
    计算植被指数（适用于RGB+近红外三波段数据）
    :param im_data: 原始波段数据 [C, H, W]，假设顺序为[R, G, B, NIR]（若波段顺序不同需调整索引）
    :return: 扩展后的特征 [C+2, H, W]（新增NDVI和EVI）
    """
    # 波段索引：假设0=R，1=G，2=B，3=NIR（根据实际数据调整）
    R = im_data[0].astype(np.float32)
    G = im_data[1].astype(np.float32)
    B = im_data[2].astype(np.float32)
    NIR = im_data[3].astype(np.float32) if im_data.shape[0] == 4 else im_data[3].astype(np.float32)  # 适配三波段（若近红外为第三波段）

    # 1. NDVI：(NIR-R)/(NIR+R) → 耕地值更集中，林地因阴影波动大
    ndvi = (NIR - R) / (NIR + R + 1e-6)
    ndvi = (ndvi + 1) / 2  # 归一化到[0,1]

    # 2. EVI：增强植被指数，减少土壤和大气影响
    evi = 2.5 * (NIR - R) / (NIR + 6 * R - 7.5 * B + 1 + 1e-6)
    evi = (evi + 2.5) / 5  # 归一化到[0,1]

    # 合并原始波段与指数（通道维度扩展）
    return np.concatenate([im_data, ndvi[np.newaxis, ...]], axis=0)
