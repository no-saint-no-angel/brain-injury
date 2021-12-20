"""
    这个文件的功能是，在切割出来的小图片中，计算颅脑区域的中心，将颅脑区域的中心调整为图片的中心。
    具体步骤：
    1、判断图片高度和宽度哪个大，将大的调整为更小一个的值。使得调整之后，在原来大的尺寸方向，颅脑中心位于图像坐标中心。
    2、在原来尺寸小的方向，根据颅脑区域中心坐标，将中心调整为该方向图像坐标中心。
    3、将所有图片尺寸调整为320*320
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy import ndimage
from data_preprocess import pic_preprocess


def max_area_extract(input_chanel):
    # 这个函数的功能是提取通道中面积最大的区域
    chanel2 = np.array(input_chanel, dtype='uint8')  # 把数据类型转换成cv2.findContours需要的类型
    # print(chanel2.dtype)
    # 1、连通域分析。
    # 只寻找 【外部区域】连通域 子连通域忽略。
    contours, hierarchy = cv2.findContours(chanel2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 【外部区域】和 【内部区域】连通域 子连通域不忽略。
    # img_contour, contours, hierarchy = cv2.findContours(chanel2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 2、轮廓面积打印
    # print(len(contours))
    if len(contours) > 1:
        area_list = []
        img_contours = []
        img_area_chenal = []
        for j in range(len(contours)):
            # 1、计算每一个连通区域的面积，保存在area_list中
            area = cv2.contourArea(contours[j])
            # print("轮廓 %d 的面积是:%d" % (j, area))
            area_list.append(area)
            # 2、定义一个白板，用来存储每个连通区域的轮廓面积
            img_temp = np.zeros(chanel2.shape, np.uint8)
            img_contours.append(img_temp)
            # 3、把每个轮廓面积画到对应的白板上
            cv2.drawContours(img_contours[j], contours, j, (255, 255, 255), -1)
            # 4、存储每一个轮廓面积到img_area_chenal中
            img_area_chenal.append(img_contours[j])
        # 求出area_list中的最大值，即面积最大连通区域对应的索引序号
        max_where = np.where(area_list == np.max(area_list, axis=0))
        max_area_chenal = max_where[0]
        img_area_chenal_max = img_area_chenal[max_area_chenal[0]]
        img_area_chenal_max = np.array(img_area_chenal_max, dtype='float32')
        # 计算最大面积区域的质心
        moments_max_area = cv2.moments(img_contours[max_area_chenal[0]])
        centerX = int(moments_max_area["m10"] / moments_max_area["m00"])
        centerY = int(moments_max_area["m01"] / moments_max_area["m00"])
        center_point = [centerX, centerY]
        # print(img_area_chenal_max.dtype)
        # cv2.imshow('img_area_chenal_max', img_area_chenal_max)
        # cv2.waitKey(0)
        out_chanel = img_area_chenal_max/255  # 除以255是因为原本输入通道的像素值是1，经过面积区域法之后像素值变成255，为了后续的处理

        # # 第二种方法
        # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(chanel2, 8, ltype=cv2.CV_32S)
    else:
        out_chanel = input_chanel
        center_point = [int(out_chanel.shape[0]/2), int(out_chanel.shape[1]/2)]
    return center_point, out_chanel


def calcu_middlepoint(input_img):
    """
        这个函数的功能是求出输入颅脑图片的中心坐标，并返回。并生成颅脑热力图
    """
    # 灰度化
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # print('img_gray.shape', img_gray.shape)
    # cv2.imshow('before_pro', img_gray)
    # 对灰度图用卷积核操作一下，去除白色边界线中间出现黑色线（一列像素值）的情况
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_gray = cv2.erode(img_gray, kernel)
    # cv2.imshow('after_erode', img_gray)
    # img_gray = cv2.dilate(img_gray, kernel)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # cv2.imshow('after_GaussianBlur', img_gray)
    # 2 二值化处理
    img_gray[img_gray < 30] = 0
    img_gray[img_gray > 0] = 1
    # 计算图片连通域面积最大的区域，保留结果，得到颅脑区域
    center_point, brain_max_area = max_area_extract(img_gray)
    # 3.二值化处理

    # ret, img_bin = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)

    # 距离变换
    # 1.scipy的方法,不要将brain_max_area数组的float32类型转换成uint8类型
    img_edt = ndimage.distance_transform_edt(brain_max_area)

    # 2.cv的方法， 要将brain_max_area数组的float32类型转换成uint8类型
    # （1）使用np数组的转换方式，调用astype函数
    # brain_max_area = brain_max_area.astype(np.uint8)
    # （2）使用OpenCV通过线性变换将数据转换成8位[uint8]
    # img_edt = cv2.convertScaleAbs(brain_max_area)
    # img_edt = cv2.distanceTransform(src=brain_max_area, distanceType=cv2.DIST_L2, maskSize=5)

    # 计算颅脑中间区域距离背景最远像素值的位置
    # 1.cv的方法
    min_value,max_value,minloc,maxloc = cv2.minMaxLoc(img_edt)  # 找最大的点
    # 2.python的方法
    index = np.where(img_edt == np.max(img_edt))
    # print('index', index, 'maxloc', maxloc)

    # 将距离变换之后的图片转换成热力图
    dist2 = cv2.normalize(img_edt, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    heat_img = cv2.applyColorMap(dist2, cv2.COLORMAP_JET)

    # # 作图显示
    # fig, axs = plt.subplots(3, figsize=(6, 4))
    # axs[0].imshow(img_gray)
    # axs[1].imshow(brain_max_area)
    # axs[2].imshow(img_edt)
    # plt.show()

    # cv2.imshow('img_gray', img_gray)
    # # cv2.imshow('img_bin', img_bin)
    # # cv2.imshow('img_opening', img_opening)
    # cv2.imshow('heat_img', heat_img)
    # cv2.waitKey(0)
    return center_point, maxloc


def pic_size_big_small_2_ss_first(input_img, big_size, small_size, middle_point, fangxiang):
    """
        这个函数的功能是，将大尺寸方向的边调整为小尺寸，并保证大尺寸方向颅脑的中心位于图像的坐标中心
        input_img：输入图像
        big_size：大尺寸边的边长
        small_size：小尺寸边的边长
        middle_point：中心点坐标
        fangxiang：传入大尺寸边属于高度方向还是宽度方向
    """
    if fangxiang == 'height':
        middle_point = middle_point[1]
    else:
        middle_point = middle_point[0]

    # 裁剪图片，将大尺寸方向的颅脑区域中心往图像坐标中心调整, 图片尺寸变为，big_size_temp*small_size
    if middle_point > int(big_size/2):  # 颅脑区域的中心位于长尺寸方向[big_size/2, big_size]内部
        # 裁剪后大尺寸方向的尺寸变为big_size_temp
        big_size_temp = 2*(big_size - middle_point)
        # 裁剪量为delta
        delta = 2*middle_point - big_size
        # 图片尺寸变为，big_size_temp*small_size
        if fangxiang == 'height':
            input_img_temp = input_img[delta-1:, :, :]
        else:
            input_img_temp = input_img[:, delta - 1:, :]
    elif middle_point < int(big_size/2):  # 颅脑区域的中心位于长尺寸方向[0, big_size/2]内部
        # 裁剪后大尺寸方向的尺寸变为big_size_temp
        big_size_temp = 2*middle_point
        # 裁剪量为delta
        delta = big_size - 2*middle_point
        # 图片尺寸变为，big_size_temp*small_size
        if fangxiang == 'height':
            input_img_temp = input_img[:big_size_temp, :, :]
        else:
            input_img_temp = input_img[:, :big_size_temp, :]
    else:
        big_size_temp = big_size
        input_img_temp = input_img

    # 判断big_size_temp和small_size大小，来确定是big_size_temp方向是裁剪还是填充
    if big_size_temp > small_size:  # 裁剪
        delta_final = int((big_size_temp-small_size)/2)
        if fangxiang == 'height':
            output_img = input_img_temp[delta_final:delta_final+small_size, :, :]
        else:
            output_img = input_img_temp[:, delta_final:delta_final+small_size, :]
    elif big_size_temp < small_size:
        delta_final = int((small_size - big_size_temp) / 2)
        # 新建一个黑板，大小是small_size*small_size*3
        output_img = np.zeros((small_size, small_size, 3))
        if fangxiang == 'height':
            #  由于对小数点取整的原因，会导致big_size_temp和input_img_temp.shape[0]相差一个像素值，所以这里先判断一下大小
            if big_size_temp > input_img_temp.shape[0]:
                output_img[delta_final:delta_final + big_size_temp-1, :, :] = input_img_temp
            elif big_size_temp < input_img_temp.shape[0]:
                output_img[delta_final:delta_final + big_size_temp+1, :, :] = input_img_temp
            else:
                output_img[delta_final:delta_final + big_size_temp, :, :] = input_img_temp
        else:
            # s = output_img[:, delta_final:delta_final + big_size_temp, :]
            # output_img[:, delta_final:delta_final + big_size_temp, :] = input_img_temp
            #  由于对小数点取整的原因，会导致big_size_temp和input_img_temp.shape[1]相差一个像素值，所以这里先判断一下大小
            if big_size_temp > input_img_temp.shape[1]:
                output_img[:, delta_final:delta_final + big_size_temp-1, :] = input_img_temp
            elif big_size_temp < input_img_temp.shape[1]:
                output_img[:, delta_final:delta_final + big_size_temp+1, :] = input_img_temp
            else:
                output_img[:, delta_final:delta_final + big_size_temp, :] = input_img_temp
    else:
        output_img = input_img_temp

    return output_img


def pic_size_big_small_2_ss_second(input_img, middle_point, fangxiang):
    """
        这个函数的功能是，将小尺寸方向的颅脑的中心调整为图像的坐标中心。其中输入图像已经是正方形，大尺寸方向已经把颅脑区域中心和坐标中心对齐
        ，现在要做的就是把原来小尺寸方向对齐
        input_img：输入图像，输入的尺寸为small_size*small_size*3
        middle_point：中心点坐标
        fangxiang：传入大尺寸边属于高度方向还是宽度方向
    """
    # 先确定图片是否为正方形，不是的话，将图片调整为正方形，边长为小边
    h, w, c = input_img.shape
    if h > w:
        input_img = input_img[:h-1, :, :]
    elif h < w:
        input_img = input_img[:, :w-1, :]

    if fangxiang == 'height':
        middle_point = middle_point[1]
    else:
        middle_point = middle_point[0]
    big_size = input_img.shape[0]
    # 新建一个黑板，大小是big_size*big_size*3
    output_img = np.zeros((big_size, big_size, 3))
    # 平移量为delta
    delta = np.abs(middle_point - int(big_size/2))

    if middle_point > int(big_size/2):  # 颅脑区域的中心位于长尺寸方向[big_size/2, big_size]内部, 把input_img的后半部分复制到output_img前半部分

        if fangxiang == 'height':
            output_img[:big_size-delta, :, :] = input_img[delta:, :, :]
        else:
            output_img[:, :big_size-delta, :] = input_img[:, delta:, :]

    elif middle_point < int(big_size/2):  # 颅脑区域的中心位于长尺寸方向[0, big_size/2]内部, 把input_img的前半部分复制到output_img后半部分

        if fangxiang == 'height':
            output_img[delta:, :, :] = input_img[:big_size-delta, :, :]
        else:
            output_img[:, delta:, :] = input_img[:, :big_size - delta, :]
    else:
        output_img = input_img

    # 两条虚线内的代码是进行图片resize操作
    # --------------------------------------------------------------------------------------------
    if output_img.shape[0] != 320:
        # 将大于320的图片resize到320*320，
        if output_img.shape[0] > 320:
            output_img = cv2.resize(output_img, dsize=(320, 320), interpolation=cv2.INTER_AREA)
        # 将小于320的图片resize到320*320，
        else:
            output_img = cv2.resize(output_img, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)
    else:
        output_img = output_img
    # --------------------------------------------------------------------------------------------
    return output_img


def data_resize(input_img):
    # 计算颅脑区域中心坐标,center_point, middle_point分别是用cv2.moments计算轮廓的质心和ndimage.distance_transform_edt计算颅脑区域距离背景最远的点
    # 两种计算方法得出的结果差异不大。坐标是[width, height]，和用.shape函数得出矩阵的坐标[height,width]刚好相反
    center_point, middle_point = calcu_middlepoint(input_img)
    h, w, c = input_img.shape
    # 调整图片尺寸
    """
        尺寸调整分为两个步骤，先将大尺寸方向的颅脑区域中心与坐标中心对齐，再将小尺寸方向的对齐。
    """
    if h > w:  # 高度大于宽度,先进行高度调整，得到w*w*3，然后调整宽度方向的中心，得到w*w*3，并resize到320*320
        # 1、调整高度以及高度方向中心
        img_output = pic_size_big_small_2_ss_first(input_img, big_size=h, small_size=w, middle_point=center_point,
                                         fangxiang='height')
        # 2、调整宽度方向中心,并resize到320*320
        img_output = pic_size_big_small_2_ss_second(img_output, middle_point=center_point, fangxiang='width')

    elif h < w:  # 高度小于宽度,先进行宽度调整，得到h*h*3，然后调整高度方向的中心，得到h*h*3，并resize到320*320
        # 1、调整宽度以及宽度方向中心
        img_output = pic_size_big_small_2_ss_first(input_img, big_size=w, small_size=h, middle_point=center_point,
                                         fangxiang='width')
        # 2、调整高度方向中心,并resize到320*320
        img_output = pic_size_big_small_2_ss_second(img_output, middle_point=center_point, fangxiang='height')
    else:
        img_output = input_img

    # 调用from data_preprocess import pic_preprocess函数对图像进行噪声去除
    # print(img_output.dtype)
    img_output = np.array(img_output, dtype='uint8')
    img_output = pic_preprocess(img_output)
    return img_output


# # 读取输入图片
# img0 = cv2.imread("D:\\projects\\brain_injury\\z_ct_big_origin\\chenbaojun (11).png")
# img0_resized = data_resize(img0)
# cv2.imshow('img0_resized', img0_resized)
# cv2.waitKey(0)
# print('img0_resized.shape', img0_resized.shape)

root = 'D:\\projects\\brain_injury\\z_ct_totalall'
root_output_dir = root + "\\ct_small_resized\\jiansu\\nie2"
root_input_dir = root + "\\ct_small\\jiansu\\nie2\\"
dir_List = os.listdir(root_input_dir)
for i in range(0, len(dir_List)):
    # 每个案例的输入图片文件夹
    img_input_dir = os.path.join(root_input_dir, dir_List[i])
    # 每个案例的输出图片文件夹
    img_output_dir = os.path.join(root_output_dir, dir_List[i])
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)

    imgList_img = os.listdir(img_input_dir)
    for count in range(0, len(imgList_img)):
        print("\nprocessing the " + str(count + 1) + "th" + " picture.......")
        im_name = imgList_img[count]
        im_name_before_point = im_name.split('.')[0]
        im_path = os.path.join(img_input_dir, im_name)
        img_input = cv2.imread(im_path)

        output = data_resize(img_input)
        cv2.imwrite(img_output_dir + '\\' + im_name, output)


