from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

"""
    从一张大图片切割出来的小图片，小图片带有字和其他的噪声，这个文件的功能是去除这些噪声，然后把图片保存下来
"""


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
        # print(img_area_chenal_max.dtype)
        # cv2.imshow('img_area_chenal_max', img_area_chenal_max)
        out_chanel = img_area_chenal_max/255  # 除以255是因为原本输入通道的像素值是1，经过面积区域法之后像素值变成255，为了后续的处理

        # # 第二种方法
        # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(chanel2, 8, ltype=cv2.CV_32S)
    else:
        out_chanel = input_chanel
    return out_chanel


def pic_preprocess(input_img):
    """
        这个函数的功能是，将颅脑区域之外的东西剔除掉，所用的方法是计算连通域的最大面积区域
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
    img_gray[img_gray < 70] = 0
    img_gray[img_gray > 0] = 1
    # 计算图片连通域面积最大的区域，保留结果，得到颅脑区域
    brain_max_area = max_area_extract(img_gray)
    brain_max_area = brain_max_area.astype(np.uint8)  # 这里需要将brain_max_area数组的float32类型转换成uint8类型，
    # 不然后面的相乘操作得到的是float32类型的数据，在cv2.imshow的时候会还是黑白的图像，只能显示整型数据
    # 将原始图片input_img分离通道，用一个通道和颅脑区域相乘
    r, g, b = cv2.split(input_img)
    # cv2.imshow('r', r)
    # 用相乘操作，去除颅脑区域之外的像素
    brain_max_area = np.multiply(brain_max_area, r)
    # cv2.imshow('brain_max_area', brain_max_area)
    # 合并成一个三通道的图片
    brain_max_area = cv2.merge([brain_max_area, brain_max_area, brain_max_area])
    # cv2.imshow('brain_max_area_3', brain_max_area)

    return brain_max_area


# # 读取输入图片
# img0 = cv2.imread("F:\\brain_injury\\z_ct_big_origin\\chenbaojun (11).png")
# print('img0.dtype', img0.dtype)
# img_preprocess = pic_preprocess(img0)
# cv2.imshow('img0', img0)
# cv2.imshow('img_preprocess', img_preprocess)
# cv2.waitKey(0)

if __name__ == '__main__':
    root = 'D:\\projects\\brain_injury\\z_data_classify_preprocess\\ebu_1130\\ct_small_resized\\jiasu'
    root_img = root + "\\1"
    root_processed = root + "\\1_denoise\\"
    if not os.path.exists(root_processed):
        os.makedirs(root_processed)
    imgList_img = os.listdir(root_img)
    for count in range(0, len(imgList_img)):
        print("\nprocessing the " + str(count + 1) + "th" + " picture.......")
        im_name = imgList_img[count]
        im_path = os.path.join(root_img, im_name)
        input = cv2.imread(im_path)
        output = pic_preprocess(input)
        cv2.imwrite(root_processed + im_name, output)