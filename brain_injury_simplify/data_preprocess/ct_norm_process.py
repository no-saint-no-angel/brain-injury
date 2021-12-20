"""
    这个文件的功能是，将正常颅脑切片图像处理成可进行分类训练的数据。分以下几个步骤：
    1、正常颅脑图像的尺寸是1024*1024，需要把图片都调整到320*320尺寸，这里的尺寸调整不需要用到data_resize文件，
    简单的resize函数就行
    2、然后再进行mask模板的生成，
    3、然后再使用generate_npy_from_pic文件生成.npy文件数据（不在本文件内操作）。
    注：这个功能可以集成到一个文件，或者说同时来完成，但是这个正常颅脑数据只需要生成一次就好了，所以没有必要再去
    花心思集成到一个文件。
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def pic_resize(img):
    output_img = img
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


root = 'D:\\projects\\brain_injury\\z_data_classify_preprocess\\norm_ct_data'
root_input_dir = root + "\\origin_data\\"
# 输出文件夹
root_output_dir_img = root + "\\processed_data\\img\\"
if not os.path.exists(root_output_dir_img):
    os.makedirs(root_output_dir_img)
root_output_dir_mask = root + "\\processed_data\\mask\\"
if not os.path.exists(root_output_dir_mask):
    os.makedirs(root_output_dir_mask)
dir_List = os.listdir(root_input_dir)
for i in range(0, len(dir_List)):
    # 每个案例的输入图片文件夹
    img_input_dir_0 = os.path.join(root_input_dir, dir_List[i])
    img_input_dir_1 = os.listdir(img_input_dir_0)
    for j in range(0, len(img_input_dir_1)):
        imgList_img_dir = os.path.join(img_input_dir_0, img_input_dir_1[j])
        imgList_img = os.listdir(imgList_img_dir)
        for count in range(0, len(imgList_img)):
            print("\nprocessing the " + str(count + 1) + "th" + " picture.......")
            im_name = imgList_img[count]
            im_name_before_point = im_name.split('.')[0]
            im_path = os.path.join(imgList_img_dir, im_name)
            # imread不能直接读取中文路径下的图片
            # img_input = cv2.imread(im_path)
            # 先用numpy读取int型数据，再读取图片数据
            img_input = cv2.imdecode(np.fromfile(im_path, dtype=np.uint8), 1)
            if im_name_before_point != 'Thumbs':
                # 调整尺寸到320*320
                output_img = pic_resize(img_input)
                # cv2.imshow('output_img', output_img)
                # cv2.waitKey(0)
                # 生成mask
                output_mask = np.zeros((320, 320, 3), np.uint8)
                # 保存img和mask到相应文件夹，并修改名字和后缀
                cv2.imwrite(root_output_dir_img + '\\' + str(i) + '_' + str(count) + '.bmp', output_img)
                cv2.imwrite(root_output_dir_mask + '\\' + str(i) + '_' + str(count) + '.bmp', output_mask)
