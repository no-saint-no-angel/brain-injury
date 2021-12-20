"""
    这个文件的目的是：
    针对人工标注的图片在每个案例的文件夹下，以及每个案例下有的有分割区域，有的没有分割区域的问题。
    使用代码，将有分割区域的图片（含有json文件)放置到一个文件夹下，后续使用批量转换代码，生成训练数据；
    没有的图片，生成一个黑板图片，并分开放置到另外两个一个文件夹下（train/val)。

    注意：这程序要运行两遍，由于存在缓存缩略图（Thumbs.db）的情况，第一遍可能还剩下一张图片，第二遍可以把剩下的图片放入目标位置
"""


import os
import shutil
import numpy as np
import cv2

root = 'D:\\projects\\brain_injury\\z_data_classify_preprocess'
# 输入文件夹
root_input_dir = root + "\\ebu\\ct_small_resized\\jiansu\\"
# 输出文件夹
root_output_dir = root + "\\ebu\\bmp_json_spilit_data_output\\jiansu\\"
if not os.path.exists(root_output_dir):
    os.makedirs(root_output_dir)
data_output_bmp_json = root_output_dir + "\\bmp_json"
if not os.path.exists(data_output_bmp_json):
    os.makedirs(data_output_bmp_json)
data_output_only_bmp = root_output_dir + "\\only_bmp"
if not os.path.exists(data_output_only_bmp):
    os.makedirs(data_output_only_bmp)

dir_List = os.listdir(root_input_dir)
for i in range(0, len(dir_List)):
    # 每个案例的输入图片文件夹
    img_input_dir = os.path.join(root_input_dir, dir_List[i])
    print("\nprocessing the " + str(img_input_dir) + " folder")
    # # 每个案例的输出图片文件夹
    # img_output_dir = os.path.join(root_output_dir, dir_List[i])
    # if not os.path.exists(img_output_dir):
    #     os.makedirs(img_output_dir)

    imgList_img = os.listdir(img_input_dir)
    # imgList_img.sort(key=lambda x: x.split('.')[0])

    if len(imgList_img) <= 2:
        for i_0 in range(len(imgList_img)):
            im_name_0 = imgList_img[i_0]
            im_name_before_point_0 = im_name_0.split('.')[0]
            if im_name_before_point_0 != "Thumbs":
                shutil.move(os.path.join(img_input_dir, im_name_0), data_output_only_bmp)
    elif len(imgList_img) > 2:
        # 如果第一张图片是Thumbs.db，那就从第二张图片开始
        if imgList_img[0].split('.')[0] == "Thumbs":
            count = 1
        else:
            count = 0
        while count < len(imgList_img)-1:
            # 当前文件
            im_name_0 = imgList_img[count]
            im_name_before_point_0 = im_name_0.split('.')[0]
            # im_path_0 = os.path.join(img_input_dir, im_name_0)
            # 后一个文件
            im_name_1 = imgList_img[count+1]
            im_name_before_point_1 = im_name_1.split('.')[0]
            # im_path_1 = os.path.join(img_input_dir, im_name_1)
            # 如果当前文件的名字和后一个文件的名字一样，则将这两个文件一起移动到，指定的bmp和json文件夹下，
            # 并将count更新为+2
            if im_name_before_point_0 == im_name_before_point_1:
                shutil.move(os.path.join(img_input_dir, im_name_0), data_output_bmp_json)
                shutil.move(os.path.join(img_input_dir, im_name_1), data_output_bmp_json)
                count += 2
            else:  # 如果不一样，就将当前文件移动到only_bmp文件夹下
                if im_name_before_point_0 != "Thumbs":
                    shutil.move(os.path.join(img_input_dir, im_name_0), data_output_only_bmp)
                    count += 1