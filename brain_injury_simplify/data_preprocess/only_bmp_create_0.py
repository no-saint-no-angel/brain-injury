"""
    这个文件的功能是：对于没有分割区域的样本图片，生成一张零的黑板来作为分割标签
"""

import os
import numpy as np
import cv2

root = "D:\\projects\\brain_injury\\z_data_classify_preprocess\\ebu\\bmp_js" \
       "on_spilit_data_output_to_seg\\jiasu\\only_bmp"
# 输入文件夹
root_input_dir = root + "\\images\\"
# 输出文件夹
root_output_dir = root + "\\masks"

if not os.path.exists(root_output_dir):
    os.makedirs(root_output_dir)

imgList_img = os.listdir(root_input_dir)
for count in range(0, len(imgList_img)):
    print("\nprocessing the " + str(count + 1) + "th" + " picture.......")
    im_name = imgList_img[count]
    # im_name_before_point = im_name.split('.')[0]
    # im_path = os.path.join(img_input_dir, im_name)
    # img_input = cv2.imread(im_path)
    output = np.zeros((320, 320, 3), np.uint8)
    cv2.imwrite(root_output_dir + '\\' + im_name, output)