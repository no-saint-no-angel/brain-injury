"""

    这个文件的功能是，根据加速和减速的images和masks，将每个案例的图片按照三张相邻的图片拼接起来，得到九个通道的输出数据块
    前三个通道是原始图片，中间三个通道是对应的mask，后面三个通道是mask点成images。
    其实关于九个通道的设定有疑问，如果输入是九个通道，那就不能用预训练模型，如果是三个通道的话，就可以
    所以想试试两种操作
"""
import os
import cv2
import copy
import numpy as np


def return_single_chenal(pic):
    b, g, r = cv2.split(pic)
    b = np.expand_dims(b, axis=0)
    return b


def merge_pic_multi_chenal(pic_1, pic_2, pic_3, mask_1, mask_2, mask_3, background_weight, chenal_num):
    # 组合图片通道
    pic_1 = return_single_chenal(pic_1)
    pic_2 = return_single_chenal(pic_2)
    pic_3 = return_single_chenal(pic_3)
    first_3 = np.concatenate((pic_1, pic_2, pic_3), axis=0)
    # 组合掩膜通道
    # cv2.imshow('mask_1', mask_1*255)
    # cv2.waitKey(0)
    mask_1 = return_single_chenal(mask_1)
    print('mask_1', np.sum(mask_1))
    mask_2 = return_single_chenal(mask_2)
    print('mask_2', np.sum(mask_2))
    mask_3 = return_single_chenal(mask_3)
    print('mask_3', np.sum(mask_3))
    middle_3 = np.concatenate((mask_1, mask_2, mask_3), axis=0)
    # 判断掩膜通道是否有分割区域
    if np.sum(middle_3) > 50:  # 阈值随便给的
        data_injury_type = True
    else:
        data_injury_type = False

    # background_weight = 0.2
    middle_3_weight = copy.deepcopy(middle_3)
    # middle_3_weight == 1 was the foreground
    middle_3_weight[middle_3_weight == 1] = background_weight
    # 组合后三个加权通道
    # 由于失误的原因，在这（七月二十六号）之前的的生成的.npy文件都是直接使用的middle_3（背景为0，前景为1）作为点成的mask(下面一句代码），
    # 原本应该是使用middle_3_weight相乘的，背景为0.2，前景为1，但是效果还行，后续再2考虑使用middle_3_weight
    # last_3 = np.multiply(first_3, middle_3)
    # 枕部和额部的数据都是用middle_3_weight的来加权，时间：十一月十九日
    last_3 = np.multiply(first_3, middle_3_weight)
    # 组合多个个通道
    if chenal_num == 3:
        output = last_3
    else:
        output = np.concatenate((first_3, middle_3, last_3), axis=0)
    # output = np.squeeze(output)
    return output, data_injury_type


def Create_npy(root_input_images_dir, root_input_masks_dir, root_data_npy_injurys, root_data_npy_norm, bg_weight, ch_num):
    images_List = os.listdir(root_input_images_dir)
    images_List = sorted(images_List)
    masks_list = os.listdir(root_input_masks_dir)
    masks_list = sorted(masks_list)
    i = 0
    name_count = 0
    while i < len(images_List)-1:
        # 第一张图片
        pic_1 = images_List[i]
        pic_1_name_before_point_0 = pic_1.split('_')[0]
        pic_1 = os.path.join(root_input_images_dir, pic_1)
        pic_1 = cv2.imread(pic_1)
        mask_1 = masks_list[i]
        mask_1 = os.path.join(root_input_masks_dir, mask_1)
        mask_1 = cv2.imread(mask_1)
        # 第二张图片
        pic_2 = images_List[i+1]
        pic_2_name_before_point_0 = pic_2.split('_')[0]
        pic_2 = os.path.join(root_input_images_dir, pic_2)
        pic_2 = cv2.imread(pic_2)
        mask_2 = masks_list[i+1]
        mask_2 = os.path.join(root_input_masks_dir, mask_2)
        mask_2 = cv2.imread(mask_2)
        # 第三张图片
        pic_3 = images_List[i + 2]
        pic_3_name_before_point_0 = pic_3.split('_')[0]
        pic_3 = os.path.join(root_input_images_dir, pic_3)
        pic_3 = cv2.imread(pic_3)
        mask_3 = masks_list[i + 2]
        mask_3 = os.path.join(root_input_masks_dir, mask_3)
        mask_3 = cv2.imread(mask_3)

        i_minus = 0

        if pic_1_name_before_point_0 == pic_2_name_before_point_0 and pic_1_name_before_point_0==pic_3_name_before_point_0:
            name_count += 1
            print('pic '+str(i+1) + str(i + 2) + str(i + 3)+' pixel value are:')
            output_npy, data_injury_type = merge_pic_multi_chenal(pic_1, pic_2, pic_3, mask_1, mask_2,
                                                                  mask_3, bg_weight, ch_num)
            if data_injury_type:
                print('pic ' + str(i+1) + str(i + 2) + str(i + 3) + ' have injurys')
                data_path = root_data_npy_injurys
            else:
                print('pic ' + str(i+1) + str(i + 2) + str(i + 3) + ' is norm')
                data_path = root_data_npy_norm
            np.save(data_path + pic_1_name_before_point_0 + "_" + str(name_count) + str(name_count+1) +
                    str(name_count+2) + ".npy", output_npy)
        # elif pic_1_name_before_point_0 != pic_2_name_before_point_0:  # 这种情况实际上是不存在的，ABB他排在AAB之后，只要AAB出现了，就会把i跳到B的位置
        #     i_minus = 1
        #     name_count = 0
        else:
            i_minus = 1
            name_count = 0
        # 更新i
        i = i+1+i_minus


if __name__ == '__main__':
    # 组合通道的个数,和背景权重
    chenal_num = 3
    # background_weight = 0.2
    foreground_weight_list = [1, 1.111, 1.25, 1.429, 1.667, 2, 2.5, 3.33, 5, 10]
    foreground_weight  =foreground_weight_list[6]
    # 脑损伤类型
    injury_type = "jiansu/"
    # root = "D:\\projects\\brain_injury\\z_data_classify_preprocess\\bmp_json_spi" \
    #        "lit_data_output_to_seg\\jiansu\\jiansu_all"
    # root = "D:\\projects\\brain_injury\\z_data_classify_preprocess\\zhenbu\\bmp_json_spilit_data" \
    #        "_output_to_seg\\jiasu\\jiasu_all"
    # norm_ct,background_weight=0.2
    # root = "/home/bigdong/projects/brain_injury1/z_data_classify_preprocess/niebu/bmp_" \
    #        "json_spilit_data_output_to_seg/jiasu/nie/jiasu_all"

    root = "/media/bigdong/Seagate Expansion Drive/brain_injury1/z_data_classify_preprocess/e" \
           "bu_1215/bmp_json_spilit_data_output_to_seg/jiansu/jiansu_all"

    # 输入文件夹
    root_input_images_dir = root + "/images_clahe/"
    root_input_masks_dir = root + "/masks/"
    # 输出文件夹
    root_data_npy = root + "/data_npy_3ch_fw2.5/"
    root_data_npy_injurys = root_data_npy + injury_type
    root_data_npy_norm = root_data_npy + "norm/"
    if not os.path.exists(root_data_npy_injurys):
        os.makedirs(root_data_npy_injurys)
    if not os.path.exists(root_data_npy_norm):
        os.makedirs(root_data_npy_norm)

    try:
        print("----------------------------------------------")
        Create_npy(root_input_images_dir, root_input_masks_dir, root_data_npy_injurys,
                   root_data_npy_norm, foreground_weight, chenal_num)

    except Exception as e:
        print(e)
