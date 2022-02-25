"""
    2022/1/12/22点18分
    这个文件的功能是实现新特征的设计，为了每张特征图可以更完全的代表这个案例的脑部损伤类型。
    具体来说，根据每个案例的切片数量，来为每个案例生成一张到两张，或者三张特征图，并且保证每张特征图可以较全面的代表该案例的脑部信息。
    由于网络结构和训练方式的限制，每张特征图的通道数量需要保证一致，也即是每张特征图包含的切片数量需要一致，这个值，需要进一步对所有
    案例的切片数量进行一个统计操作。目前目测的数量是11，两张特征图的切片数量是22，分界点是17，三张是33，分界点是28.多了的舍弃前面和
    后面的片子，至于舍弃的规则（初步设定舍弃前面的多于后面的），少了的就补充全为零的黑板。

    具体实现：
    1、对于输入：
        之前是对于每个损伤类型，所有案例都是再同一个文件夹里，需要根据不同的命名来判断不同案例。这次我想分开来做，
        先把每个案例的片子放到同一个以案例名命名的小文件夹里，然后再来对单个小文件夹的图片进行读取。
        在操作这个步骤的时候，可以统计一下单个损伤类型的切片数量分布直方图。
    2、对于单个案例，根据切片的数量，确定合成到一张还是多张特征图。
        对于舍弃片子的原则：
           由于排在前面的切片是靠近脖子的部分，一般不会有损伤，最后的片子是靠近头部顶端的，一般也不会有损伤。
        但是，排在前面的无用信息会多一点，所以可以整一个字典的键值对来存储各种情况。键表示舍弃片子数量的情况，值表示舍弃最后
        片子的数量。比如对于单张特征图含有11张切片的情况，{1:1, 2:1, 3:1, 4:1, 5:1, 6:2}，或者更简单的直接用一个阈值解决。
    3、对于读取问题：
          对于一张，直接连续读取；对于两张，隔一张读取；对于N张，隔N-1张读取。
          具体实现，先根据合成的特征图数量，来创建新的特征图，根据上面提到的读取原则来对原图和分割mask进行读取和加权，
          然后以此填充到新建的空白特征图中。
    4、对于格式的输出，和之前的npy一样，3*320*320.
"""
import copy
import os
import numpy as np
import cv2
import math


def return_single_chenal(pic):
    b, g, r = cv2.split(pic)
    # b = np.expand_dims(b, axis=2)
    return b


def img_mask_to_weighted_single_ch(img_path, mask_path, bg_weight):
    img = cv2.imread(img_path)
    img_b = return_single_chenal(img)
    mask = cv2.imread(mask_path)
    mask_b = return_single_chenal(mask)
    # 设置权重
    mask_b_clone = copy.deepcopy(mask_b)
    mask_b_clone = mask_b_clone.astype(np.float16)
    mask_b_clone[mask_b == 0] = bg_weight

    img_b = img_b.astype(np.float16)
    # 加权
    ans = np.multiply(mask_b_clone, img_b)
    return ans


def generate_npy_11ch(input_anli_img, input_anli_mask, output_dir, background_weight):
    """

    :param input_anli_img: 输入的原始图片文件夹，该目录下包含N个案例小文件夹
    :param input_anli_mask: 输入的分割图片文件夹，该目录下包含N个案例小文件夹
    :param output_dir: 输出文件夹，该目录下包含所有案例的11通道的npy文件
    :return:
    """

    anli_img_dir_list = os.listdir(input_anli_img)
    print("该输入文件下下包含 "+str(len(anli_img_dir_list))+" 个案例"+"*"*50)
    for _anli_img_dir in anli_img_dir_list:
        # 每个案例的文件夹路径
        _anli_img_dir_path = os.path.join(input_anli_img, _anli_img_dir)
        _anli_mask_dir_path = os.path.join(input_anli_mask, _anli_img_dir)
        img_list = os.listdir(_anli_img_dir_path)  # 这里debug的时候看看文件夹下的图片顺序乱不乱，乱的话需要排序
        # 根据切片数量判断合成npy的数量
        # 这里分成四个区间[(0,17),(18,28),(28,39),(39,~)],再在对应的四个区间内进行[11,22,33,44]进行判断，决定是舍弃还是补充图片
        num_npy = 0   # 合成新特征图的数量
        num_trim = 0  # 需要配平到11的倍数的差值，正数表示舍弃的切片数量，负数表示填充的
        num_img_to_npy_stand_list = [11, 22, 33, 44]
        num_img = len(img_list)
        # 当案例切片数量太少时，舍弃该案例
        if num_img < 7:
            print("案例 " + str(_anli_img_dir) + " 只包含 " + str(num_img) + " 张片子，故舍弃")
            continue

        print("案例 " + str(_anli_img_dir) + " 包含 " + str(num_img) + " 张片子"+"*"*20)
        if num_img < 28:
            if num_img > 17:  # (18,28)
                num_npy = 2
                num_trim = num_img - num_img_to_npy_stand_list[1]
            else:             # (0,17)
                num_npy = 1
                num_trim = num_img - num_img_to_npy_stand_list[0]
        else:
            if num_img > 39:  # (39,~)
                num_npy = 4
                num_trim = num_img - num_img_to_npy_stand_list[3]
            else:
                num_npy = 3   # (28,39)
                num_trim = num_img - num_img_to_npy_stand_list[2]
        print("案例 " + str(_anli_img_dir) + " 合成 " + str(num_npy) + " 张特征图"+"*"*20)
        # 根据npy和需要配平的切片数量，来读取原始图片和mask，从而来合成npy
        # 这里需要解决填充和舍弃的切片数量的分配问题，所谓分配就是，前面填充或者舍弃多少，后面填充或者舍弃多少张切片
        # 对于舍弃，按照舍弃原则；对于填充，就对称填充就好。这里引入两根指针来实现
        # 读取的指针,主要针对多出来的片子需要舍弃一部分的操作
        trim_front_imread = 0
        trim_later_imread = num_img
        dict_abandon = {1:0, 2:0, 3:1, 4:1, 5:1, 6:2}  # 键表示对应多出来的图片，值表示后面舍弃切片的数量
        if num_trim > 0:
            trim_front_imread = num_trim - dict_abandon[num_trim]
            trim_later_imread -= dict_abandon[num_trim]
            print("前面舍弃 "+str(num_trim - dict_abandon[num_trim])+" 张切片，"+"后面舍弃 "+str(
                dict_abandon[num_trim])+" 张切片")
        # 写入的指针，主要针对填充的案例，需要填充的片子的操作，不过写入的时候，使用front一根指针就行
        trim_front_imwrite = 0
        trim_later_imwrite = num_img
        if num_trim < 0:
            trim_front_imwrite = round(np.abs(num_trim)/2)
            trim_later_imwrite -= np.abs(num_trim) - trim_front_imwrite
            print("前面填充 " + str(round(np.abs(num_trim)/2)) + " 张切片，" + "后面填充 " + str(
                np.abs(num_trim) - trim_front_imwrite) + " 张切片")

        # 根据num_npy初始化空的npy
        npy_tmp = np.zeros([11*num_npy, 320, 320])
        # 开始读取片子
        print("选取该案例的第 " + str(trim_front_imread) + " 到第 " + str(trim_later_imread) + " 张切片，共"+str(num_npy)+"个楼层")
        for i in range(trim_front_imread, trim_later_imread):
            img_path = os.path.join(_anli_img_dir_path, img_list[i])
            mask_path = os.path.join(_anli_mask_dir_path, img_list[i])

            img_weighted = img_mask_to_weighted_single_ch(img_path, mask_path, background_weight)
            # 写入大矩阵中
            # 需要根据i的值来确定到底放到大矩阵的哪一层楼（层数即num_npy）
            # 这里只能具体问题具体分析，凑一凑
            # 这里用trim_front_imwrite + i对num_npy取余来确定，当前img_weighted应该放入哪一层楼
            # 这里用trim_front_imwrite + i对num_npy的商, 并且减去 trim_front_imread 来确定，当前img_weighted应该
            # 放入固定层楼的第几个位置,需要向上取整，这里的逻辑太绕了，需要画示意图理理
            num_louceng = (trim_front_imwrite + i)%num_npy
            num_louceng_weizhi = math.ceil((trim_front_imwrite + i)/num_npy) - trim_front_imread
            if num_louceng == 0:
                num_louceng = num_npy  # 因为取余是零的话，说明在最顶楼，比如3%3=0，第三张片子要放入第三层
                # num_louceng_weizhi -= 1  #
            print("其中第 "+str(i)+" 张切片在第"+str(num_louceng)+"层楼的第"+str(num_louceng_weizhi)+"个位置")
            npy_tmp[11*(num_louceng-1) + num_louceng_weizhi, :, :] = img_weighted  # 正式写入

        for i in range(num_npy):
            npy_tmp_small = npy_tmp[i*11:(i+1)*11, :, :]
            npy_tmp_small_name = _anli_img_dir+'_'+str(i)
            np.save(os.path.join(output_dir, npy_tmp_small_name), npy_tmp_small)


if __name__ == '__main__':
    root = "D:\\projects\\brain_injury\\z_data_classify_preprocess\\zz_final_0" \
           "221\\bmp_json_spilit_data_output_to_seg\\nie\\jiasu\\jiasu_all\\new_data_npy_0222"
    input_img = root + "\\single_anli_dir_png\\images\\"
    input_mask = root + "\\single_anli_dir_png\\masks\\"
    bg_weight = 0.2
    output = root + "\\data_npy_11ch\\"
    if not os.path.exists(output):
        os.makedirs(output)
    generate_npy_11ch(input_img, input_mask, output, background_weight=bg_weight)
















