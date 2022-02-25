import cv2
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os

import network_big.network_resnet_aspp as network
from classification_file.custom_transforms_mine import ToTensor_img, Normalize_img, RandomScaleCrop, RandomHorizontalFlip

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 超参数
buwei = "nie_bu/"

# 0113 foreground
# # zhen
# data_transform =transforms.Compose([
#                                  ToTensor_img(),
#             Normalize_img((0.028741075, 0.042851698, 0.052193183, 0.053777132, 0.055602387, 0.054543514, 0.05322038,
#                            0.04975702, 0.048499312, 0.047692165, 0.035568357),
#                           (0.06229688, 0.07148748, 0.075318076, 0.07997214, 0.08599801, 0.086562574, 0.086359866,
#                            0.078616984, 0.07670473, 0.07281253, 0.06652234))
#                                                ])
# # e
# data_transform =transforms.Compose([
#                                  ToTensor_img(),
#         Normalize_img((0.024423284, 0.0395244, 0.049368355, 0.050226655, 0.050920025, 0.050512586, 0.04724509,
#                        0.04579699, 0.04330135, 0.04259867, 0.030353326),
#                       (0.054849934, 0.06817282, 0.07322584, 0.07798163, 0.08169261, 0.08565484, 0.082851276,
#                        0.08047398, 0.0745105, 0.069362596, 0.05962247))
#                                                ])
# # nie
data_transform =transforms.Compose([
                                 ToTensor_img(),
            Normalize_img((0.023350995, 0.037016116, 0.04961458, 0.04976879, 0.047215525, 0.04567317, 0.043022927,
                           0.03976199, 0.03989872, 0.040963583, 0.030989274),
                          (0.057115704, 0.07067053, 0.07545463, 0.075328365, 0.07267302, 0.07227043, 0.0690233,
                           0.064079516, 0.06535724, 0.06412881, 0.05721703))
])


# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
# deeplab
model_map = {
        'deeplabv3plus_resnet18': network.deeplabv3plus_resnet18,
        'deeplabv3plus_resnet34': network.deeplabv3plus_resnet34,
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        # 'deeplabv3plus_xception': network.deeplabv3plus_xception,
        # 'upernet_resnet101': network.upernet_resnet101,
    }

model = model_map['deeplabv3plus_resnet18'](num_classes=3, output_stride=16)
# 把网络放入GPU或CPU -------------------------------------------------------------
model.to(device)
# load model weights
model_weight_path = "./weights/cla_data_shanchu_xinzeng_nie_0224/11ch/"+buwei+"2500deeplabv3plus_resnet180.00014/best_model.th"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
# 三个文件夹（损伤类型）对应三个label
label_list = {'norm': 0, 'jiasu': 1, 'jiansu': 2}
predict_label = ['norm', 'jiasu', 'jiansu', 'unknow']
# 存储所有npy文件的数组
big_single_npy_data = []
# 输入文件路径
root_path = './brain_data/cla_final_0222_valid/data_anli_npy_11ch_mangce_new/'+buwei
# 输出文件路径
big_single_npy_data_dir = './data_output/cla_mangce/cla_data_shanchu/11ch/'+buwei
if not os.path.exists(big_single_npy_data_dir):
    os.makedirs(big_single_npy_data_dir)
big_single_npy_data_excel = os.path.join(big_single_npy_data_dir, 'single_npy_single_person_mangce_0224.xlsx')
class_dir_list = os.listdir(root_path)
# for class_dir in class_dir_list:
# 初始化每个损伤类别的预测个数
count_norm = 0
count_jiasu = 0
count_jiansu = 0
count_unknow = 0
#     print('Processing the ' + str(class_dir) + '.'*10)
#     class_dir_path = os.path.join(root_path, class_dir)
person_dir_list = os.listdir(root_path)
num_jiajian_dict = {'norm': 0, 'jiasu': 0, 'jiansu': 0}
for person_dir in person_dir_list:
    print('Processing the ' + str(person_dir) + '.'*50)
    person_dir_path = os.path.join(root_path, person_dir)
    data_npy_list = os.listdir(person_dir_path)
    # 存储每个案例的预测结果
    predict_result_dict = {}
    person_result = []
    for data_npy in data_npy_list:
        data_npy_path = os.path.join(person_dir_path, data_npy)
        # npy格式
        img = data_transform(np.load(data_npy_path))
        # bmp格式
        # img = data_transform(cv2.imread(data_npy_path))

        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device)))
            predict = torch.softmax(output, dim=0).cpu()
            predict_cla = torch.argmax(predict).cpu()
            predict_cla = predict_cla.numpy()
            predict_prob = predict[predict_cla].numpy()
            print('  Predict:', class_indict[str(predict_cla)], '  Probability:', predict_prob)
            # 将当前案例中出现的预测情况作为键，对应的概率作为值，存入字典中，相同的情况进行概率求和
            if class_indict[str(predict_cla)] not in predict_result_dict.keys():
                predict_result_dict[class_indict[str(predict_cla)]] = 0
            predict_result_dict[class_indict[str(predict_cla)]] += predict_prob

            single_npy_data = [person_dir + '-' + data_npy, int(predict_cla),
                               np.array(predict)[0], np.array(predict)[1], np.array(predict)[2]]
            big_single_npy_data.append(single_npy_data)
            person_result.append(int(predict_cla))
    # 对字典的值进行排序，输出最大值对应的损伤类型
    print(predict_result_dict)
    ans = sorted(predict_result_dict.items(), key=lambda d: d[1], reverse=True)
    # 根据预测结果判断该案例是何种损伤类型
    num_jiasu = np.sum((np.array(person_result) == 1))
    num_jiansu = np.sum((np.array(person_result) == 2))
    num_norm = np.sum((np.array(person_result) == 0))
    print(person_dir + ' have ' + str(len(person_result)) + ' .npy file.')
    print('The number of jiasu: ', num_jiasu, ' jiansu: ', num_jiansu, ' norm is:', num_norm)

    # print('The true injury type of ' + person_dir + ' is ' + class_dir)
    print('The predict injury type of ' + person_dir + ' is ' + ans[0][0])
    # 对每个案例保存一行最终的损伤类型结果
    person_injury_result_final = [person_dir, ans[0][0]
        , num_norm / len(person_result), num_jiasu / len(person_result), num_jiansu / len(person_result)]
    big_single_npy_data.append(person_injury_result_final)
    # 存储每种损伤类型文件夹的预测结果个数
    num_jiajian_dict[ans[0][0]] += 1

# 打印某个损伤类型的预测情况
print('***'*100)
print(str(person_dir_list)+' injury type have '+str(len(person_dir_list))+' person, and the predict jiasu:',num_jiajian_dict['jiasu'],
          ' jiansu:', num_jiajian_dict['jiansu'], ' norm:', num_jiajian_dict['norm'], ' unknow:', count_unknow)

big_single_npy_data = pd.DataFrame(big_single_npy_data)
writer = pd.ExcelWriter(big_single_npy_data_excel)
big_single_npy_data.to_excel(writer, 'big_single_npy_data', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()
writer.close()


