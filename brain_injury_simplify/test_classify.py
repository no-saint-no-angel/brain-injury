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

# # ebu
# data_transform =transforms.Compose([
#                                  ToTensor_img(),
#     Normalize_img((0.002617, 0.002439, 0.002107),
#                   (0.041543, 0.040433, 0.037582))])
# zhenbu
data_transform =transforms.Compose([
                                 ToTensor_img(),
        Normalize_img((0.003895,0.003372,0.003421),
                      (0.049658,0.045803,0.046156))])

# # load image
# img = np.load("brain_data/cla/nie_bu/val/norm/chouweiying_567.npy")
# # plt.imshow(img)
# # [N, C, H, W]
# img = data_transform(img)
# # expand batch dimension
# img = torch.unsqueeze(img, dim=0)

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
model_weight_path = "/home/bigdong/projects/brain_injury/brain_injury/weights/cla/zhen" \
                    "_bu_1215/2500deeplabv3plus_resnet180.000316/best_model.th"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
# 三个文件夹（损伤类型）对应三个label
label_list = {'norm': 0, 'jiasu': 1, 'jiansu': 2}
# 存储所有npy文件的数组
big_single_npy_data = []
# 输入文件路径
root_path = '/home/bigdong/projects/brain_injury/brain_injury/brain_data/cla/zhen_bu_1215/val'
# 输出文件路径
big_single_npy_data_excel = './data_output/cla/zhenbu_1215/big_single_npy_data_12_15.xlsx'
class_dir_list = os.listdir(root_path)
for class_dir in class_dir_list:
    print('Processing the ' + str(class_dir) + '.'*10)
    class_dir_path = os.path.join(root_path, class_dir)
    data_npy_list = os.listdir(class_dir_path)
    for data_npy in data_npy_list:
        data_npy_path = os.path.join(class_dir_path, data_npy)
        img = data_transform(np.load(data_npy_path))
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device)))
            predict = torch.softmax(output, dim=0).cpu()
            predict_cla = torch.argmax(predict).cpu()
            predict_cla = predict_cla.numpy()
            print('Label:', class_dir, '  Predict:', class_indict[str(predict_cla)], '  Probability:', predict[predict_cla].numpy())
            single_npy_data = [data_npy, float(label_list[str(class_dir)]), int(predict_cla), np.array(predict)[0], np.array(predict)[1], np.array(predict)[2]]
            big_single_npy_data.append(single_npy_data)

big_single_npy_data = pd.DataFrame(big_single_npy_data)
writer = pd.ExcelWriter(big_single_npy_data_excel)
big_single_npy_data.to_excel(writer, 'big_single_npy_data', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()
writer.close()


