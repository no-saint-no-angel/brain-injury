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

data_transform =transforms.Compose([
                                 ToTensor_img(),
                                 Normalize_img((0.005257,0.004783,0.004360),
                                               (0.063351,0.060287,0.057721))])

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
model_weight_path = "./weights/cla/ebu/deeplabv3plus_resnet180.000332/best_model.th"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
# 三个文件夹（损伤类型）对应三个label
label_list = {'norm': 0, 'jiasu': 1, 'jiansu': 2}
predict_label = ['norm', 'jiasu', 'jiansu', 'unknow']
# 存储所有npy文件的数组
big_single_npy_data = []
# 输入文件路径
root_path = './brain_data/cla/e_bu/val_oneperson'
# 输出文件路径
big_single_npy_data_excel = './data_output/cla/ebu/big_single_npy_data_one_person.xlsx'
class_dir_list = os.listdir(root_path)
for class_dir in class_dir_list:
    # 初始化每个损伤类别的预测个数
    count_norm = 0
    count_jiasu = 0
    count_jiansu = 0
    count_unknow = 0
    print('Processing the ' + str(class_dir) + '.'*10)
    class_dir_path = os.path.join(root_path, class_dir)
    person_dir_list = os.listdir(class_dir_path)
    for person_dir in person_dir_list:
        print('Processing the ' + str(person_dir) + '.'*10)
        person_dir_path = os.path.join(class_dir_path, person_dir)
        data_npy_list = os.listdir(person_dir_path)
        # 存储每个案例的预测结果
        person_result = []
        for data_npy in data_npy_list:
            data_npy_path = os.path.join(person_dir_path, data_npy)
            img = data_transform(np.load(data_npy_path))
            img = torch.unsqueeze(img, dim=0)
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device)))
                predict = torch.softmax(output, dim=0).cpu()
                predict_cla = torch.argmax(predict).cpu()
                predict_cla = predict_cla.numpy()
                print('Label:', class_dir, '  Predict:', class_indict[str(predict_cla)], '  Probability:', predict[predict_cla].numpy())
                single_npy_data = [person_dir+'-'+data_npy, float(label_list[str(class_dir)]), int(predict_cla), np.array(predict)[0], np.array(predict)[1], np.array(predict)[2]]
                big_single_npy_data.append(single_npy_data)
                person_result.append(int(predict_cla))
        # 根据预测结果判断该案例是何种损伤类型
        num_jiasu = np.sum((np.array(person_result) == 1))
        num_jiansu = np.sum((np.array(person_result) == 2))
        num_norm = np.sum((np.array(person_result) == 0))
        print(person_dir+' have '+str(len(person_result))+' .npy file.')
        print('The number of jiasu: ', num_jiasu, ' jiansu: ', num_jiansu, ' norm is:', num_norm)
        # 如果加速或者减速的个数小于等于2，就认为是正常案例，否则根据加速减速个数大小来判断损伤类型
        if num_jiasu + num_jiansu < 3:
            person_injury_class = 0
            count_norm += 1
        else:
            if num_jiansu < num_jiasu:
                person_injury_class = 1
                count_jiasu += 1
            elif num_jiansu > num_jiasu:
                person_injury_class = 2
                count_jiansu += 1
            else:
                person_injury_class = 3
                count_unknow += 1
                print('The number of acceleration and deceleration is ', num_jiansu)
        print('The true injury type of ' + person_dir + ' is ' + class_dir)
        print('The predict injury type of ' + person_dir + ' is ' + predict_label[person_injury_class])
        # 对每个案例保存一行最终的损伤类型结果
        person_injury_result_final = [person_dir, class_dir, predict_label[person_injury_class]
            , num_jiasu/len(person_result), num_jiansu/len(person_result), num_norm/len(person_result)]
        big_single_npy_data.append(person_injury_result_final)

    # 打印某个损伤类型的预测情况
    print('***'*100)
    print(class_dir+' injury type have '+str(len(person_dir_list))+' person, and the predict jiasu:',count_jiasu,
          ' jiansu:', count_jiansu, ' norm:', count_norm, ' unknow:', count_unknow)

big_single_npy_data = pd.DataFrame(big_single_npy_data)
writer = pd.ExcelWriter(big_single_npy_data_excel)
big_single_npy_data.to_excel(writer, 'big_single_npy_data', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()
writer.close()


