import torch
import torch.nn as nn
from torchvision import transforms
from classification_file.custom_dataset_npy import MyDataset
from classification_file.custom_transforms_mine import ToTensor_img, Normalize_img, RandomScaleCrop, RandomHorizontalFlip
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import pandas as pd
import network_big.network_resnet_aspp as network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 超参数
epochs = 100
batch_size = 16
lr = 0.0003
best_acc = 0.0
num_classes = 3
model_type = 'deeplabv3plus_resnet18'
injury_location = 'nie_bu'
# niebu
data_transform = {
    "train": transforms.Compose([
                                 ToTensor_img(),
        Normalize_img((0.002251,0.002072,0.001802),
                      (0.036181,0.035072,0.032817))
    ]),
    "val": transforms.Compose([
                               ToTensor_img(),
        Normalize_img((0.002251,0.002072,0.001802),
                      (0.036181,0.035072,0.032817))
    ])}
# ebu
# data_transform = {
#     "train": transforms.Compose([
#                                  ToTensor_img(),
#         Normalize_img((0.002617,0.002439,0.002107),
#                       (0.041543,0.040433,0.037582))
#     ]),
#     "val": transforms.Compose([
#                                ToTensor_img(),
#         Normalize_img((0.002617,0.002439,0.002107),
#                       (0.041543,0.040433,0.037582))
#     ])}

# # zhenbu
# data_transform = {
#     "train": transforms.Compose([
#                                  ToTensor_img(),
#         Normalize_img((0.003895,0.003372,0.003421),
#                       (0.049658,0.045803,0.046156))
#     ]),
#     "val": transforms.Compose([
#                                ToTensor_img(),
#         Normalize_img((0.003895,0.003372,0.003421),
#                       (0.049658,0.045803,0.046156))
#     ])}


data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
image_path = data_root + "/brain_data/cla/" + injury_location + '/'# flower data set path

train_dataset = MyDataset(dir_path=image_path+"train",
                                     transform=data_transform["train"])  # data_transform["train"]
train_num = len(train_dataset)

# # {'jiansu':0, 'jiasu':1}
# flower_list = train_dataset.class_to_idx
# cla_dict = dict((val, key) for key, val in flower_list.items())
# # write dict into json file
# json_str = json.dumps(cla_dict, indent=4)
# with open('class_indices.json', 'w') as json_file:
#     json_file.write(json_str)


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=4)

validate_dataset = MyDataset(dir_path=image_path + "val",
                                        transform=data_transform["val"])  # data_transform["val"]
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=4)

# 初始化网络模型-------------------------------------------------------------
# resnet
# net = network.model_name('resnet50', num_class=num_classes, pretrained_backbone=True,
#                          replace_stride_with_dilation=[False, False, True])
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

net = model_map[model_type](num_classes=num_classes, output_stride=16)
# 把网络放入GPU或CPU -------------------------------------------------------------
net.to(device)
# # load pretrain weights
# model_weight_path = "./pre_weights/resnet50-pre.pth"
# # 载入模型参数方法一：
# pretrained_dict = torch.load(model_weight_path, map_location=torch.device('cpu'))

model_dict = net.state_dict()  # pytorch自动初始化
#
# # 将pretrained_dict里不属于model_dict的键剔除掉
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # # # 更新现有的model_dict
# model_dict.update(pretrained_dict)
# # # 加载我们真正需要的state_dict
net.load_state_dict(model_dict)
# 保存路径 -------------------------------------------------------------
experiments_dir = './weights/cla/' + injury_location + '/' + str(model_type)+str(lr)+str(batch_size)
if not os.path.exists(experiments_dir):
    os.makedirs(experiments_dir)
save_path = os.path.join(experiments_dir, 'best_model.th')
train_loss_log_path = os.path.join(experiments_dir, 'train_loss_log.csv')
val_acc_log_path = os.path.join(experiments_dir, 'val_acc_log.csv')
val_loss_log_path = os.path.join(experiments_dir, 'val_loss_log.csv')
train_acc_log_path = os.path.join(experiments_dir, 'train_acc_log.csv')
# 损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
# 保存train_loss和val_acc
train_loss = []
val_acc = []
val_loss = []
train_acc = []
for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    acc_train = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data.values()
        optimizer.zero_grad()
        logits = net(images.to(device))
        # print('logits:', logits.shape)
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # train acc
        outputs = net(images.to(device))  # eval model only have last output layer
        predict_y = torch.max(outputs, dim=1)[1]
        acc_train += (predict_y == labels.to(device)).sum().item()

        # save_loss
        train_loss.append(loss.cpu().item())
        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()
    train_accurate = acc_train / train_num
    # save acc
    train_acc.append(train_accurate)
    # validate
    net.eval()
    acc_train = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data.values()
            # val acc
            outputs = net(val_images.to(device))  # eval model only have last output layer

            predict_y = torch.max(outputs, dim=1)[1]
            acc_train += (predict_y == val_labels.to(device)).sum().item()
            # val loss
            loss_val = loss_function(outputs, val_labels.to(device))
            # save_loss
            val_loss.append(loss_val.cpu().item())
        val_accurate = acc_train / val_num
        # save acc
        val_acc.append(val_accurate)
        if val_accurate > best_acc:
            best_epoch = epoch+1
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
        print('The best test_accuracy is: %.3f  epoch: %d' % (best_acc, best_epoch))


train_loss = pd.DataFrame({'train loss': train_loss})
train_loss.to_csv(train_loss_log_path, index=False, sep=',')
train_acc = pd.DataFrame({'train acc': train_acc})
train_acc.to_csv(train_acc_log_path, index=False, sep=',')

val_loss = pd.DataFrame({'val loss': val_loss})
val_loss.to_csv(val_loss_log_path, index=False, sep=',')
val_acc = pd.DataFrame({'val acc': val_acc})
val_acc.to_csv(val_acc_log_path, index=False, sep=',')
print('Finished Training')



