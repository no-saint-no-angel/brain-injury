import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import torch.optim as optim
from torchsummary import summary

import network_big.network_deeplab_seg as network
# from unet import unet

from segmentation_file.dataset1 import LiverDataset
from collections import defaultdict
from segmentation_file.custom_transforms_mine import *
from segmentation_file.loss import calcu_loss, print_metrics
from segmentation_file.calc_loss import diceCoeffv2
from tensorboardX import SummaryWriter
from segmentation_file.calcu_accuracy import calcu_accuracy
# from scheduler import PolyLR
import torchvision.models.resnet
# from thop import profile
writer = SummaryWriter('log/ebu_plus1215_4bei/log_resnet50_0.1bce_0.0001_epoch100')
# source activate pytorch
# pip3 install pandas

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 超参数
epochs = 100
batch_size = 16
lr = 0.0001
best_loss = 1e10
num_classes = 2
injury_location = 'ebu_plus1215_4bei'
# niebu
# 2通道的320*320 cv数据增强之后  5bei
# normMean = [0.20734909, 0.20734909, 0.20734909]
# normStd = [0.28372735, 0.28372735, 0.28372735]

# new only_bmptobmp_json=1to3
# normMean = [0.22057007, 0.22057007, 0.22057007]
# normStd = [0.28936288, 0.28936288, 0.28936288]

# ebu
# 0.285443,0.285443,0.285443
# 0.330193,0.330193,0.330193

# zhenbu
# 0.245062,0.245062,0.245062
# 0.313987,0.313987,0.313987


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask

# 这个是没有图片增加的数据增强，改写的函数


image_and_mask_transform = DualCompose([
    Normalize_Totensor()])
# 对image和mask都使用的transform


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
            target = t(target)
        return image, target


train_dataset = LiverDataset("./brain_data/seg/"+injury_location+"/train",
                             image_and_mask_transform=image_and_mask_transform)
train_num = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=8)

validate_dataset = LiverDataset("./brain_data/seg/"+injury_location+"/val",
                                image_and_mask_transform=image_and_mask_transform)
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=8)


# 初始化网络模型-------------------------------------------------------------
# unet-------------------------------------------------------------------------------
# net = unet(3, num_classes)

# resnet_with_unet --------------------------------------------------------------------
# net = resnet34(num_classes)
# net = resnet101(3)
# net = resnet152(num_classes)
# summary(net, input_size=(3, 512, 512))
# net = resnet50(num_classes)

# pspnet------------------------------------------------------------------------------
# downsample_factor
#   下采样的倍数
#   16显存占用小
#   8显存占用大
# aux_branch = False
#   是否使用辅助分支
#   会占用大量显存
# pretrained=False
# 不在这里加载预训练模型
# 后面初始化参数的时候再来加载
# net = PSPNet(num_classes=num_classes, backbone="resnet50", downsample_factor=16, pretrained=False, aux_branch=False)
# deeplab -----------------------------------------------------------------------------
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

net = model_map['deeplabv3plus_resnet50'](num_classes=num_classes, output_stride=16)
# print('net', net)
# summary(net, input_size=(3, 512, 512), batch_size=-1)  # 这里想知道总的参数量，老是出bug


# 把网络放入GPU或CPU -------------------------------------------------------------
net.to(device)
# # summary(net, input_size=(3, 512, 512), batch_size=-1)  # 这里想知道总的参数量，老是出bug
# # 加载或初始化模型参数 -------------------------------------------------------------
model_weight_path = "./pre_weights/resnet50-pre.pth"
#
# # 使用unet网络时，不人工对模型进行初始化，pytorch自动初始化
pretrained_dict = torch.load(model_weight_path)
# pretrained_dict = torch.load(model_weight_path, map_location=torch.device('cpu'))

model_dict = net.state_dict()  # pytorch自动初始化
# 打印参数和大小
# for name, p in net.named_parameters():
#     # print(name)
#     # print(p.requires_grad)
#     print(name, '      ', p.size())
#     print(...)
# for p in net.parameters():
#     print(p)
#     print(...)
# 打印参数的总数
# 方法1
# total = sum([param.nelement() for param in net.parameters()])
# print("Number of parameter: %.2fM" % (total / 1e6))
# # 方法2
# input = torch.randn(16, 3, 512, 512)
# flops, params = profile(net, inputs=(input,))
# print(flops)
# print(params)


# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # # 更新现有的model_dict
model_dict.update(pretrained_dict)
# # # 加载我们真正需要的state_dict
net.load_state_dict(model_dict)


# # 固定resnet网络全部权值参数，unet的参数需要学习
# for param in net.parameters():
#     param.requires_grad = False
# # 解放unet的参数
# for param in net.up4to3.parameters():
#     param.requires_grad = True
# 保存路径 -------------------------------------------------------------
# save_path = './34weight/resNet34_yaozhui.pth'
# save_path = './101weight/resNet101.pth'
save_path = 'weights/seg/'+injury_location+'/deeplab_resnet50_0.0001_epoch100'
# save_path = 'weights/seg/'+injury_location+'/unet_0.0003_epoch100'
# save_path = './unet_weight/unet1.pth'

# 优化器及优化策略 -------------------------------------------------------------
# optimizer = optim.Adam(net.parameters())
# opt_Momentum = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.8)
# opt_RMSprop = torch.optim.RMSprop(net.parameters(), lr=LR, alpha=0.9)
# opt_Adam = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=8, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# optimizer = torch.optim.SGD(params=[
#         {'params': net.backbone.parameters(), 'lr': 0.1*lr},
#         {'params': net.classifier.parameters(), 'lr': lr},
#     ], lr=lr, momentum=0.9, weight_decay=0.5)
# lr_policy_all = ['poly', 'step']
# lr_policy = lr_policy_all[1]
# if lr_policy == 'poly':
#     scheduler = PolyLR(optimizer, max_iters=9, power=0.9)
# elif lr_policy == 'step':
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
train_loss = []
train_acc = []
val_loss = []
val_acc = []

for epoch in range(epochs):
    net.train()
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)
    epoch_loss = 0
    step = 0
    since = time.time()
    # d_len_train = 0
    acc_epoch_train = []
    acc_epoch_val = []
    metrics = defaultdict(float)
    epoch_samples = 0
    for x, y in train_loader:
        # d_len_train += 1

        # metrics = defaultdict(float)

        inputs = x.to(device)
        labels = y.to(device)
        optimizer.zero_grad()
        # forward
        output = net(inputs)
        # print('outputs', outputs.shape)
        output = nn.Softmax2d()(output)
        # m = nn.Softmax(dim=1)
        # output = m(outputs)
        loss = calcu_loss(output, labels, metrics)
        # 作训练的损失图
        # if d_len % 4 == 0:
        #     niter = epoch * train_num + d_l
        #     writer.add_scalar('Train/Loss', loss, niter)
        # 打印各类的dice，总觉得这里的sigmoid要换成argmax
        # output = torch.sigmoid(outputs)
        # # output = output.cpu().detach()
        loss.backward()
        optimizer.step()

        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        acc = calcu_accuracy(output, labels)  # 需要四个通道
        acc_epoch_train.append(acc[1])
        epoch_samples += inputs.size(0)

    # 计算每一个epoch的平均准确率,并保存在train_acc中
    acc_epoch = np.mean(np.array(acc_epoch_train))
    writer.add_scalar('Train/acc', acc_epoch, epoch + 1)
    train_acc.append(acc_epoch)

    # 计算每一个epoch的平均loss,并保存在train_loss中
    # a0 = len(train_loader)
    print_metrics(metrics, epoch_samples, 'train')
    epoch_loss = metrics['loss'] / epoch_samples
    # 作训练的损失图，每一个epoch记录一个点
    writer.add_scalar('Train/Loss', epoch_loss, epoch + 1)
    train_loss.append(epoch_loss)
    print("epoch %d loss:%0.3f" % (epoch, epoch_loss))

    # validate
    net.eval()
    metrics = defaultdict(float)
    epoch_samples = 0
    dices = 0
    # tumor_dices = 0
    # bladder_dices = 0
    # d_len_val = 0
    # acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_images, val_labels in validate_loader:
            # d_len_val += 1
            # 解决数据增强之后输入变成一个5D-tensor的问题，压缩前两维为一维
            # bs_x, ncrops_x, c_x, h, w = x.size()
            # x = x.view(-1, c_x, h, w)
            inputs = val_images.to(device)
            # bs_y, ncrops_y, c_y, h, w = y.size()
            # y = y.view(-1, c_y, h, w)
            labels = val_labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            output = nn.Softmax2d()(outputs)
            loss = calcu_loss(output, labels, metrics)

            # output = torch.sigmoid(outputs)
            # # output = output.cpu().detach()
            output[output <= 0.5] = 0
            output[output > 0.5] = 1
            acc = calcu_accuracy(output, labels)  # 需要四个通道
            # 监测平均准确率是否连续多个epoch没有提升，没有的话改变学习率
            acc_epoch_val.append(acc[1])
            epoch_samples += inputs.size(0)

            # 选出损失最小的权重保存
        # 计算每一个epoch的平均准确率,并保存在val_acc中
        acc_epoch = np.mean(np.array(acc_epoch_val))
        writer.add_scalar('Val/acc', acc_epoch, epoch + 1)
        val_acc.append(acc_epoch)

        # 计算每一个epoch的平均loss,并保存在val_loss中
        a = len(validate_loader)
        print_metrics(metrics, epoch_samples, 'val')
        epoch_loss = metrics['loss'] / epoch_samples
        # 作训练的损失图，每一个epoch记录一个点
        writer.add_scalar('Val/Loss', epoch_loss, epoch + 1)
        val_loss.append(epoch_loss)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(net.state_dict(), save_path)

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val loss: {:4f}'.format(best_loss))
print('Finished Training')
# 保存train_loss,train_acc, val_loss,val_acc到Excel表格中
# train
train_loss = np.array(train_loss)
train_loss_excel = pd.DataFrame(train_loss)
writer = pd.ExcelWriter('./loss&acc/'+injury_location+'/loss/train_loss.xlsx')
train_loss_excel.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()
writer.close()

train_acc = np.array(train_acc)
train_acc_excel = pd.DataFrame(train_acc)
writer = pd.ExcelWriter('./loss&acc/'+injury_location+'/acc/train_acc.xlsx')
train_acc_excel.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()
writer.close()
# val
val_loss = np.array(val_loss)
val_loss_excel = pd.DataFrame(val_loss)
writer = pd.ExcelWriter('./loss&acc/'+injury_location+'/loss/val_loss.xlsx')
val_loss_excel.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()
writer.close()

val_acc = np.array(val_acc)
val_acc_excel = pd.DataFrame(val_acc)
writer = pd.ExcelWriter('./loss&acc/'+injury_location+'/acc/val_acc.xlsx')
val_acc_excel.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()
writer.close()

# watch -n 10 nvidia-smi
# tensorboard --logdir=./log/seg_all/log_resnet101_0.1bce_0.0003_epoch100
# tensorboard --logdir=./log/ebu/log_resnet50_0.1bce_0.0003_epoch100
# tensorboard --logdir=./log/log_0.1bce_unet
# log_resnet101_0.1bce_0.0003_epoch100



