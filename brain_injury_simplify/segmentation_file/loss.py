import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from segmentation_file.calc_loss import diceCoeff, diceCoeff_tversky


class SoftDiceLoss(_Loss):
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []
        # print(y_pred.shape)
        # print(y_true.shape)
        class_weight = [1.42621741, 42.30525223]
        class0_w = class_weight[0]/sum(class_weight)
        class1_w = 1 - class0_w
        for i in range(0, self.num_classes):
            # 计算单个通道每张图片的平均重合度
            if i == 0:
                class_dice.append(class0_w*diceCoeff(y_pred[:, i:i + 1, :, :],
                                                     y_true[:, i:i + 1, :, :], activation=self.activation))
            else:
                class_dice.append(class1_w * diceCoeff(y_pred[:, i:i + 1, :, :],
                                                       y_true[:, i:i + 1, :, :], activation=self.activation))
        # 计算所有通道每张图片的平均重合度，再平均得每个通道每张图片的重合度
        # mean_dice = sum(class_dice) / len(class_dice)
        mean_dice = sum(class_dice)
        return 1 - mean_dice


# tversky_Loss
class TverskyLoss(_Loss):
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation=None, reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []
        # print(y_pred.shape)
        # print(y_true.shape)
        for i in range(0, self.num_classes):
            class_dice.append(diceCoeff_tversky(y_pred[:, i:i + 1, :, :], y_true[:, i:i + 1, :, :],
                                                activation=self.activation))

        # 计算所有通道每张图片的平均重合度，再平均得每个通道每张图片的重合度
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def calcu_loss(pred, target, metrics, bce_weight=0.1):
    # # # bce
    # class_weight = [1.42621741, 42.30525223]
    # class_weight = torch.tensor(np.array(class_weight))
    # bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=class_weight)
    bce = F.binary_cross_entropy_with_logits(pred, target)
    # # dice
    criterion_dice = SoftDiceLoss(2, activation=None)
    dice = criterion_dice(pred, target)
    # loss = dice
    # focal bunengyong?
    # criterion_focal = FocalLoss(ignore_index=255, size_average=True)
    # focal = criterion_focal(pred, target)
    # loss = focal
    # focaltversky_Loss
    # criterion_tversky = TverskyLoss(2, activation=None)
    # tversky = criterion_tversky(pred, target)
    # focaltversky_loss = torch.pow(tversky, 0.75)
    # weight
    loss = bce * bce_weight + dice * (1 - bce_weight)
    # loss = focaltversky_loss
    # target.size(0)的大小就是每个batch_size的样本个数，累积每张图片的平均损失得到所有图片的损失
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    # metrics['focaltversky_loss'] += focaltversky_loss.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    # metrics['focal_loss'] += focal.data.cpu().numpy() * target.size(0)
    # metrics['bce'] += bce.data.cpu().numpy()
    # metrics['dice'] += dice.data.cpu().numpy()
    # # metrics['focaltversky_loss'] += focaltversky_loss.data.cpu().numpy() * target.size(0)
    # metrics['loss'] += loss.data.cpu().numpy()
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))  # 计算每项加权的平均loss
    print("{}: {}".format(phase, ", ".join(outputs)))