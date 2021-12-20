

import torch
import torch.nn as nn
import numpy as np

# 这两种方法的结果是一样的，都是dice指数，
# Jacarrd index也就是IOU,即人们常说的交并比，Jacarrd index =  tp / (tp + fp + fn)，和dice不一样


def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")
    # print(pred.shape)
    # print(gt.shape)
    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)  # 这里的并集是直接相加的，那应该是dice = (2 * (pred ∩ gt)) / (pred + gt)
    loss = (2 * intersection + eps) / (unionset + eps)

    return loss.sum() / N


#   tversky_loss

def diceCoeff_tversky(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
    alpha = 0.3
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp

    tversky = (tp + eps) / (tp + alpha * fp + (1 - alpha) * fn + eps)
    # intersection = (pred_flat * gt_flat).sum(1)
    # unionset = pred_flat.sum(1) + gt_flat.sum(1)  # 这里的并集是直接相加的，那应该是dice = (2 * (pred ∩ gt)) / (pred + gt)
    # loss = (2 * intersection + eps) / (unionset + eps)

    return tversky.sum() / N


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)   # 图片的数量，就是这个数据块由几张图片压缩的
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    segmentationMetric = ['classPixelAccuracy', 'intersectionOverUnion', 'dice']
    clculate = segmentationMetric[2]  # 选择求取每个类别的精确率，或者iou
    if clculate == 'classPixelAccuracy':
        loss = (tp + eps) / (tp + fp + eps)
    elif clculate == 'intersectionOverUnion':
        loss = (tp + eps) / (tp + fp + fn + eps)
    else:
        loss = (2*tp + eps) / (2*tp + fp + fn + eps)
    return loss.sum() / N
# 这里的loss是每张图片所有通道加起来的loss，（数据块除以图片数，得每张图片平均loss）



