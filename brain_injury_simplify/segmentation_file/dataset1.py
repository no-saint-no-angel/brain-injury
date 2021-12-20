from random import random
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
from segmentation_file.one_hot import mask_to_onehot
import torch
from torchvision import transforms, datasets
import cv2

palette = [[0, 0, 0], [1, 1, 1]]

# img_normMean = [0.18133941, 0.18133941, 0.18133941]
# img_normStd = [0.17976207, 0.17976207, 0.17976207]
# mask_normMean = [0.0029872623, 0.0029872623, 0.0029872623]
# mask_normStd = [0.01015095, 0.01015095, 0.01015095]

# palette = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]


def make_dataset(root):
    imgs = []
    # n = len(os.listdir(root))//2
    root_img = root + "/images"
    root_mask = root + "/masks"
    imgList_img = os.listdir(root_img)
    n = len(imgList_img)
    # imgList_img.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    # print(imgList_img)
    imgList_mask = os.listdir(root_mask)
    # imgList_mask.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    for i in range(n):
        name_img = imgList_img[i]
        name_mask = imgList_mask[i]
        img = os.path.join(root_img, name_img)
        mask = os.path.join(root_mask, name_mask)
        imgs.append((img, mask))
        # print(images)
    return imgs


class LiverDataset(Dataset):
    def __init__(self, root, transform_torch=None, transform_mine=None, image_and_mask_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.palette = palette
        self.transform_torch = transform_torch
        self.transform_mine = transform_mine
        self.image_and_mask_transform = image_and_mask_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x_temp = Image.open(x_path)
        img_y_temp = Image.open(y_path)
        if self.transform_torch is not None:
            img_x_temp, img_y_temp = self.transform_torch(img_x_temp, img_y_temp)
        if self.transform_mine is not None:
            img_x_temp, img_y_temp = self.transform_mine(img_x_temp, img_y_temp)
        img_x_temp = np.array(img_x_temp)
        img_y_temp = np.array(img_y_temp)
        # print(img_x_temp.shape)
        # print(img_y_temp.shape)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
        # img_y_temp = np.expand_dims(img_y_temp, axis=2)  # (H, W, 1)
        img_y_temp = mask_to_onehot(img_y_temp, self.palette)  # (H, W, 3)
        # print(img_y_temp.shape)
        if self.image_and_mask_transform is not None:
            img_x_temp, img_y_temp = self.image_and_mask_transform(img_x_temp, img_y_temp)
            # print(img_x_temp.shape)
            # print(img_y_temp.shape)
        return img_x_temp, img_y_temp

    def __len__(self):
        return len(self.imgs)
