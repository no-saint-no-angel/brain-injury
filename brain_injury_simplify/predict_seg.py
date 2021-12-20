import os
import cv2
import torch
import numpy as np
import network_big.network_deeplab_seg as network

from torchvision import transforms
from segmentation_file.custom_transforms_mine import *
from segmentation_file.dataset1 import LiverDataset
from segmentation_file.one_hot import onehot_to_mask, mask_to_onehot

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


palette2 = [[0, 0, 0], [255, 255, 255]]
img_normMean = [0.207349,0.207349,0.207349]
img_normStd = [0.283727,0.283727,0.283727]


# 限制自适应直方图均衡化，OpenCV
def clahe_cv(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    output_cv = cv2.merge([b, g, r])
    return output_cv


class DualCompose_img:  # 只有图片没有mask

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# 这个是没有图片增加的数据增强，改写的函数
image_only_transform = DualCompose_img([
    # RandomGaussianBlur_img(),
    ToTensor_img(),
    Normalize_img()
])
# deeplab -----------------------------------------------------------------------------
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

model = model_map['deeplabv3plus_resnet50'](num_classes=2, output_stride=16)
# load model weights

# model_weight_path = "./weights/seg/deeplab_resnet50_0.0003"
model_weight_path = "./weights/seg/deeplab_resnet50_0.0003_epoch100_only_bmptobmp_json"
model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
model.eval()

# dir_test = 'brain_data/seg/nie/seg/images/'
dir_test = 'brain_data/seg/nie/pred_for_paper_1116/jiasu'
imgList = os.listdir(dir_test)
# 320*320的图片没有年龄标签，所以采用".“来分隔
# imgList.sort(key=lambda x: int(x.split('.')[0]))  # 读取句号前面的数字，然后再排序
# print(imgList)

with torch.no_grad():
    for count in range(0, len(imgList)):
        print("\nprocessing the " + str(count + 1) + "th" + " picture.......")
        # inputs = inputs.to(device)
        im_name = imgList[count]

        im_path = os.path.join(dir_test, im_name)
        input = cv2.imread(im_path)  # 原始图片用于计算信号强度
        # print('input: ', input.shape[0])

        # 对原始图图片进行限制自适应直方图均衡化
        out_cv = clahe_cv(input)
        # out_cv = input
        input_img = image_only_transform(out_cv)
        input_img = torch.unsqueeze(input_img, 0)
        pred_img = model(input_img)  # 这里加了一个module
        output = torch.nn.Softmax2d()(pred_img)  # 输入是4d，指定对第二维进行取概率

        output[output > 0.3] = 1  # 这里的意思就是说，预测图片经过Softmax2d()之后，每个位置属于哪个通道有了一个概率，
        # 把概率超过阈值的令做1
        output[output <= 0.3] = 0

        pred = torch.squeeze(output).numpy()  # 压缩batchsize维度，4变3, numpy.ndarray
        pred = pred.transpose([1, 2, 0])  # numpy.ndarray
        pred = onehot_to_mask(pred, palette2)  # 这里的输入格式是numpy.ndarray
        pred = np.squeeze(pred)  # w*h

        # input_img = torch.squeeze(out_cv).numpy()
        # input_img = input_img.transpose([1, 2, 0])  # w*h*c
        # 保存pred和gt图片
        # cv2.imwrite('./ct_data/predict_images/pred.BMP', pred)
        # cv2.imwrite('./ct_data/predict_images/gt.BMP', gt)

        # 将预测（推断）的图片保存在文件夹下
        # 保存pred和gt图片
        cv2.imwrite('brain_data/seg/nie/pred_for_paper_1116/jiasu/' + str(im_name) + '_seg'+'.BMP', pred)

        # cv2.imshow("inputs", out_cv)
        # cv2.imshow("predict", pred)
        # # show_res = np.uint8(np.hstack([pred, gt]))
        # # cv2.imshow("predict(right)&masks(left)", show_res)  # 输出图片
        # cv2.waitKey(0)



