## 介绍
这个项目的的功能是：使用resnet-18+ASPP网络分别对额部、颞部和枕部三个部位的颅脑CT进行加速、减速和正常脑损伤类型分类。
### 环境依赖  
详见`./requirements.txt`文件，使用requirements.txt安装依赖：`pip install -r requirements.txt `
### 所使用到的文件结构如下
```angular2
.
├── brain_data
│   ├── cla
│   └── seg
├── classification_file
│   ├── class_indices.json
│   ├── custom_dataset_npy.py
│   └── custom_transforms_mine.py
├── data_preprocess
│   ├── __pycache__
├── loss&acc_seg
│   ├── ebu_plus1215_4bei
│   ├── nie_bmp_json
│   └── zhenbu_plus1215_4bei
├── network_big
│   ├── network_deeplab_seg
│   └── network_resnet_aspp
├── predict_seg.py
├── pre_weights
│   └── resnet50-pre.pth
├── requirements.txt
├── segmentation_file
│   ├── calc_loss.py
│   ├── calcu_accuracy.py
│   ├── custom_transforms_mine.py
│   ├── dataset1.py
│   ├── loss.py
│   ├── one_hot.py
│   └── __pycache__
├── predict_classify_one_person_3ch.py
├── predict_classify_one_person_11ch.py
├── test_classify.py
├── train_classify.py
├── train_seg.py
└── weights
    ├── cla
    └── seg     
```
###  分类特征图合成
分类特征图合成是为了加入人工先验知识，以提高脑损伤分类准确率。  
包含三个部分，首先进行图像预处理；然后将病灶分割出来；最后将分割结果加权赋值到原始图像上得到一张特征图，
再叠加连续三张这样的特征图，最终得到分类特征图。 

1、预处理  
- 将一张大CT切割成N张单个颅脑CT小图像，然后在进行配准去噪等操作，这些操作文件在`./data_preprocess`文件夹下；
  
2、分割模型训练
- 模型训练：执行 `train_seg.py`。三个部位的分割数据在`./brain_data/seg/`文件夹下； 
- 训练过程的损失和准确率数据在`./loss&acc_seg`对应的文件夹下，模型参数在`./weights/seg/`对应的文件夹下；  
- 模型推断：执行 `predict_seg.py`。

3、合成特征图  
由于一些“历史遗留”问题，导致目前有两种合成特征图的方法，进而导致有两种不同的训练和预测文件。
- **合成三通道特征图**：对于每个案例，CT切片有数量不等。比如有`N`张CT，可以合成`N-2`张特征图。合成特征图：执行`./data_preprocess/generate_3ch_npy_from_pic.py`。
- **合成十一通道特征图**：合成十一通道的目的是想尽可能的把每个案例所有的特征信息融合到一个特征图里面，因为判定单个案例的损伤类型
  是依靠全部的切片而不只是其中的某几张。但考虑到案例切片数量的差异较大，经过大致的统计得出每个特征图包含11个切片，
  不足或者多出来的切片以添加空白切片或者删除的形式调整到11的倍数。合成特征图：执行`./data_preprocess/generate_11ch_npy_from_pic.py`

###  分类训练和测试
本项目设计的分类网络是基于resnet-18改进的，在卷积层和全连接层之间，
加入了空间金字塔池化模块ASPP，对于本任务有5%左右的效益提升。  
此外，由于最终的目的是为了对每个案例进行脑损伤类别判定，而不仅仅是每张特征图。
故设计了一个基于经验的判别算法，对分类网络的输出结果再次处理，得到最终每个案例的脑损伤类型。  

1、分类模型训练
- 模型训练：执行 `train_classify.py`。三个部位的分割数据在`./brain_data/cla/`对应的文件夹下； 
- 训练过程的损失、准确率数据以及模型参数在`./weights/cla/`对应的文件夹下；  

2、分类模型预测和验证  
- 模型推断及准确率验证（每张特征图）：执行 `test_classify.py`，预测结果和金标准写入对应的文件夹下；  
- 模型推断及准确率验证（每个案例）：三通道特征图：执行 `predict_classify_one_person_3ch.py`；十一通道特征图：执行 `predict_classify_one_person_11ch.py`。

