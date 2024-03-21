import os

import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
class AWP(nn.Module):
    def __init__(self, in_channels):
        super(AWP, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Calculate attention weights
        b, c, h, w = x.size()
        y = self.conv(x)
        y = self.relu(y)
        y = y.view(b, c, -1)
        y = torch.transpose(y, 1, 2)
        y = nn.functional.softmax(y, dim=1)
        y = y.view(b, c, h, w)
        # Apply attention weights
        return x * y
class ECANet(nn.Module):
    def __init__(self, in_channels, r=8):
        super(ECANet, self).__init__()
        self.awp = AWP(in_channels)#原本是GAP
        #定义全局平均池化层
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channels, in_channels // r, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Conv2d(in_channels // r, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Calculate attention weights
        y = self.awp(x)
        # y = self.gap(x)
        y = self.conv(y)
        y = self.relu(y)
        y = self.fc(y)
        y = self.sigmoid(y)
        # Apply attention weights
        return x * y

class MyNetwork(nn.Module):
    def __init__(self, num_classes=5):
        super(MyNetwork, self).__init__()
        # 参数共享层p1---VGG16 ----X1-空间/x2-时间
        self.p1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=1),  # 224  1
                                nn.ReLU(inplace=True),
                                nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1), #3
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2),  # 112    5

                                nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=1),  # 112X112X32  6
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1),#8
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2),  # 56#10

                                nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1),  # 56X56X64  11
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),#13
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2),  # 28   15
                                )
        self.p2_1 = nn.AdaptiveAvgPool2d((7, 7))  # 7X7X64--空间
        # GAP
        self.p2_2 = nn.AdaptiveMaxPool2d((7, 7))  # 7X7X64--时间
        # 参数共享层p3
        self.p3 = nn.Sequential(nn.Linear(7 * 7 * 64, 256),
                                nn.ReLU(inplace=True),
                                nn.Dropout()
                                )

        self.p4 = ECANet(512)
        self.p5 = nn.Sequential(nn.Linear(512, 64),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Linear(64, num_classes)
                                )
    def forward(self, x1, x2):
        # 参数共享层p1
        x1_conv = []
        x2_conv = []
        for layer in self.p1:
            x1 = layer(x1)
            x1_conv.append(x1.clone())  # 存储特征图用于可视化
            x2 = layer(x2)
            x2_conv.append(x2.clone())  # 存储特征图用于可视化

        # 接下来的特征提取
        x1 = self.p2_1(x1)
        x1_pool = x1.clone()  # 存储特征图用于可视化
        x1 = torch.flatten(x1, 1)
        x1 = self.p3(x1)

        x2 = self.p2_2(x2)
        # print(x2.shape)#torch.Size([1, 64, 7, 7])
        x2_pool = x2.clone()  # 存储特征图用于可视化
        x2 = torch.flatten(x2, 1)
        x2 = self.p3(x2)

        x = torch.cat((x1, x2), dim=1)  # 拼接
        x = x.view(x.size(0), x.size(1), 1, 1)
        # print(x.shape)#torch.Size([1, 512, 1, 1])
        x_cat = x.clone()  # 存储特征图用于可视化
        x = self.p4(x)

        # 提取注意力机制后的特征图
        attention_features = x.clone()

        x = x.view(x.size(0), x.size(1))

        # ...（接下来的分类器）
        x = self.p5(x)

        return  x,attention_features,x_cat,x1_pool,x2_pool,x1_conv, x2_conv  # 返回特征图用于可视化

import torch
from torchvision import transforms
from PIL import Image
# 定义预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载图像并进行预处理
image_path1 = "D:/code/two-stream/datas/class4/3997.jpg"
image_path2 = "D:/code/two-stream/Frames/class4/diff_3998.jpg.jpg"
image1 = Image.open(image_path1).convert("L")  # 转为灰度图
image2 = Image.open(image_path2).convert("L")
input1 = transform(image1).unsqueeze(0)
input2 = transform(image2).unsqueeze(0)
# 实例化模型
model = MyNetwork()
model.load_state_dict(torch.load('D:/code/two-stream/model_data/best_model_2023-08-20_22-37-01.pth'))
model.eval()

# 定义保存特征图的文件夹路径
save_dir1 = "D:/code/two-stream/Feature map/map822/image1"
os.makedirs(save_dir1, exist_ok=True)  # 确保文件夹存在
save_dir2 = "D:/code/two-stream/Feature map/map822/image2"
os.makedirs(save_dir2, exist_ok=True)  # 确保文件夹存在
save_dir3 = "D:/code/two-stream/Feature map/map822/image3"
os.makedirs(save_dir3, exist_ok=True)  # 确保文件夹存在
save_dir4 = "D:/code/two-stream/Feature map/map822/image4"
os.makedirs(save_dir4, exist_ok=True)  # 确保文件夹存在
save_dir5 = "D:/code/two-stream/Feature map/map822/image5"
os.makedirs(save_dir5, exist_ok=True)  # 确保文件夹存在
save_dir6 = "D:/code/two-stream/Feature map/map822/image6"
os.makedirs(save_dir6, exist_ok=True)  # 确保文件夹存在
# 前向传播，获取输出
with torch.no_grad():
    outputs, attention_features, x_cat,x1_pool,x2_pool,x1_conv, x2_conv = model(input1, input2)

# 保存卷积层特征图
for i, conv_feature in enumerate(x1_conv):
    for j in range(conv_feature.size(1)):
        feature_map = conv_feature[0, j].cpu().detach().numpy()
        # 保存原始尺寸的特征图
        original_feature_map = feature_map.copy()
        file_name = os.path.join(save_dir1, f"conv_layer{i+1}_channel{j+1}.png")
        plt.imsave(file_name, feature_map, cmap='cividis')

        # 使用插值方法调整特征图尺寸并保存
        feature_map_resized = cv2.resize(original_feature_map, (224, 224))
        resized_file_name = os.path.join(save_dir1, f"resized_conv_layer{i+1}_channel{j+1}.png")
        plt.imsave(resized_file_name, feature_map_resized, cmap='cividis')


# 保存卷积层特征图
for i, conv_feature in enumerate(x2_conv):
    for j in range(conv_feature.size(1)):
        feature_map = conv_feature[0, j].cpu().detach().numpy()
        # 保存原始尺寸的特征图
        original_feature_map = feature_map.copy()
        file_name = os.path.join(save_dir2, f"conv_layer{i+1}_channel{j+1}_image2.png")
        plt.imsave(file_name, feature_map, cmap='cividis')

        # 使用插值方法调整特征图尺寸并保存
        feature_map_resized = cv2.resize(original_feature_map, (224, 224))
        resized_file_name = os.path.join(save_dir2, f"resized_conv_layer{i+1}_channel{j+1}_image2.png")
        plt.imsave(resized_file_name, feature_map_resized, cmap='cividis')

# 保存池化层特征图
for i in range(x1_pool.size(1)):
    feature_map = x1_pool[0, i].cpu().detach().numpy()
    # 使用插值方法调整特征图尺寸
    feature_map_resized = cv2.resize(feature_map, (224, 224))
    file_name = os.path.join(save_dir3, f"pool_layer{i+1}.png")
    plt.imsave(file_name, feature_map_resized, cmap='cividis')

for i in range(x2_pool.size(1)):
    feature_map = x2_pool[0, i].cpu().detach().numpy()
    # 使用插值方法调整特征图尺寸
    feature_map_resized = cv2.resize(feature_map, (224, 224))
    file_name = os.path.join(save_dir4, f"pool_layer{i+1}_image2.png")
    plt.imsave(file_name, feature_map_resized, cmap='cividis')


# 保存cat层特征图
for i in range(x_cat.size(1)):
    feature_map = x_cat[0, i].cpu().detach().numpy()
    # 使用插值方法调整特征图尺寸
    feature_map_resized = cv2.resize(feature_map, (224, 224))
    file_name = os.path.join(save_dir5, f"cat_layer{i+1}.png")
    plt.imsave(file_name, feature_map_resized, cmap='cividis')


