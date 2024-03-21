import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img

# def main():
#     model = MyNetwork()
#     model.load_state_dict(torch.load('D:/code/two-stream/model_data/best_model_2023-04-02_19-04-35.pth'))
#     model.eval()
#     target_layers = [model.p1[-1]]
#
#     data_transform = transforms.Compose([transforms.ToTensor(),
#                                          transforms.Normalize((0.5,), (0.5,))])
#     # load image
#     img_path = "D:/code/two-stream/classification/Sound weld/5999.jpg"
#     assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
#     img = Image.open(img_path).convert('L')
#     img = np.array(img, dtype=np.uint8)
#
#     # [C, H, W]
#     img_tensor = data_transform(img)
#     # expand batch dimension
#     # [C, H, W] -> [N, C, H, W]
#     input_tensor = torch.unsqueeze(img_tensor, dim=0)
#
#     cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
#     target_category = 'Sound weld'  # tabby, tabby cat
#     # target_category = 254  # pug, pug-dog
#
#     grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
#
#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
#                                       grayscale_cam,
#                                       use_rgb=True)
#     plt.imshow(visualization)
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
#



import torch
import torch.nn as nn
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
        # print(y.shape)
        y = self.relu(y)
        y = self.fc(y)
        y = self.sigmoid(y)
        # Apply attention weights
        return x * y
class MyNetwork(nn.Module):
    def __init__(self, num_classes=5):
        super(MyNetwork, self).__init__()
        # 参数共享层p1---VGG16 ----X1-空间/x2-时间
        self.p1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=1),  # 16X224X224
                                nn.ReLU(inplace=True),
                                nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2),  # 16X112X112

                                nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=1),  # 32X112X112
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2),  # 32X56X56

                                nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1),  # 64X56X56
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2),  # 64X28X28
                                )
        self.p2_1 = nn.AdaptiveAvgPool2d((7, 7))  # 7X7X64--空间
        # GAP
        self.p2_2 = nn.AdaptiveMaxPool2d((7, 7))  # 7X7X64--时间
        # 参数共享层p3
        self.p3 = nn.Sequential(nn.Linear(7 * 7 * 64, 256),
                                nn.ReLU(inplace=True),
                                nn.Dropout()
                                )
        # self.p3_2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1),  # 添加一个卷积层用于 Conv Fusion
        #                           nn.ReLU(inplace=True),
        #                           nn.Dropout())#128x7x7
        self.p4 = ECANet(512)
        self.p5 = nn.Sequential(nn.Linear(512, 64),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Linear(64, num_classes)
                                )

    def forward(self, x1, x2):
        x1 = self.p1(x1)
        # print(x1.shape)#[32, 64, 28, 28]
        x1_conv_output = x1  # 获取 x1 经过 p1 之后的卷积层输出
        # print(x1_conv_output.shape)
        x1 = self.p2_1(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.p3(x1)

        x2 = self.p1(x2)
        x2_conv_output = x2  # 获取 x2 经过 p1 之后的卷积层输出
        # print(x2_conv_output.shape)
        x2 = self.p2_2(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.p3(x2)

        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.p4(x)
        x = x.view(x.size(0), x.size(1))
        x = self.p5(x)

        return x, x1_conv_output, x2_conv_output  # 返回 CAM 所需的中间层特征图

    def get_cam(self, features, weights):
        # Weighted sum of features
        cam = torch.sum(features * weights.unsqueeze(2).unsqueeze(3), dim=1)
        # Apply ReLU
        cam = F.relu(cam)
        # Normalize the CAM
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam

import torch
import torch.nn.functional as F
import cv2

# 加载预训练模型
model = MyNetwork(num_classes=5)
model.load_state_dict(torch.load('D:/code/two-stream/model_data/best_model_2023-08-20_22-37-01.pth'))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
# 加载示例图像
image1 = cv2.imread('D:/code/two-stream/datas/class3/2779.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('D:/code/two-stream/Frames/class3/diff_2780.jpg.jpg', cv2.IMREAD_GRAYSCALE)
pil_image1 = Image.fromarray(image1)
pil_image2 = Image.fromarray(image2)
transformed_image1 = transform(pil_image1)
transformed_image2 = transform(pil_image2)
image_tensor1 = torch.unsqueeze(torch.FloatTensor(transformed_image1), 0)
image_tensor2 = torch.unsqueeze(torch.FloatTensor(transformed_image2), 0)
print(image_tensor1.shape)
# 进行前向传播，获取 p1 最后一个卷积层的特征图
with torch.no_grad():
    x, x1_conv_output, x2_conv_output = model(image_tensor1, image_tensor2)

# 获取分类得分最高的类别
predicted_class = torch.argmax(x, dim=1).item()

# 获取 p1 最后一个卷积层的权重
last_conv_layer_weights = model.p1[-3].weight
print(last_conv_layer_weights.shape)

# 计算两个分支的CAM，使用 p1 最后一个卷积层的权重
cam1 = model.get_cam(x1_conv_output, last_conv_layer_weights)
cam2 = model.get_cam(x2_conv_output, last_conv_layer_weights)

# 将CAM缩放到原始图像的尺寸
cam1 = F.interpolate(cam1.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
cam2 = F.interpolate(cam2.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)

# 转换为numpy数组
cam1 = cam1.squeeze().cpu().numpy()
cam2 = cam2.squeeze().cpu().numpy()

# 将CAM叠加到原始图像上
heatmap1 = cv2.applyColorMap(np.uint8(255 * cam1), cv2.COLORMAP_JET)
heatmap2 = cv2.applyColorMap(np.uint8(255 * cam2), cv2.COLORMAP_JET)

result1 = heatmap1 * 0.4 + cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR) * 0.6
result2 = heatmap2 * 0.4 + cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR) * 0.6

# 显示结果
cv2.imshow('CAM for Image 1', result1)
cv2.imshow('CAM for Image 2', result2)
cv2.waitKey(0)
cv2.destroyAllWindows()
