import torch
import torch.nn as nn
import torch.nn.functional as F

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
        #参数共享层p1---VGG16 ----X1-空间/x2-时间
        self.p1= nn.Sequential( nn.Conv2d(1, 16, kernel_size=5, padding=2,stride=1), # 224
                                nn.ReLU(inplace=True),
                                nn.Conv2d(16, 16, kernel_size=5, padding=2,stride=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2),#112


                                nn.Conv2d(16, 32, kernel_size=5, padding=2,stride=1),  # 112X112X32
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=5, padding=2,stride=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2), # 56


                                nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1),  # 56X56X64
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2),  # 28

        )
        self.p2_1 = nn.AdaptiveAvgPool2d((7, 7))    # 7X7X64--空间
        #GAP
        self.p2_2 = nn.AdaptiveMaxPool2d((7, 7))  # 7X7X64--时间
        #参数共享层p3
        self.p3 = nn.Sequential(nn.Linear(7*7*64, 256),
                                nn.ReLU(inplace=True),
                                nn.Dropout()
                                )
        # self.p3_2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1),  # 添加一个卷积层用于 Conv Fusion
        #                           nn.ReLU(inplace=True),
        #                           nn.Dropout())#128x7x7
        self.p4 = ECANet(512)
        self.p5 = nn.Sequential(nn.Linear(512 , 64),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Linear(64, num_classes)
                                )
    def forward(self, x1, x2):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        x1 = self.p1(x1)
        # print(x1.shape)
        x1 = self.p2_1(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.p3(x1)
        # print(x1.shape)#32,256
        x2 = self.p1(x2)
        x2 = self.p2_2(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.p3(x2)
       # print(x2.shape)#32,256
        x = torch.cat((x1, x2), dim=1)#拼接
        # print(x.shape)#32,512
        # x = x.unsqueeze(2).unsqueeze(3)#(batch_size,512,1,1)卷积
        # x = x1 + x2 #Sum fusion
        # x = torch.max(x1, x2)  # 最大融合
        # x= self.p3_2(x_cat)#卷积融合
        # 扩展维度适应ECANet
        x = x.view(x.size(0), x.size(1), 1, 1)#(batch_size,512,1,1)
        x = self.p4(x)#注意力机制
        # print(x.shape)
        #展平x
        x = x.view(x.size(0), x.size(1))#(batch_size,512)
        # x = torch.flatten(x, 1)#卷积
        x = self.p5(x)
        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time_ms = start.elapsed_time(end)
        # print(f"Implementation time: {elapsed_time_ms:.2f} ms")
        return x







