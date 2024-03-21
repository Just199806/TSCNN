import torch.nn as nn
import torch

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
        self.awp = AWP(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels // r, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Conv2d(in_channels // r, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Calculate attention weights
        y = self.awp(x)
        y = self.conv(y)
        y = self.relu(y)
        y = self.fc(y)
        y = self.sigmoid(y)
        # Apply attention weights
        return x * y

class MyNetwork(nn.Module):
    def __init__(self, num_classes=5):
        super(MyNetwork, self).__init__()
        #参数共享层p1
        self.p1= nn.Sequential( nn.Conv2d(1, 16, kernel_size=5, padding=2,stride=1), # 224X224X16/112X112X16
                                nn.ReLU(inplace=True),
                                nn.Conv2d(16, 16, kernel_size=5, padding=2,stride=1), #
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2),

                                nn.Conv2d(16, 32, kernel_size=5, padding=2,stride=1),  # 112X112X32
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=5, padding=2,stride=1),  # 112X112X32/56X56X32
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2), # 56X56X32/28X28X32


                                nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1),  #
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),  # 56X56X64/28X28X644
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2),  # 28X28X64/14X14X64

        )
        #参数共享层p3
        self.p2_1 = nn.AdaptiveAvgPool2d((7, 7))    # 7X7X64--空间
        #GAP
        self.p2_2 = nn.AdaptiveMaxPool2d((7, 7))  # 7X7X64--时间
        #参数共享层p3
        self.p3 = nn.Sequential(nn.Linear(7*7*64, 256),
                                nn.ReLU(inplace=True),
                                nn.Dropout()
                                  )
        self.p4 = ECANet(512)

        self.p5 = nn.Sequential(nn.Linear(512 , 64),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Linear(64, num_classes)
                                )
    #可视化双流网络的中间层特征矩阵
    def forward(self, x1,x2):
        # 第一流 卷积池化
        # outputs1 = []
        # for name, module in self.p1.named_children():
        #     x1 = module(x1)
        #     if name in ["0", "2", "4", "5","7", "9", "10", "12", "14"]:
        #         outputs1.append(x1)
        # #第二流 卷积池化
        # outputs2 = []
        # for name, module in self.p1.named_children():
        #     x2 = module(x2)
        #     if name in ["0", "2", "4", "5","7", "9", "10", "12", "14"]:
        #         outputs2.append(x2)
        #
        # return outputs1, outputs2

        #自适应池化层可视化
        x1_outputs = []
        for name, module in self.p1.named_children():
            x1 = module(x1)
            if name in ["0", "2", "4", "5", "7", "9", "10", "12", "14"]:
                x1_outputs.append(x1)
        x1_p2_1_output = self.p2_1(x1_outputs[-1])

        x2_outputs = []
        for name, module in self.p1.named_children():
            x2 = module(x2)
            if name in ["0", "2", "4", "5", "7", "9", "10", "12", "14"]:
                x2_outputs.append(x2)
        # 获取 (p2_1) 和 (p2_2) 层的输出
        x2_p2_2_output = self.p2_2(x2_outputs[-1])

        return x2_p2_2_output,x1_p2_1_output

        # x1_outputs = []
        # for name, module in self.p1.named_children():
        #     x1 = module(x1)
        #     if name in ["0", "2", "4", "5", "7", "9", "10", "12", "14"]:
        #         x1_outputs.append(x1)
        # x1_p2_1_output = self.p2_1(x1_outputs[-1])

        # 可视化线性层
        # x2_outputs = []
        # for name, module in self.p1.named_children():
        #     x2 = module(x2)
        #     if name in ["0", "2", "4", "5", "7", "9", "10", "12", "14"]:
        #         x2_outputs.append(x2)
        # # 获取 (p2_1) 和 (p2_2) 层的输出
        # x2_p2_2_output = self.p2_2(x2_outputs[-1])
        #
        # # 获取 (p3) 层的输出，并进行可视化
        # x1_p3_output = x1_p2_1_output.view(x1_p2_1_output.size(0), -1)
        # x1_p3_output = self.p3[0](x1_p3_output)
        # x1_p3_output.register_hook(lambda grad: print('Gradient of x1_p3_output:', grad))
        # x1_p3_output = self.p3(x1_p3_output)
        #
        # x2_p3_output = x2_p2_2_output.view(x2_p2_2_output.size(0), -1)
        # x2_p3_output = self.p3[0](x2_p3_output)
        # x2_p3_output.register_hook(lambda grad: print('Gradient of x2_p3_output:', grad))
        # x2_p3_output = self.p3(x2_p3_output)
        #
        # return  x1_p3_output,x2_p3_output

        # x1_outputs = []
        # for name, module in self.p1.named_children():
        #     x1 = module(x1)
        #     if name in ["0", "2", "4", "5", "7", "9", "10", "12", "14"]:
        #         x1_outputs.append(x1)
        # x1_p2_1_output = self.p2_1(x1_outputs[-1])
        # x1_p3_output = x1_p2_1_output.view(x1_p2_1_output.size(0), -1)
        # x1_p3_output = self.p3(x1_p3_output)
        #
        # x2_outputs = []
        # for name, module in self.p1.named_children():
        #     x2 = module(x2)
        #     if name in ["0", "2", "4", "5", "7", "9", "10", "12", "14"]:
        #         x2_outputs.append(x2)
        # # 获取 (p2_1) 和 (p2_2) 层的输出
        # x2_p2_2_output = self.p2_2(x2_outputs[-1])
        # x2_p3_output = x2_p2_2_output.view(x2_p2_2_output.size(0), -1)
        # x2_p3_output = self.p3(x2_p3_output)
        #
        # # Cat 操作
        # x_p3_output = torch.cat((x1_p3_output, x2_p3_output), dim=1)#32,512
        #
        # x_p3_output = x_p3_output.view(x_p3_output.size(0), x_p3_output.size(1), 1, 1)#32,512,1,1
        # # ECANet
        # for name, module in self.p4.named_children():
        #     x_p3_output = module(x_p3_output)
        #     if name in ["conv"]:
        #         x_p3_output = x_p3_output * x_p3_output
        #
        #
        # x_p4_output = self.p4(x_p3_output)
        #
        # x_p4_output = x_p4_output.view(x_p4_output.size(0), x_p4_output.size(1))  # (batch_size,512)
        # # 线性层
        # x_p5_output = self.p5(x_p4_output)
        #
        # return x_p4_output,x_p5_output



        # outputs3 = []
        # for name, module in self.p3.named_children():
        #     x1= module(x1)
        #     print(x1.shape)
        #     if name in ["0"]:
        #         outputs3.append(x1)
        # outputs4 = []
        # for name, module in self.p3.named_children():
        #     x2 = module(x2)
        #     if name in ["0"]:
        #         outputs4.append(x2)
        # #注意力机制
        # outputs5 = []
        # for name, module in self.ecanet.named_children():
        #     x = torch.cat((self.p3(x1), self.p3(x2)), dim=1)
        #     print(x.shape)
        #     x = module(x)
        #     if name in ["fc"]:
        #         outputs5.append(x)

        # return outputs1, outputs2


