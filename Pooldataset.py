import sys
import torch
from tqdm import tqdm
import math
from torch.utils.data import Dataset
from PIL import Image, ImageChops
class TSCNNDataset(Dataset):#TSCNN
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.data[index]
        # 计算三帧差分
        img1 = Image.open(img_path)
        img1 = Image.open(img_path).convert('L')
        #打印图像模式
        # print('img1.mode',img1.mode)
        img2 = Image.open(self.data[index + 1][0]).convert('L')
        img3 = Image.open(self.data[index + 2][0]).convert('L')
        #查看img1/img2/img3是否导入正确，只看前三帧
        # if index < 3:
        #     img1.show()
        #     img2.show()
        #     img3.show()
        diff1 = ImageChops.difference(img1, img2)
        diff2 = ImageChops.difference(img2, img3)
        diff = ImageChops.add(diff1, diff2)#将两帧差分相加
        # 进行二值化处理
        diff = diff.point(lambda x: x * 2)
        diff = diff.point(lambda x: 255 if x > 30 else 0).convert('L')
        if self.transform is not None:
            img1 = self.transform(img1)
            diff = self.transform(diff)
            # print('img1.shape',img1.shape)
        return img1, diff, label

    def __len__(self):
        return len(self.data) - 2



import torch
import sys
from tqdm import tqdm

def train_one_epoch(model, optimizer, train_loader, device, epoch, lr_scheduler):
    model.train()

    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    train_loader = tqdm(train_loader, file=sys.stdout)

    # 新增一个列表来记录学习率变化
    lr_values = []

    for step, (images, diffs, labels) in enumerate(train_loader):
        images, diffs, labels = images.to(device), diffs.to(device), labels.to(device)

        sample_num += images.shape[0]

        pred = model(images, diffs)  # 前向传播
        pred_classes = torch.max(pred, dim=1)[1]  # 取预测结果中概率最大的类别
        accu_num += torch.eq(pred_classes, labels).sum()  # 计算预测正确的样本数

        loss = loss_function(pred, labels)
        loss.backward()  # 反向传播
        accu_loss += loss.detach()

        train_loader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]  # 打印学习率
        )

        if not torch.isfinite(loss):  # 判断loss是否为无穷大
            print('WARNING: non-finite loss, ending training ', loss)  # 如果loss为无穷大，打印警告信息
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        # 记录学习率的变化
        lr_values.append(optimizer.param_groups[0]["lr"])

        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, lr_values


@torch.no_grad()#
def evaluate(model, val_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(val_loader, file=sys.stdout)
    for step, (images, diffs, labels) in enumerate(data_loader):
        images, diffs, labels = images.to(device), diffs.to(device), labels.to(device)
        sample_num += images.shape[0]

        pred = model(images,diffs)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def create_lr_scheduler(optimizer,
                        num_step: int,#每个epoch的step数
                        epochs: int,#总训练轮数
                        warmup=True,#是否使用warmup
                        warmup_epochs=1,#warmup阶段的轮数
                        warmup_factor=1e-3,#warmup阶段的学习率倍率因子
                        end_factor=1e-4):#最后阶段的学习率倍率因子
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)#返回一个学习率调整器




