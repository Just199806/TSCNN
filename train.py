from matplotlib import pyplot as plt
from torch import optim, nn
import random
import numpy as np
from Pooldataset import train_one_epoch, evaluate ,TSCNNDataset,create_lr_scheduler#TSCN
from torch.utils import data
from torchvision.transforms import transforms
import torch
from model import MyNetwork
random_seed = 3407
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
def main():
    #定义数据预处理方式
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    data_dir = 'D:/code/two-stream/data/'
    classes = ['class1', 'class2', 'class3','class4','class5']

    train_data = []
    val_data = []
    test_data = []
    def sort_by_number(file_path):
        file_name = os.path.basename(file_path)
        file_number = int(file_name.split('.')[0])
        return file_number
    import os
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = os.listdir(cls_dir)
        files = [os.path.join(cls_dir, f) for f in files]
        files = sorted(files, key=sort_by_number)  # 按照文件名数字顺序排序
        labels = [classes.index(cls)] * len(files)
        num_data = len(files)
        train_data += list(zip(files[:int(0.7 * num_data)], labels[:int(0.7 * num_data)]))
        val_data += list(zip(files[int(0.7 * num_data):int(0.9 * num_data)], labels[int(0.7 * num_data):int(0.9 * num_data)]))
        test_data += list(zip(files[int(0.9 * num_data):], labels[int(0.9 * num_data):]))
    # 创建数据集
    train_dataset = TSCNNDataset(train_data, transform=train_transform)#TSCNN
    val_dataset = TSCNNDataset(val_data, transform=val_transform)#
    # 定义超参数
    batch_size = 32
    lr = 0.001
    num_epochs = 100
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 判断是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义模型
    model = MyNetwork(num_classes=5).to(device)
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.0008)
    #定义学习率衰减策略
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), num_epochs,
                                       warmup=True, warmup_epochs=1)#
    save_every = 1  # 每隔多少轮保存一次模型
    if not os.path.exists('model_data'):
        os.makedirs('model_data')
    # 判断是否存在metrics_output文件夹，不存在则创建
    if not os.path.exists('metrics_out'):  #
        os.mkdir('metrics_out')
    from datetime import datetime
    # 文件名中添加当前时间日期信息
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 每个 epoch 记录的 loss
    train_losses = []
    val_losses = []
    best_val_acc = 0.0  # 保存最佳验证准确率
    # 定义日志文件路径
    import datetime
    logs_folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_path = os.path.join('model_data', 'logs', logs_folder)#
    os.makedirs(logs_path, exist_ok=True)
    # 使用当前时间戳作为文件名的一部分
    train_loss_path = os.path.join(logs_path, f'train_loss_{now}.txt')
    val_loss_path = os.path.join(logs_path, f'val_loss_{now}.txt')

    from collections import deque
    num_trials = 10  # 如果连续 num_trials 轮没有提升，则停止训练
    val_acc_history = deque(maxlen=num_trials)  # 保存最近 num_trials 轮的验证集准确率
    import time
    start_time = time.time()
    # 定义用于记录学习率的变化
    learning_rates = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch, lr_scheduler)
        # train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_loss, train_acc,lr_epoch = train_one_epoch(model, optimizer, train_loader, device, epoch, lr_scheduler)

        val_loss, val_acc = evaluate(model, val_loader, device, epoch)#TSCNN
        learning_rates.extend(lr_epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        #记录学习率
        with open('metrics_out/8.8_ATSCNN_train_acc.txt', 'a') as f:
            f.write(f'{train_acc:.4f}\n')

        with open('metrics_out/8.8_ATSCNN_val_acc.txt', 'a') as f:
            f.write(f'{val_acc:.4f}\n')

        with open(train_loss_path, 'a') as f:
            f.write(f'{train_loss:.4f}\n')

        with open(val_loss_path, 'a') as f:
            f.write(f'{val_loss:.4f}\n')

        if (epoch + 1) % save_every == 0:
            model_path = os.path.join('model_data',
                                      f'model_{epoch + 1}_train_loss_{train_loss:.4f}_val_loss_{val_loss:.4f}_{now}.pth')
            torch.save(model.state_dict(), model_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join('model_data', f'best_model_{now}.pth')
            torch.save(model.state_dict(), best_model_path)
            print('Saved best model!')

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch}: {epoch_time:.2f} seconds")
        # 判断是否早停，如果早停则加载最佳模型开始测试
        val_acc_history.append(val_acc)
        if len(val_acc_history) == num_trials and all(acc <= val_acc_history[0] for acc in val_acc_history):
            print(f'Validation accuracy did not improve for {num_trials} trials. Stopping early.')
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # # 绘制学习率曲线
    # plt.plot(range(1, len(learning_rates) + 1), learning_rates)
    # plt.xlabel('Training Steps')
    # plt.ylabel('Learning Rate')
    # plt.title('Warm-up + Cosine Annealing Learning Rate Schedule')
    # plt.grid(True)  # 绘制背景网格
    # plt.savefig('8.1.1_learning_rate_curve.jpg' , dpi=300)
    # plt.show()
    # 打印最佳准确率和所对应的epoch
    # print(f"Best validation accuracy: {best_val_acc:.4f}, corresponding epoch: {val_acc_history.index(max(val_acc_history))}")

if __name__ == '__main__':
    main()


