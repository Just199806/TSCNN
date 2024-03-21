#测试模型
import pandas as pd
import torch
from torchvision import datasets, transforms
from Pooldataset import TSCNNDataset
from model import MyNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    # Define the test transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    data_dir = 'D:/code/two-stream/data/'
    classes = ['class1', 'class2', 'class3', 'class4', 'class5']#原始数据集
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

    from torch.utils import data
    test_dataset =TSCNNDataset(test_data, transform=test_transform)

    batch_size = 32
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = MyNetwork(num_classes=5).to(device)#TSCNN
    # load model weights
    model_weight_path = "D:/code/two-stream/model_data/best_model_2023-08-20_22-37-01.pth"#TSCNN
    model.load_state_dict(torch.load(model_weight_path, map_location=device))#加载模型参数
    model.eval()
    import time
    # Test the model
    y_true = []
    y_pred = []
    start_time = time.time()  # start timing
    with torch.no_grad():
        for image, diff, labels in test_loader:
            image, diff, labels = image.to(device), diff.to(device), labels.to(device)
            outputs = model(image, diff)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    end_time = time.time()  # end timing
    test_time = end_time - start_time  # calculate testing time
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Convert the matrix to a DataFrame with class names as row and column labels
    conf_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    print(conf_df)
    # Save the DataFrame as a CSV file,保存至model_data文件夹下
    conf_df.to_csv('metrics_out/confusion_matrix.csv')
    print('Accuracy: %.3f' % accuracy)
    print('Precision: %.3f' % precision)
    print('Recall: %.3f' % recall)
    print('F1 score: %.3f' % f1)
    print('Testing time: %.3f seconds' % test_time)  # print testing time
    # 保存Accuracy, Precision, Recall, F1 score至TXT文件，在model_data文件夹下
    with open('metrics_out/result.txt', 'w') as f:
        f.write('Accuracy: %.3f\n' % accuracy)
        f.write('Precision: %.3f\n' % precision)
        f.write('Recall: %.3f\n' % recall)
        f.write('F1 score: %.3f\n' % f1)
        f.write('Testing time: %.3f seconds ' % test_time)
    # 分别保存真实标签和预测标签
    with open('metrics_out/y_true.txt', 'w') as f: # 保存真实标签
        for i in y_true:
            f.write(str(i) + '\n')
    with open('metrics_out/y_pred.txt', 'w') as f: # 保存预测标签
        for i in y_pred:
            f.write(str(i) + '\n')
    from sklearn.metrics import classification_report
    # 获取分类报告
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    # 将分类报告保存为 CSV 文件，在 model_data 文件夹下
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('metrics_out/classification_report.csv')
    # 保存分类报告中每一类的准确率、召回率和 F1 值至 TXT 文件，在 model_data 文件夹下
    import numpy as np
    with open('metrics_out/classification_report.txt', 'w') as f:
        for cls in classes:
            f.write('Class: %s\n' % cls)
            f.write('Precision: %.3f\n' % report[cls]['precision'])
            f.write('Recall: %.3f\n' % report[cls]['recall'])
            f.write('F1 score: %.3f\n\n' % report[cls]['f1-score'])
    # 计算标准偏差
    std_deviation = np.std(y_pred)
    print('Standard Deviation:', std_deviation)

if __name__ == '__main__':
    main()

