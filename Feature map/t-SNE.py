import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model import MyNetwork
from torchvision import transforms
from Pooldataset import TSCNNDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def extract_features_cat(model, test_loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for image,diff, label in test_loader:
            x1, x2= image, diff
            #双流融合
            x1_features = model.p2_1(model.p1(x1))
            x1_features = x1_features.view(x1_features.size(0), -1)
            features_x1 = model.p3(x1_features)

            x2_features = model.p2_2(model.p1(x2))
            x2_features = x2_features.view(x2_features.size(0), -1)
            features_x2 = model.p3(x2_features)

            features.append(torch.cat((features_x1, features_x2), dim=1).detach().cpu().numpy())
            labels.append(label.numpy())
    features = np.concatenate(features)#
    labels = np.concatenate(labels)
    return features, labels

#model.p1
def extract_features_p12(model, test_loader):
    model.eval()
    features_x1_list = []
    features_x2_list = []
    labels = []
    with torch.no_grad():
        for image, diff, label in test_loader:
            x1, x2 = image, diff
            # 提取model.p1的特征
            features_x1 = model.p1(x1)
            features_x2 = model.p1(x2)

            features_x1_list.append(features_x1.detach().cpu().numpy())
            features_x2_list.append(features_x2.detach().cpu().numpy())
            labels.append(label.numpy())

    features_x1 = np.concatenate(features_x1_list)
    features_x2 = np.concatenate(features_x2_list)
    labels = np.concatenate(labels)
    return features_x1, features_x2, labels

def main():
    model = MyNetwork()
    model.load_state_dict(torch.load('D:/code/two-stream/model_data/best_model_2023-08-14_15-09-41(98.7).pth'))
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # data_dir = 'D:/code/two-stream/datas/'
    # classes = ['class1', 'class2', 'class3', 'class4', 'class5']#原始数据集
    # train_data = []
    # val_data = []
    # test_data = []
    # def sort_by_number(file_path):
    #     file_name = os.path.basename(file_path)
    #     file_number = int(file_name.split('.')[0])
    #     return file_number
    # import os
    # for cls in classes:
    #     cls_dir = os.path.join(data_dir, cls)
    #     files = os.listdir(cls_dir)
    #     files = [os.path.join(cls_dir, f) for f in files]
    #     files = sorted(files, key=sort_by_number)  # 按照文件名数字顺序排序
    #     labels = [classes.index(cls)] * len(files)
    #     num_data = len(files)
    #     train_data += list(zip(files[:int(0.7 * num_data)], labels[:int(0.7 * num_data)]))
    #     val_data += list(zip(files[int(0.7 * num_data):int(0.9 * num_data)], labels[int(0.7 * num_data):int(0.9 * num_data)]))
    #     test_data += list(zip(files[int(0.9 * num_data):], labels[int(0.9 * num_data):]))

    # tsne_data
    data_dir = 'D:/code/compare/TSNE_data/'
    classes = ['class1', 'class2', 'class3', 'class4', 'class5']#泛化性数据集
    test_data = []
    import os
    def sort_by_number(file_path):
        file_name = os.path.basename(file_path)
        file_number = int(file_name.split('.')[0])
        return file_number
    # import os
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = os.listdir(cls_dir)
        files = [os.path.join(cls_dir, f) for f in files if
                 f.endswith('.jpg') or f.endswith('.png')]  # 仅保留 .jpg 和 .png 图片文件
        files = sorted(files, key=sort_by_number)  # 按照文件名数字顺序排序
        labels = [classes.index(cls)] * len(files)
        test_data += list(zip(files, labels))
    #
    from torch.utils import data
    test_dataset = TSCNNDataset(test_data, transform=test_transform)
    batch_size = 32
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 双流融合
    features, labels = extract_features_cat(model, test_loader)
    tsne = TSNE(n_components=2, random_state=42,perplexity=20)#perplexity值是30，learning_rate值是200
    features_tsne = tsne.fit_transform(features)

    class_labels_to_names = {
        0: 'Undercut weld',
        1: 'Under penetration',
        2: 'Excessive penetration',
        3: 'Sound weld',
        4: 'Multiple defects'
    }
    # 可视化 features_p2_1
    plt.figure(figsize=(8, 6))
    for class_label in np.unique(labels):
        indices = labels == class_label
        size_factor = 20  # 调整此值以控制散点的大小
        plt.scatter(features_tsne[indices, 0],
                    features_tsne[indices, 1],
                    label=class_labels_to_names[class_label],  # 使用映射获取标签名称
                    s=size_factor,
                    marker='o',  # 设置散点类型为圆圈
                    alpha=0.7)

    plt.legend()
    plt.xlabel('Dimension 1', fontdict={'fontsize': 14}, labelpad=10)
    plt.ylabel('Dimension 2', fontdict={'fontsize': 14})
    plt.savefig('D:/code/Feature map/TSNE/Cat_t-SNE.png', dpi=300, format='png')
    plt.show()

    # # model.p1
    # features_x1, features_x2, labels = extract_features(model, test_loader)
    # features_x1_flat = features_x1.reshape(features_x1.shape[0], -1)  # 展平为二维数组
    # features_x2_flat = features_x2.reshape(features_x2.shape[0], -1)  # 展平为二维数组
    #
    # # 二维映射
    # tsne = TSNE(n_components=2, random_state=42,perplexity=20)#perplexity值是30，learning_rate值是200
    # features_x1_tsne = tsne.fit_transform(features_x1_flat)
    # features_x2_tsne = tsne.fit_transform(features_x2_flat)
    # # 创建颜色映射数组，根据类别设置不同的颜色
    # color_map = ['blue', 'green', 'red', 'purple', 'orange']
    # # 可视化 features_x1
    # plt.figure(figsize=(8, 6))
    # for class_label in np.unique(labels):
    #     indices = labels == class_label
    #     plt.scatter(
    #         features_x1_tsne[indices, 0],
    #         features_x1_tsne[indices, 1],
    #         label=str(class_label),
    #         s=15,  # 设置散点大小为50
    #         marker='o',  # 使用圆圈作为散点类型
    #         c=color_map[class_label],  # 使用颜色映射数组设置不同的颜色
    #         alpha=0.7  # 设置透明度
    #     )
    # plt.legend()
    # plt.xlabel('Dimension 1', fontdict={'fontsize': 18, 'fontname': 'Times New Roman'}, labelpad=10)
    # plt.ylabel('Dimension 2', fontdict={'fontsize': 18, 'fontname': 'Times New Roman'})
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.savefig('D:/code/two-stream/Feature map/t-nse/F_features_x1_t-SNE.png', dpi=300, format='png')
    # plt.show()
    #
    # # 可视化 features_x2
    # plt.figure(figsize=(8, 6))
    # for class_label in np.unique(labels):
    #     indices = labels == class_label
    #     plt.scatter(
    #         features_x2_tsne[indices, 0],
    #         features_x2_tsne[indices, 1],
    #         label=str(class_label),
    #         s=15,  # 设置散点大小为50
    #         marker='o',  # 使用三角形作为散点类型
    #         c=color_map[class_label],  # 使用颜色映射数组设置不同的颜色
    #         alpha=0.7  # 设置透明度
    #     )
    # plt.legend()
    # plt.xlabel('Dimension 1', fontdict={'fontsize': 18, 'fontname': 'Times New Roman'}, labelpad=10)
    # plt.ylabel('Dimension 2', fontdict={'fontsize': 18, 'fontname': 'Times New Roman'})
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.savefig('D:/code/two-stream/Feature map/t-nse/F_features_x2_t-SNE.png', dpi=600, format='png')
    # plt.show()
    # # 三维映射
    # tsne = TSNE(n_components=3, random_state=42, perplexity=20)
    # features_x1_tsne_3d = tsne.fit_transform(features_x1_flat)
    # features_x2_tsne_3d = tsne.fit_transform(features_x2_flat)
    #  color_map = ['blue', 'green', 'red', 'purple', 'orange']
    # # 可视化 features_x1 三维空间
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # for class_label in np.unique(labels):
    #     indices = labels == class_label
    #     size_factor = 20  # 调整此值以控制散点的大小
    #     ax.scatter(
    #         features_x1_tsne_3d[indices, 0],
    #         features_x1_tsne_3d[indices, 1],
    #         features_x1_tsne_3d[indices, 2],
    #         label=str(class_label),
    #         marker='o',  # 使用圆圈作为散点类型
    #         c=color_map[class_label],  # 使用颜色映射数组设置不同的颜色
    #         s=size_factor,  # 设置散点大小
    #         alpha=0.7  # 设置透明度
    #     )
    # ax.legend()
    # ax.set_xlabel('Dimension 1', fontsize=18, fontname='Times New Roman', labelpad=10)
    # ax.set_ylabel('Dimension 2', fontsize=18, fontname='Times New Roman', labelpad=10)
    # ax.set_zlabel('Dimension 3', fontsize=18, fontname='Times New Roman', labelpad=10)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # plt.savefig('D:/code/two-stream/Feature map/t-nse/F_features_x1_t-SNE_3D.png', dpi=300, format='png')
    # plt.show()
    # # 可视化 features_x2
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # for class_label in np.unique(labels):
    #     indices = labels == class_label
    #     size_factor = 20  # 调整此值以控制散点的大小
    #     ax.scatter(
    #         features_x2_tsne_3d[indices, 0],
    #         features_x2_tsne_3d[indices, 1],
    #         features_x2_tsne_3d[indices, 2],
    #         label=str(class_label),
    #         marker='o',  # 使用圆圈作为散点类型
    #         c=color_map[class_label],  # 使用颜色映射数组设置不同的颜色
    #         s=size_factor,  # 设置散点大小
    #         alpha=0.7  # 设置透明度
    #     )
    # ax.legend()
    # ax.set_xlabel('Dimension 1', fontsize=18, fontname='Times New Roman', labelpad=10)
    # ax.set_ylabel('Dimension 2', fontsize=18, fontname='Times New Roman', labelpad=10)
    # ax.set_zlabel('Dimension 3', fontsize=18, fontname='Times New Roman', labelpad=10)
    # ax.tick_params(axis='both', which='major', labelsize=12)#字体大小
    # plt.savefig('D:/code/two-stream/Feature map/t-nse/F_features_x2_t-SNE_3D.png', dpi=300, format='png')
    # plt.show()
if __name__ == '__main__':
    main()