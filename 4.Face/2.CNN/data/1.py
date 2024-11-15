import numpy as np
import scipy.io as sio
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# 步骤1: 加载训练数据和标签
train_data_file = 'train_label.csv'  # 假设这个是标签文件路径
train_label_file = 'train_label.mat'  # 假设这个是数据文件路径

# 加载标签
train_labels = pd.read_csv(train_data_file, header=None).values.ravel()  # 标签数据

# 加载训练数据
train_data = sio.loadmat(train_label_file)['train_data']  # 数据

# 步骤2: 高斯噪声扩充
def add_gaussian_noise(image, mean=0, std=0.1):
    """
    给图像添加高斯噪声
    :param image: 输入图像
    :param mean: 噪声均值
    :param std: 噪声标准差
    :return: 添加噪声后的图像
    """
    noise = torch.randn_like(image) * std + mean  # 生成高斯噪声
    noisy_image = image + noise  # 添加噪声
    noisy_image = torch.clamp(noisy_image, 0., 1.)  # 限制像素值范围在0到1之间
    return noisy_image

# 将训练数据转换为Tensor并添加噪声
train_data_tensor = torch.tensor(train_data, dtype=torch.float32)

# 用于存储扩充后的数据
augmented_data = []
augmented_labels = []

# 生成加噪声的数据集
for i in range(len(train_data_tensor)):
    original_image = train_data_tensor[i].unsqueeze(0)  # 选取一个样本，增加一个维度（模拟批量）
    noisy_image = add_gaussian_noise(original_image)  # 添加噪声

    augmented_data.append(original_image.squeeze(0).numpy())  # 保存原始图像
    augmented_labels.append(train_labels[i])  # 保存标签

    augmented_data.append(noisy_image.squeeze(0).numpy())  # 保存加噪后的图像
    augmented_labels.append(train_labels[i])  # 保存标签

# 步骤3: 保存扩充后的数据集
augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)

# 保存为.mat文件（可以根据需要修改保存方式）
augmented_data_file = 'augmented_train_data.mat'
augmented_labels_file = 'augmented_train_labels.csv'

# 保存数据
sio.savemat(augmented_data_file, {'augmented_train_data': augmented_data})
pd.DataFrame(augmented_labels).to_csv(augmented_labels_file, index=False, header=False)

print("数据扩充完毕并保存：", augmented_data_file, augmented_labels_file)
