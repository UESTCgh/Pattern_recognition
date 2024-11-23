import numpy as np
import scipy.io as sio
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

from torchvision import transforms
import os
# ---------------------------- 设置训练设备 ----------------------------

# 设置设备为 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------- 超参数设置 ----------------------------
# 学习率和训练轮次
learning_rate = 0.008
epochs = 100

# 早停相关设置
early_stop_patience = 5
best_loss = float('inf')
patience_counter = 0

# 图像大小，假设为 19x19（可以根据需要调整）
image_height = 19
image_width = 19

#训练规模
batch_size = 16
# 设置是否进行数据增强
apply_data_augmentation = False
# 给类别 1 （正类）更高的权重
class_weights = torch.tensor([1, 160.0]).to(device)  # 可以根据实际情况调整权重
class_weights = torch.tensor([1, 140.0]).to(device)  # 可以根据实际情况调整权重
#训练参数
weight_decay = 1e-4  # 权重衰减，用于 L2 正则化
scheduler_factor = 0.5  # 学习率调度器减少学习率的因子
scheduler_patience = 3  # 学习率调度器的耐心值
# ---------------------------- 数据加载与预处理 ----------------------------

# 加载训练数据和标签
train_data = sio.loadmat('../data/train_data.mat')['train_data']  # 训练数据
train_label = pd.read_csv('../data/train_label.csv', header=None).values.ravel()  # 训练标签

# 加载测试数据和标签
test_data = sio.loadmat('../data/test_data.mat')['test_data']  # 测试数据
test_label_manual = sio.loadmat('../data/test_label_manual.mat')['test_label_manual'].ravel()  # 测试标签

# 数据标准化处理
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# 使用SMOTE进行数据增强（解决类别不平衡）
smote = SMOTE(random_state=42)
train_data_resampled, train_label_resampled = smote.fit_resample(train_data, train_label)

# 标签转换：将 -1 转换为 0，符合 CrossEntropyLoss 要求
train_label_resampled = np.where(train_label_resampled == -1, 0, train_label_resampled)
test_label_manual = np.where(test_label_manual == -1, 0, test_label_manual)

# 将训练数据转换为图像格式
train_data_resampled = train_data_resampled.reshape((-1, 1, image_height, image_width))
test_data = test_data.reshape((-1, 1, image_height, image_width))

# ---------------------------- 数据增强选项 ----------------------------
if apply_data_augmentation:
    # ---------------------------- 数据增强 - 镜像和加噪声 ----------------------------
    def add_gaussian_noise(images, mean=0, std=0.1):
        noise = torch.randn_like(images) * std + mean
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, 0., 1.)  # 限制像素值在0到1之间
        return noisy_images
    # 将训练数据转换为 Tensor
    train_data_tensor = torch.tensor(train_data_resampled, dtype=torch.float32)
    # 水平镜像翻转
    flipped_data = torch.flip(train_data_tensor, dims=[3])  # 对第3个维度（宽度）进行水平翻转
    # 添加高斯噪声
    noisy_data = add_gaussian_noise(train_data_tensor)
    # 合并原始数据、镜像数据和加噪声数据
    augmented_data = torch.cat((train_data_tensor, flipped_data, noisy_data), dim=0)
    # 合并标签（每种增强方式都有与原始数据相同的标签）
    augmented_labels = np.concatenate((train_label_resampled, train_label_resampled, train_label_resampled))
    # 打印扩充后的数据集数量
    print(f"扩充后的训练数据集数量: {augmented_data.shape[0]}")
    print(f"扩充后的训练标签数量: {augmented_labels.shape[0]}")
else:
    # 如果不进行数据增强，则使用原始数据
    augmented_data = torch.tensor(train_data_resampled, dtype=torch.float32)
    augmented_labels = train_label_resampled
    # 打印原始训练数据集数量
    print(f"原始训练数据集数量: {augmented_data.shape[0]}")
    print(f"原始训练标签数量: {augmented_labels.shape[0]}")
# ---------------------------- 自定义数据集类 ----------------------------

class FacialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.data = data
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建训练和测试数据集及数据加载器
train_dataset = FacialDataset(train_data_resampled, train_label_resampled)
test_dataset = FacialDataset(test_data, test_label_manual)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
train_dataset = FacialDataset(augmented_data, augmented_labels)
test_dataset = FacialDataset(torch.tensor(test_data, dtype=torch.float32), test_label_manual)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
# ---------------------------- 定义 CNN 模型 ----------------------------
class CNNModel(nn.Module):
    def __init__(self, image_height, image_width):
        super(CNNModel, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Batch Normalization 层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * (image_height // 8) * (image_width // 8), 128)
        self.fc2 = nn.Linear(128, 2)

        # Dropout 层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积层 + BatchNorm + 激活 + 池化
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(nn.ReLU()(self.bn3(self.conv3(x))))

        # 展平操作
        x = x.view(-1, 128 * (image_height // 8) * (image_width // 8))

        # 全连接层
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# ---------------------------- 创建模型和优化器 ----------------------------

# 创建模型实例
model = CNNModel(image_height, image_width).to(device)

# ---------------------------- 配置训练参数 ----------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)  # 动态调整学习率
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 优化器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True
)  # 学习率调度器
# ---------------------------- 训练模型 ----------------------------

# 训练循环
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # 调整学习率
    scheduler.step(avg_loss)

    # 早停条件判断
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        print("早停触发，停止训练。")
        break

# 在整个训练完成后保存模型
torch.save(model.state_dict(), 'final_trained_model.pth')
print("训练完成，模型已保存为 'final_trained_model.pth'")
# ---------------------------- 测试模型 ----------------------------

# 测试循环
# 加载训练好的模型
model.load_state_dict(torch.load('final_trained_model.pth'))
model.eval()
all_predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())

# 计算多种评价指标
accuracy = accuracy_score(test_label_manual, all_predictions)
precision = precision_score(test_label_manual, all_predictions, average='binary', pos_label=1)
recall = recall_score(test_label_manual, all_predictions, average='binary', pos_label=1)
f1 = f1_score(test_label_manual, all_predictions, average='binary', pos_label=1)
conf_matrix = confusion_matrix(test_label_manual, all_predictions)

# 打印评价指标
print(f"测试集准确率: {accuracy:.2f}")
print(f"测试集精确率: {precision:.2f}")
print(f"测试集召回率: {recall:.2f}")
print(f"测试集F1得分: {f1:.2f}")
print("混淆矩阵:")
print(conf_matrix)

# 打印分类报告
print("\n分类报告:")
print(classification_report(test_label_manual, all_predictions))
# ---------------------------- 测试本地模型 ----------------------------
# print("************************本地模型******************************")
# # 加载训练好的模型
# model.load_state_dict(torch.load('model.pth'))
# model.eval()
#
# all_predictions = []
# with torch.no_grad():
#     for inputs, _ in test_loader:
#         inputs = inputs.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         all_predictions.extend(predicted.cpu().numpy())
#
# # 计算多种评价指标
# accuracy = accuracy_score(test_label_manual, all_predictions)
# precision = precision_score(test_label_manual, all_predictions, average='binary', pos_label=1)
# recall = recall_score(test_label_manual, all_predictions, average='binary', pos_label=1)
# f1 = f1_score(test_label_manual, all_predictions, average='binary', pos_label=1)
# conf_matrix = confusion_matrix(test_label_manual, all_predictions)
#
# # 打印评价指标
# print(f"本地模型测试集准确率: {accuracy:.2f}")
# print(f"本地模型测试集精确率: {precision:.2f}")
# print(f"本地模型测试集召回率: {recall:.2f}")
# print(f"本地模型测试集F1得分: {f1:.2f}")
# print("本地模型混淆矩阵:")
# print(conf_matrix)
#
# # 打印分类报告
# print("\n本地模型分类报告:")
# print(classification_report(test_label_manual, all_predictions))