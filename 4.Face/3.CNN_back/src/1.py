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
import matplotlib.pyplot as plt

# 设置字体，SimHei 字体可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    return device

# 设置设备
device = setup_device()

# -----------------------------设置超参数---------------------------------
learning_rate = 0.008
epochs = 100
early_stop_patience = 10
image_height, image_width = 19, 19
batch_size = 16
apply_data_augmentation = False
class_weights = torch.tensor([1, 160.0]).to(device)
weight_decay = 5e-4
scheduler_factor = 0.7
scheduler_patience = 4

# 数据预处理

def load_and_preprocess_data(image_height, image_width, apply_data_augmentation):
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

    # 可视化原始数据和增强后的数据集分布
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label, alpha=0.5, cmap='viridis')
    plt.title('原始训练数据集分布')
    plt.xlabel('特征1')
    plt.ylabel('特征2')

    plt.subplot(1, 2, 2)
    plt.scatter(train_data_resampled[:, 0], train_data_resampled[:, 1], c=train_label_resampled, alpha=0.5, cmap='viridis')
    plt.title('SMOTE增强后的数据集分布')
    plt.xlabel('特征1')
    plt.ylabel('特征2')

    plt.tight_layout()
    plt.show()

    # 标签转换：将 -1 转换为 0，符合 CrossEntropyLoss 要求
    train_label_resampled = np.where(train_label_resampled == -1, 0, train_label_resampled)
    test_label_manual = np.where(test_label_manual == -1, 0, test_label_manual)

    # 将训练数据转换为图像格式
    train_data_resampled = train_data_resampled.reshape((-1, 1, image_height, image_width))
    test_data = test_data.reshape((-1, 1, image_height, image_width))

    # ---------------------------- 数据增强选项 ----------------------------
    if apply_data_augmentation:
        train_data_resampled, train_label_resampled = apply_data_augmentations(train_data_resampled,
                                                                               train_label_resampled)

    augmented_data = torch.tensor(train_data_resampled, dtype=torch.float32).clone().detach()
    augmented_labels = train_label_resampled
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

    # 打印训练数据集数量
    print(f"训练数据集数量: {augmented_data.shape[0]}")
    print(f"训练标签数量: {augmented_labels.shape[0]}")

    return augmented_data, augmented_labels, test_data_tensor, test_label_manual


def apply_data_augmentations(train_data, train_labels):
    def add_gaussian_noise(images, mean=0, std=1):
        noise = torch.randn_like(images) * std + mean
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, 0., 1.)  # 限制像素值在0到1之间
        return noisy_images

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    flipped_data = torch.flip(train_data_tensor, dims=[3])  # 水平镜像翻转
    noisy_data = add_gaussian_noise(train_data_tensor)
    augmented_data = torch.cat((train_data_tensor, flipped_data, noisy_data), dim=0)
    augmented_labels = np.concatenate((train_labels, train_labels, train_labels))

    # 可视化数据增强后的数据分布
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(train_data_tensor[0, 0], cmap='gray')
    plt.title('原始数据样本')

    plt.subplot(1, 3, 2)
    plt.imshow(flipped_data[0, 0], cmap='gray')
    plt.title('水平翻转后的样本')

    plt.subplot(1, 3, 3)
    plt.imshow(noisy_data[0, 0], cmap='gray')
    plt.title('添加高斯噪声后的样本')

    plt.tight_layout()
    plt.show()

    return augmented_data, augmented_labels

class FacialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.clone().detach()  # 修复用户警告
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


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


def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs, early_stop_patience):
    import matplotlib.pyplot as plt
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0

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
        train_losses.append(avg_loss)
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

    # 绘制损失曲线
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b', label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True)
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'final_trained_model.pth')
    print("训练完成，模型已保存为 'final_trained_model.pth'")

def evaluate_model(model, test_loader, test_labels, device, threshold=0.4):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    model.load_state_dict(torch.load('final_trained_model.pth'))
    model.eval()

    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            all_probabilities.extend(probabilities.cpu().numpy())
            predicted = (probabilities > threshold).long()
            all_predictions.extend(predicted.cpu().numpy())

    # 绘制 ROC 和 PR 曲线
    fpr, tpr, _ = roc_curve(test_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
    plt.fill_between(fpr, tpr, alpha=0.3, color='blue')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('假阳性率')
    plt.ylabel('真正率')
    plt.title('ROC 曲线')
    plt.legend(loc='lower right')

    precision, recall, _ = precision_recall_curve(test_labels, all_probabilities)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='b', lw=2)
    plt.fill_between(recall, precision, alpha=0.3, color='blue')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('PR 曲线')

    plt.tight_layout()
    plt.show()

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(test_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    plt.figure(figsize=(8, 8))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('混淆矩阵')
    plt.show()

    return all_predictions

def save_predictions(predictions, output_file='model_predictions.txt'):
    final_predictions = [1 if label == 1 else -1 for label in predictions]
    with open(output_file, 'w') as f:
        for idx, label in enumerate(final_predictions, start=1):
            f.write(f"{idx} {label}\n")
    print(f"预测结果已保存到 '{output_file}'")

def main():

    # 加载和预处理数据
    augmented_data, augmented_labels, test_data, test_label_manual = load_and_preprocess_data(
        image_height, image_width, apply_data_augmentation)

    # 创建数据集和数据加载器
    train_dataset = FacialDataset(augmented_data, augmented_labels)
    test_dataset = FacialDataset(test_data.clone().detach(), test_label_manual)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # 创建模型
    model = CNNModel(image_height, image_width).to(device)

    # 配置训练参数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs, early_stop_patience)

    # 测试模型
    all_predictions = evaluate_model(model, test_loader, test_label_manual, device)

    # 计算评价指标
    accuracy = accuracy_score(test_label_manual, all_predictions)
    precision = precision_score(test_label_manual, all_predictions, average='binary', pos_label=1)
    recall = recall_score(test_label_manual, all_predictions, average='binary', pos_label=1)
    f1 = f1_score(test_label_manual, all_predictions, average='binary', pos_label=1)
    conf_matrix = confusion_matrix(test_label_manual, all_predictions)

    print(f"测试集准确率: {accuracy:.2f}")
    print(f"测试集精确率: {precision:.2f}")
    print(f"测试集召回率: {recall:.2f}")
    print(f"测试集F1得分: {f1:.2f}")
    print("混淆矩阵:")
    print(conf_matrix)
    print("\n分类报告:")
    print(classification_report(test_label_manual, all_predictions))

    # 保存预测结果
    save_predictions(all_predictions)


if __name__ == "__main__":
    main()
