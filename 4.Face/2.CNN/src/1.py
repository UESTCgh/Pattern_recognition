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

# ---------------------------- 设置训练设备 ----------------------------

# 设置设备为 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------- 超参数设置 ----------------------------

# 学习率和训练轮次
learning_rate = 0.007  # 调整学习率，使模型训练更稳定
epochs = 100  # 增加训练轮次

# 早停相关设置
early_stop_patience = 5  # 提高耐心值，让模型有更多轮次可以改善
best_loss = float('inf')
patience_counter = 0

# 图像大小，假设为 19x19（可以根据需要调整）
image_height = 19
image_width = 19

<<<<<<< HEAD
# 训练规模
batch_size = 16  # 增大批次规模

# 设置是否进行数据增强
apply_data_augmentation = False

# 给类别 1 （正类）更高的权重
class_weights = torch.tensor([1, 190.0]).to(device)  # 调整权重，增加对少数类的关注

# 训练参数
weight_decay = 5e-4  # 增加权重衰减，减少过拟合的风险
scheduler_factor = 0.7  # 学习率调度器减少学习率的因子
scheduler_patience = 2  # 学习率调度器的耐心值

=======
# 给类别 1 （正类）更高的权重
class_weights = torch.tensor([1, 160.0]).to(device)  # 可以根据实际情况调整权重
>>>>>>> parent of 82162fc (11.15 模型修改)
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

# ---------------------------- 自定义数据集类 ----------------------------

class FacialDataset(Dataset):
    def __init__(self, data, labels):
<<<<<<< HEAD
        self.data = data.clone().detach()  # 修复用户警告
=======
        self.data = torch.tensor(data, dtype=torch.float32)
>>>>>>> parent of 82162fc (11.15 模型修改)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建训练和测试数据集及数据加载器
<<<<<<< HEAD
train_dataset = FacialDataset(augmented_data, augmented_labels)
test_dataset = FacialDataset(torch.tensor(test_data, dtype=torch.float32).clone().detach(), test_label_manual)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
=======
train_dataset = FacialDataset(train_data_resampled, train_label_resampled)
test_dataset = FacialDataset(test_data, test_label_manual)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---------------------------- 定义 CNN 模型 ----------------------------
class CNNModel(nn.Module):
    def __init__(self, image_height, image_width):
        super(CNNModel, self).__init__()
>>>>>>> parent of 82162fc (11.15 模型修改)

# ---------------------------- 定义 SE 模块 ----------------------------
class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # [B, C]
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class CNNModelWithMoreConvAndAttention(nn.Module):
    def __init__(self, image_height, image_width):
        super(CNNModelWithMoreConvAndAttention, self).__init__()

        # 卷积层，增加两层卷积
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 新增卷积层
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 新增卷积层

        # Batch Normalization 层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        # 注意力机制（SE模块）
        self.se1 = SEModule(32)
        self.se2 = SEModule(64)
        self.se3 = SEModule(128)
        self.se4 = SEModule(256)  # 新增注意力模块
        self.se5 = SEModule(512)  # 新增注意力模块

        # 池化层，减少池化次数以防止特征图尺寸为零
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层，调整输入维度以适配新增卷积层
        # 注意：减少池化操作后，特征图的尺寸应该有所调整
        self.fc1 = nn.Linear(512 * (image_height // 8) * (image_width // 8), 256)
        self.fc2 = nn.Linear(256, 2)

        # Dropout 层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积层 + BatchNorm + 注意力 + 激活 + 池化
        x = self.pool(nn.ReLU()(self.se1(self.bn1(self.conv1(x)))))
        x = self.pool(nn.ReLU()(self.se2(self.bn2(self.conv2(x)))))
        x = nn.ReLU()(self.se3(self.bn3(self.conv3(x))))  # 第三层不使用池化
        x = self.pool(nn.ReLU()(self.se4(self.bn4(self.conv4(x)))))  # 第四个卷积层 + 池化
        x = nn.ReLU()(self.se5(self.bn5(self.conv5(x))))  # 第五层不使用池化

        # 展平操作
        x = x.view(-1, 512 * (image_height // 8) * (image_width // 8))

        # 全连接层
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ---------------------------- 创建模型和优化器 ----------------------------
# 替换原有模型为新增卷积层并保留注意力机制的模型
model = CNNModelWithMoreConvAndAttention(image_height, image_width).to(device)


criterion = nn.CrossEntropyLoss(weight=class_weights)  # 损失函数
<<<<<<< HEAD
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 优化器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True
)  # 学习率调度器
=======
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)  # 动态调整学习率
>>>>>>> parent of 82162fc (11.15 模型修改)

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

# ---------------------------- 测试模型 ----------------------------

# 测试循环
model.eval()
all_predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # 获取正类概率
        predicted = (probabilities > 0.45).long()  # 设置阈值为 0.4
        all_predictions.extend(predicted.cpu().numpy())


# 计算多种评价指标
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
<<<<<<< HEAD

# ---------------------------- 测试模型并保存预测结果 ----------------------------

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

# 将预测结果从0/1转换为-1/1的格式
final_predictions = [1 if label == 1 else -1 for label in all_predictions]

# 保存预测结果到文本文件
with open('model_predictions.txt', 'w') as f:
    for idx, label in enumerate(final_predictions, start=1):
        f.write(f"{idx} {label}\n")
print("预测结果已保存到 'model_predictions.txt'")
=======
>>>>>>> parent of 82162fc (11.15 模型修改)
