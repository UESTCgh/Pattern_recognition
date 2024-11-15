import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# 加载训练数据和标签
train_data = sio.loadmat('../data/train_data.mat')['train_data']  # 假设训练数据是二维数组，每行代表一个样本
train_label = pd.read_csv('../data/train_label.csv', header=None).values.ravel()  # 训练标签，转换为一维数组

# 加载测试数据和标签
test_data = sio.loadmat('../data/test_data.mat')['test_data']
test_label_manual = sio.loadmat('../data/test_label_manual.mat')['test_label_manual'].ravel()  # 测试标签

# 数据标准化处理 - 标准化有助于基于距离的分类器
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# 使用SMOTE进行数据增强，增加少数类样本，使数据集更加平衡
smote = SMOTE(random_state=42)
train_data_resampled, train_label_resampled = smote.fit_resample(train_data, train_label)

# 使用网格搜索来寻找最佳的随机森林参数
param_grid = {
    'n_estimators': [50, 100, 150],       # 森林中树的数量
    'max_depth': [10, 20, None],          # 树的最大深度
    'min_samples_split': [2, 5, 10],      # 分裂节点的最小样本数
    'min_samples_leaf': [1, 2, 4]         # 叶节点的最小样本数
}
rf = RandomForestClassifier(random_state=42)

# 使用5折交叉验证找到最佳超参数
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(train_data_resampled, train_label_resampled)

# 输出最佳的参数
best_params = grid_search.best_params_
print(f"最佳的随机森林参数为: {best_params}")

# 使用最佳参数重新训练随机森林分类器
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(train_data_resampled, train_label_resampled)

# 测试集数据增强（加入一些随机噪声以增强泛化能力）
noise_factor = 0
test_data_noisy = test_data + noise_factor * np.random.normal(size=test_data.shape)

# 对测试集进行预测
test_predictions = best_rf.predict(test_data_noisy)

# 将测试结果保存到 test_label.txt 文件中
with open('../data/test_label.txt', 'w') as f:
    for idx, label in enumerate(test_predictions, start=1):
        f.write(f"{idx} {label}\n")

# 计算多种评价指标
accuracy = accuracy_score(test_label_manual, test_predictions)
precision = precision_score(test_label_manual, test_predictions, average='binary', pos_label=1)
recall = recall_score(test_label_manual, test_predictions, average='binary', pos_label=1)
f1 = f1_score(test_label_manual, test_predictions, average='binary', pos_label=1)
conf_matrix = confusion_matrix(test_label_manual, test_predictions)

# 打印评价指标
print(f"测试集准确率: {accuracy:.2f}")
print(f"测试集精确率: {precision:.2f}")
print(f"测试集召回率: {recall:.2f}")
print(f"测试集F1得分: {f1:.2f}")
print("混淆矩阵:")
print(conf_matrix)

# 打印分类报告
print("\n分类报告:")
print(classification_report(test_label_manual, test_predictions))