import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据过滤范围
MIN_HEIGHT = 140.0  # cm
MAX_HEIGHT = 220.0
MIN_WEIGHT = 30.0   # kg
MAX_WEIGHT = 150.0
MIN_SHOE_SIZE = 30.0
MAX_SHOE_SIZE = 50.0
MIN_50M_TIME = 5.0  # seconds
MAX_50M_TIME = 15.0

# 1. 数据预处理函数
def preprocess_data(input_file, output_file, invalid_file):
    # 读取CSV文件，指定中文列名
    df = pd.read_csv(input_file)

    # 尝试将相关列转换为float类型，处理异常值
    try:
        df['身高(cm)'] = pd.to_numeric(df['身高(cm)'], errors='coerce')
        df['体重(kg)'] = pd.to_numeric(df['体重(kg)'], errors='coerce')
        df['鞋码'] = pd.to_numeric(df['鞋码'], errors='coerce')
        df['50米成绩'] = pd.to_numeric(df['50米成绩'], errors='coerce')
        df['性别 男1女0'] = pd.to_numeric(df['性别 男1女0'], errors='coerce')
    except KeyError:
        print("确保CSV文件中包含正确的列名：身高(cm), 体重(kg), 鞋码, 50米成绩, 性别 男1女0")
        return

    # 只保留性别列中值为0或1的样本，确保标签是二分类
    df = df[df['性别 男1女0'].isin([0, 1])]

    # 检查并过滤掉无效数据
    valid_data = df[
        (df['身高(cm)'] >= MIN_HEIGHT) & (df['身高(cm)'] <= MAX_HEIGHT) &
        (df['体重(kg)'] >= MIN_WEIGHT) & (df['体重(kg)'] <= MAX_WEIGHT) &
        (df['鞋码'] >= MIN_SHOE_SIZE) & (df['鞋码'] <= MAX_SHOE_SIZE) &
        (df['50米成绩'] >= MIN_50M_TIME) & (df['50米成绩'] <= MAX_50M_TIME)
    ]

    # 无效数据：使用`isna()`检查哪些数据被转换为NaN
    invalid_data = df[~df.index.isin(valid_data.index)]

    # 保存数据到对应文件
    valid_data.to_csv(output_file, index=False)
    invalid_data.to_csv(invalid_file, index=False)

    print(f"数据过滤结果保存在 {output_file}")
    print(f"无效数据保存在 {invalid_file}")

# 2. 使用交叉验证评估SVM分类器性能
def train_and_evaluate_svm_with_cv(filtered_data_file):
    # 读取数据
    df = pd.read_csv(filtered_data_file)

    # 提取特征和标签
    X = df[['身高(cm)', '体重(kg)', '鞋码', '50米成绩']]
    y = df['性别 男1女0']

    # 使用LabelEncoder进行编码（确保y只包含0和1）
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 使用StratifiedKFold进行分层交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 初始化不同核函数的SVM模型并进行评估
    kernels = ['linear', 'poly', 'rbf']
    results = []  # 用于存储所有核函数的性能指标
    for kernel in kernels:
        print(f"正在训练和评估核函数: {kernel}")

        # 为当前核函数创建单独的子目录
        kernel_result_dir = os.path.join("result", kernel)
        if not os.path.exists(kernel_result_dir):
            os.makedirs(kernel_result_dir)

        # 初始化SVM分类器
        clf = SVC(kernel=kernel, C=1.0, gamma='scale', probability=True)

        # 交叉验证预测
        y_pred = cross_val_predict(clf, X, y, cv=skf, method='predict')
        y_pred_proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]

        # 计算性能指标
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 灵敏度（Sensitivity, SE）
        sensitivity = tp / (tp + fn)
        # 特异度（Specificity, SP）
        specificity = tn / (tn + fp)
        # 准确率（Accuracy, ACC）
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        # 曲线下面积（AUC）
        auc_score = roc_auc_score(y, y_pred_proba)

        # 保存当前核函数的指标到results中
        results.append({
            'kernel': kernel,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'auc': auc_score
        })

        # 输出性能指标
        print(f"***************{kernel}性能指标***********************")
        print("混淆矩阵:")
        print(cm)
        print(f"灵敏度（Sensitivity, SE）: {sensitivity:.4f}")
        print(f"特异度（Specificity, SP）: {specificity:.4f}")
        print(f"准确率（Accuracy, ACC）: {accuracy:.4f}")
        print(f"曲线下面积（AUC）: {auc_score:.4f}")

        # 可视化混淆矩阵并保存
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('预测值')
        plt.ylabel('真实值')
        plt.title(f'混淆矩阵 - 核函数: {kernel}')
        plt.tight_layout()
        confusion_matrix_path = os.path.join(kernel_result_dir, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.show()

        # 可视化ROC曲线并保存
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_score:.4f}')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlabel('假阳性率 (False Positive Rate)')
        plt.ylabel('真阳性率 (True Positive Rate)')
        plt.title(f'ROC 曲线 - 核函数: {kernel}')
        plt.legend(loc="lower right")
        plt.grid()
        plt.tight_layout()
        roc_curve_path = os.path.join(kernel_result_dir, "roc_curve.png")
        plt.savefig(roc_curve_path)
        plt.show()

        # 可视化灵敏度、特异度和准确率并保存
        metrics = {'灵敏度 (SE)': sensitivity, '特异度 (SP)': specificity, '准确率 (ACC)': accuracy}
        plt.figure(figsize=(8, 5))
        plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'orange', 'green'])
        plt.ylabel('得分')
        plt.ylim(0, 1)
        plt.title(f'性能指标 - 核函数: {kernel}')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        metrics_path = os.path.join(kernel_result_dir, "metrics.png")
        plt.savefig(metrics_path)
        plt.show()
        print(f"***************{kernel}结果保存***********************")
        print(f"核函数 {kernel} 的混淆矩阵保存到: {confusion_matrix_path}")
        print(f"核函数 {kernel} 的ROC曲线保存到: {roc_curve_path}")
        print(f"核函数 {kernel} 的性能指标图保存到: {metrics_path}")
        print(f"***************{kernel}结束***********************")

    # 3. 绘制并保存三种核函数的性能指标对比图
    vs_dir = os.path.join("result", "vs")
    if not os.path.exists(vs_dir):
        os.makedirs(vs_dir)
    print(f"*****************显示指标对比图***********************")
    metrics_to_plot = ['sensitivity', 'specificity', 'accuracy', 'auc']
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 8))
        values = [result[metric] for result in results]
        kernels = [result['kernel'] for result in results]
        bars = plt.bar(kernels, values, color=['#4E79A7', '#F28E2C', '#76B7B2'], alpha=0.85)
        plt.ylabel('得分', fontsize=14)
        plt.ylim(0, 1)
        plt.title(f'{metric.upper()} 对比', fontsize=18)
        plt.grid(axis='y', linestyle='--')

        # 在每个柱状图上显示具体数值
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{bar.get_height():.2f}',
                     ha='center', va='bottom', fontsize=12, color='black', weight='bold')

        plt.tight_layout()
        comparison_path = os.path.join(vs_dir, f"{metric}_comparison.png")
        plt.savefig(comparison_path)
        plt.show()


# 主程序
if __name__ == "__main__":
    # 数据文件路径
    input_file = "data/data.csv"
    filtered_data_file = "data/filtered_data.csv"
    invalid_data_file = "data/invalid_data.csv"

    # 如果输出目录不存在，创建它
    if not os.path.exists("data"):
        os.makedirs("data")

    # 1. 数据预处理
    print(f"*******************数据预处理*******************")
    preprocess_data(input_file, filtered_data_file, invalid_data_file)

    # 2. SVM模型训练与交叉验证评估
    print(f"*******************模型训练与评估*******************")
    train_and_evaluate_svm_with_cv(filtered_data_file)
