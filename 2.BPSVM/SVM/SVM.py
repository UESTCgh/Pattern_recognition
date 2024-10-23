import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
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

# 训练编号的全局变量
training_id = 1
num = 1  # 训练总次数

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
    global training_id  # 声明全局变量，以便更新训练编号

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

    # 创建vs目录用于保存汇总评估结果
    vs_dir = os.path.join("result", "vs")
    if not os.path.exists(vs_dir):
        os.makedirs(vs_dir)

    # 汇总评估结果的CSV文件路径
    summary_results_csv_path = os.path.join(vs_dir, "summary_results.csv")

    # 如果 CSV 文件不存在，创建文件并添加表头
    if not os.path.exists(summary_results_csv_path):
        df_headers = pd.DataFrame(
            columns=["Training_ID", "Kernel", "Accuracy", "Sensitivity", "Specificity", "AUC", "TP", "FP", "TN", "FN"])
        df_headers.to_csv(summary_results_csv_path, index=False)

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

        # 保存各核函数的性能指标到CSV文件中，增加 "训练编号" 字段
        results_dict = {
            "Training_ID": training_id,
            "Kernel": kernel,
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "AUC": auc_score,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        }
        results_df = pd.DataFrame([results_dict])
        results_df.to_csv(summary_results_csv_path, mode='a', index=False, header=False)

        print(f"核函数 {kernel} 的性能指标汇总保存到: {summary_results_csv_path}")

        # 如果是最后一次训练，保存可视化结果
        if training_id == num:
            print(f"保存第 {training_id} 次训练的可视化结果 - 核函数: {kernel}")

            # 保存模型参数到文本文件（每个核函数独立）
            model_params_path = os.path.join(kernel_result_dir, "model_params.txt")
            with open(model_params_path, "w") as file:
                file.write(f"kernel: {kernel}\n")
                file.write(f"params: {clf.get_params()}\n")

            print(f"模型参数保存到: {model_params_path}")

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
            plt.close()  # 关闭图表，防止显示

            # 可视化ROC曲线并保存
            fpr, tpr, thresholds = roc_curve(y, y_pred_proba)  # 获取假阳性率、真阳性率和阈值数据
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
            plt.close()  # 关闭图表，防止显示

            # 保存ROC曲线数据到CSV文件
            roc_data_path = os.path.join(kernel_result_dir, f"{kernel}_roc_curve_data.csv")
            roc_data = pd.DataFrame({
                'FPR': fpr,  # 假阳性率
                'TPR': tpr,  # 真阳性率
                'Thresholds': thresholds  # 阈值
            })
            roc_data.to_csv(roc_data_path, index=False)
            print(f"ROC 曲线数据保存到: {roc_data_path}")

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
            plt.close()  # 关闭图表，防止显示

    # 完成一轮训练后，递增训练编号
    training_id += 1


# 3. 计算平均值并绘图
def calculate_average_and_plot():
    # 读取所有训练的汇总结果
    summary_results_csv_path = os.path.join("result", "vs", "summary_results.csv")
    df = pd.read_csv(summary_results_csv_path)

    # 计算每个核函数的平均值
    avg_results = df.groupby('Kernel').mean().reset_index()

    # 保存平均值结果到一个新的 CSV 文件
    average_results_csv_path = os.path.join("result", "vs", "average_results.csv")
    avg_results.to_csv(average_results_csv_path, index=False)
    print(f"正在保存前 {training_id - 1} 次训练的平均结果到: {average_results_csv_path}")

    # 绘制平均性能指标图
    metrics_to_plot = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 8))
        values = avg_results[metric].values
        kernels = avg_results['Kernel'].values
        bars = plt.bar(kernels, values, color=['#4E79A7', '#F28E2C', '#76B7B2'], alpha=0.85)
        plt.ylabel('得分', fontsize=14)
        plt.ylim(0, 1)
        plt.title(f'{metric} 平均值对比', fontsize=18)
        plt.grid(axis='y', linestyle='--')

        # 在每个柱状图上显示具体数值
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{bar.get_height():.4f}',
                     ha='center', va='bottom', fontsize=12, color='black', weight='bold')

        plt.tight_layout()
        comparison_path = os.path.join("result", "vs", f"{metric}_average_comparison.png")
        plt.savefig(comparison_path)
        plt.close()

# 4. 特征组合投影函数，针对三种核函数进行可视化（实现特征两两组合）
def feature_combination_projection(filtered_data_file):
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

    # 核函数列表
    kernels = ['linear', 'poly', 'rbf']
    feature_names = ['身高', '体重', '鞋码', '50米成绩']

    # 获取所有两两特征组合的索引
    feature_combinations = [(i, j) for i in range(len(feature_names)) for j in range(i + 1, len(feature_names))]

    # 对每个核函数进行投影和可视化
    for kernel in kernels:
        for (i, j) in feature_combinations:
            # 使用两个特征进行组合投影
            X_selected = X[:, [i, j]]

            # 创建子文件夹保存每个核函数的结果
            touying_file = os.path.join("result/keshihua", kernel)
            if not os.path.exists(touying_file):
                os.makedirs(touying_file)

            # 使用 SVM 重新训练模型
            clf = SVC(kernel=kernel, C=1.0, gamma='scale', probability=True)
            clf.fit(X_selected, y)

            # 创建网格以绘制决策边界
            x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
            y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))

            # 使用训练好的模型进行预测
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # 绘制决策边界和支持向量
            plt.figure(figsize=(10, 6))
            plt.contourf(xx, yy, Z, alpha=0.8)
            plt.scatter(X_selected[:, 0], X_selected[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Paired)
            plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                        facecolors='none', edgecolors='k', linewidth=1.5)  # 支持向量
            plt.xlabel(f'{feature_names[i]} (标准化)')
            plt.ylabel(f'{feature_names[j]} (标准化)')
            plt.title(f'特征组合投影 - 核函数: {kernel}')
            plt.tight_layout()

            # 保存特征组合投影图到对应子文件夹
            projection_path = os.path.join(touying_file, f"{kernel}_feature_combination_{feature_names[i]}_{feature_names[j]}.png")
            plt.savefig(projection_path)
            plt.close()  # 关闭图表，防止显示
            print(f"特征组合投影图保存到: {projection_path}")

            # 绘制性能指标图
            metrics = ['准确率', '灵敏度', '特异度', 'AUC']
            # 使用交叉验证得到的预测结果计算性能
            y_pred = cross_val_predict(clf, X_selected, y, cv=5)
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            auc_score = roc_auc_score(y, y_pred)

            performance_values = [accuracy, sensitivity, specificity, auc_score]

            plt.figure(figsize=(8, 5))
            bars = plt.bar(metrics, performance_values, color=['#4E79A7', '#F28E2C', '#76B7B2', '#E15759'], alpha=0.85)
            plt.ylabel('得分')
            plt.ylim(0, 1)
            plt.title(f'性能指标 - 核函数: {kernel} ({feature_names[i]} & {feature_names[j]})')
            plt.grid(axis='y', linestyle='--')

            # 在每个柱状图上显示具体数值
            for bar in bars:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{bar.get_height():.4f}',
                         ha='center', va='bottom', fontsize=12, color='black', weight='bold')

            performance_path = os.path.join(touying_file, f"{kernel}_performance_metrics_{feature_names[i]}_{feature_names[j]}.png")
            plt.savefig(performance_path)
            plt.close()
            print(f"性能指标图保存到: {performance_path}")



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

    # 2. SVM模型训练与交叉验证评估，执行 num 次
    print(f"*******************模型训练与评估*******************")
    for i in range(num):
        print(f"*******************第 {i + 1} 次训练*******************")
        train_and_evaluate_svm_with_cv(filtered_data_file)

    # 3. 计算平均值并绘图
    print(f"*******************计算平均值并绘制图表*******************")
    calculate_average_and_plot()

    # 4. 特征组合投影
    print(f"*******************特征组合投影*******************")
    #feature_combination_projection(filtered_data_file)

