import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 创建结果保存目录
result_dir = './result'
os.makedirs(result_dir, exist_ok=True)

result_M_dir = './result/Model_only'
os.makedirs(result_M_dir, exist_ok=True)

result_Compare_dir = './result/Model_Compare'
os.makedirs(result_Compare_dir, exist_ok=True)

# 读取表格文件 (假设文件名为 'model_results.xlsx')
# 如果是CSV文件，请将文件名和相应方法改为 read_csv
data_bp = pd.read_csv('./data/bp.csv')
data_linear = pd.read_csv('./data/linear.csv')
data_rbf = pd.read_csv('./data/rbf.csv')
data_poly = pd.read_csv('./data/poly.csv')

# 计算 Precision 和 Recall
data_bp['Precision'] = data_bp['TP'] / (data_bp['TP'] + data_bp['FP'])
data_bp['Recall'] = data_bp['TP'] / (data_bp['TP'] + data_bp['FN'])

data_linear['Precision'] = data_linear['TP'] / (data_linear['TP'] + data_linear['FP'])
data_linear['Recall'] = data_linear['TP'] / (data_linear['TP'] + data_linear['FN'])

data_rbf['Precision'] = data_rbf['TP'] / (data_rbf['TP'] + data_rbf['FP'])
data_rbf['Recall'] = data_rbf['TP'] / (data_rbf['TP'] + data_rbf['FN'])

data_poly['Precision'] = data_poly['TP'] / (data_poly['TP'] + data_poly['FP'])
data_poly['Recall'] = data_poly['TP'] / (data_poly['TP'] + data_poly['FN'])

# 查看数据是否读取正确
print(data_bp.head())
print(data_linear.head())
print(data_rbf.head())
print(data_poly.head())

# 设置 Seaborn 的样式
sns.set(style="whitegrid")


# 通用可视化函数
def visualize_model(data, model_name):
    plt.figure(figsize=(16, 10))

    # 子图 1: Accuracy, Sensitivity, Specificity 对比
    plt.subplot(2, 2, 1)
    sns.boxplot(data=data[['Accuracy', 'Sensitivity', 'Specificity']])
    plt.title(f'{model_name} Model Performance Metrics')
    plt.ylabel('Score')

    # 子图 2: AUC
    plt.subplot(2, 2, 2)
    sns.histplot(data['AUC'], kde=True, color='blue')
    plt.title(f'{model_name} Model AUC Distribution')
    plt.xlabel('AUC')
    plt.ylabel('Frequency')

    # 子图 3: 混淆矩阵中的 TP, FP, TN, FN
    plt.subplot(2, 2, 3)
    sample_data = data[['TP', 'FP', 'TN', 'FN']].iloc[::max(1, len(data) // 20)]  # 仅绘制部分数据以减少拥挤
    sample_data.plot(kind='bar', stacked=True, ax=plt.gca(), color=['skyblue', 'orange', 'green', 'red'])
    plt.title(f'{model_name} Model Confusion Matrix Values (Sampled)')
    plt.xlabel('Sampled Training Iterations')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    # 子图 4: Precision and Recall
    plt.subplot(2, 2, 4)
    sns.boxplot(data=data[['Precision', 'Recall']])
    plt.title(f'{model_name} Model Precision and Recall')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.savefig(os.path.join(result_M_dir, f'{model_name}_performance.png'))
    plt.close()


# 可视化多个模型对比
def visualize_model_comparison(models_data, model_names):
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'Precision', 'Recall']
    comparison_data = pd.DataFrame()

    for data, name in zip(models_data, model_names):
        model_metrics = data[metrics].mean()
        model_metrics['Model'] = name
        comparison_data = pd.concat([comparison_data, pd.DataFrame([model_metrics])], ignore_index=True)

    plt.figure(figsize=(16, 10))

    # 子图 1: Accuracy, Sensitivity, Specificity 对比
    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='value', hue='variable',
                data=pd.melt(comparison_data, id_vars=['Model'], value_vars=['Accuracy', 'Sensitivity', 'Specificity']))
    plt.title('Model Comparison: Accuracy, Sensitivity, Specificity')
    plt.ylabel('Score')

    # 子图 2: AUC
    plt.subplot(2, 2, 2)
    sns.barplot(x='Model', y='AUC', data=comparison_data)
    plt.title('Model Comparison: AUC')
    plt.ylabel('AUC')

    # 子图 3: Precision
    plt.subplot(2, 2, 3)
    sns.barplot(x='Model', y='Precision', data=comparison_data)
    plt.title('Model Comparison: Precision')
    plt.ylabel('Precision')

    # 子图 4: Recall
    plt.subplot(2, 2, 4)
    sns.barplot(x='Model', y='Recall', data=comparison_data)
    plt.title('Model Comparison: Recall')
    plt.ylabel('Recall')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'model_comparison.png'))
    plt.close()


# 可视化多个模型对比
def visualize_model_comparison1(models_data, model_names):
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'Precision', 'Recall']
    comparison_data = pd.DataFrame()

    for data, name in zip(models_data, model_names):
        model_metrics = data[metrics].mean()
        model_metrics['Model'] = name
        comparison_data = pd.concat([comparison_data, pd.DataFrame([model_metrics])], ignore_index=True)

    # 图 1: Accuracy, Sensitivity, Specificity 对比
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='value', hue='variable',
                data=pd.melt(comparison_data, id_vars=['Model'], value_vars=['Accuracy', 'Sensitivity', 'Specificity']))
    plt.title('Model Comparison: Accuracy, Sensitivity, Specificity')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(result_Compare_dir, 'performance_metrics.png'))
    plt.close()

    # 图 2: AUC 对比
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='AUC', data=comparison_data)
    plt.title('Model Comparison: AUC')
    plt.ylabel('AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(result_Compare_dir, 'auc.png'))
    plt.close()

    # 图 3: Precision 对比
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Precision', data=comparison_data)
    plt.title('Model Comparison: Precision')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig(os.path.join(result_Compare_dir, 'precision.png'))
    plt.close()

    # 图 4: Recall 对比
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Recall', data=comparison_data)
    plt.title('Model Comparison: Recall')
    plt.ylabel('Recall')
    plt.tight_layout()
    plt.savefig(os.path.join(result_Compare_dir, 'recall.png'))
    plt.close()

    # 图 1: 综合对比 (Accuracy, Sensitivity, Specificity, AUC, Precision, Recall)
    plt.figure(figsize=(14, 8))
    sns.set_palette('Set2')
    sns.barplot(x='Model', y='value', hue='variable',
                data=pd.melt(comparison_data, id_vars=['Model'], value_vars=metrics))
    plt.title('Model Comparison Across All Metrics')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(result_Compare_dir, 'all.png'))
    plt.close()

# 调用可视化函数
visualize_model(data_bp, "BP")
visualize_model(data_linear, "SVM_linear")
visualize_model(data_rbf, "SVM_rbf")
visualize_model(data_poly, "SVM_poly")

# # 调用模型对比可视化函数
visualize_model_comparison([data_bp, data_linear, data_rbf, data_poly], ["BP", "SVM_linear", "SVM_rbf", "SVM_poly"])

# # 调用模型保存可视化函数
# visualize_model_comparison1([data_bp, data_linear, data_rbf, data_poly], ["BP", "SVM_linear", "SVM_rbf", "SVM_poly"])
