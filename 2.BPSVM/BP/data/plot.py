import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# 创建结果目录以保存图片
results_dir = 'result'
os.makedirs(results_dir, exist_ok=True)

# 加载损失值数据
loss_file_path = 'loss_values.csv'
loss_data = pd.read_csv(loss_file_path)

# 加载性能指标数据
metrics_file_path = 'mat.csv'
metrics = pd.read_csv(metrics_file_path)

# 清理性能指标数据，去掉空值和非数值行
metrics = metrics.dropna()  # 删除包含 NaN 的行
metrics = metrics.apply(pd.to_numeric, errors='coerce')  # 将所有数据转换为数值，无法转换的变为 NaN
metrics = metrics.dropna()  # 再次删除无法转换的 NaN 数据行

# 绘制损失曲线
def plot_loss_curve(loss_data, results_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_data['Epoch'], loss_data['Loss'], marker='o', linestyle='-', color='b')
    plt.title('Training Loss over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # 保存损失图像
    plot_path = os.path.join(results_dir, 'training_loss_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

# 绘制混淆矩阵
def plot_confusion_matrix(metrics, results_dir):
    # 提取混淆矩阵数据
    fold_num = 1
    for index, row in metrics.iterrows():
        try:
            # 读取混淆矩阵的各个值
            tn = int(row['TN'])
            fp = int(row['FP'])
            fn = int(row['FN'])
            tp = int(row['TP'])

            conf_matrix = [[tn, fp],
                           [fn, tp]]
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'])
            plt.title(f'Confusion Matrix - Fold {fold_num}', fontsize=16)
            plt.xlabel('Predicted Label', fontsize=14)
            plt.ylabel('True Label', fontsize=14)
            plt.tight_layout()

            # 保存混淆矩阵图像
            plot_path = os.path.join(results_dir, f'confusion_matrix_fold_{fold_num}.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            fold_num += 1
        except ValueError as e:
            print(f"Skipping due to invalid data. Error: {e}")

# 绘制四个性能指标的对比图
def plot_metrics_comparison(metrics, results_dir):
    plt.figure(figsize=(12, 8))

    # 计算实验次数编号
    experiment_numbers = metrics.index / 2  # 假设你想让实验次数的横坐标除以 2

    # 绘制 Accuracy 曲线
    plt.plot(experiment_numbers, metrics['Accuracy'], marker='o', linestyle='-', label='Accuracy', color='b')

    # 绘制 Sensitivity 曲线
    plt.plot(experiment_numbers, metrics['Sensitivity'], marker='o', linestyle='-', label='Sensitivity', color='r')

    # 绘制 Specificity 曲线
    plt.plot(experiment_numbers, metrics['Specificity'], marker='o', linestyle='-', label='Specificity', color='g')

    # 绘制 AUC 曲线
    plt.plot(experiment_numbers, metrics['AUC'], marker='o', linestyle='-', label='AUC', color='orange')

    # 设置标题和坐标轴标签
    plt.title('Performance Metrics Comparison across Different Experiments', fontsize=16)
    plt.xlabel('Experiment Number', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)

    # 添加图例
    plt.legend()

    # 显示网格线
    plt.grid(True)

    # 调整布局并保存图像
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'metrics_comparison_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')

    # 显示图像
    plt.close()

# 主函数
def main():
    # 打印列名以检查格式是否正确
    print("Loaded loss_values.csv columns:", loss_data.columns)
    print("Loaded mat.csv columns:", metrics.columns)

    # 绘制损失曲线
    plot_loss_curve(loss_data, results_dir)

    # 绘制混淆矩阵
    plot_confusion_matrix(metrics, results_dir)

    # 绘制四个性能指标的对比图
    plot_metrics_comparison(metrics, results_dir)

if __name__ == "__main__":
    main()
