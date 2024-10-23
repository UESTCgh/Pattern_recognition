import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 设置字体，确保支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体，支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义目录路径（相对路径）
data_dir = './data/'
result_dir = './result/'

# 如果result目录不存在，则创建它
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 定义一个函数，用于绘制并保存所有ROC曲线到一张图
def plot_and_save_all_roc():
    # 加载CSV文件到DataFrame
    linear_data = pd.read_csv(os.path.join(data_dir, 'linear_roc_curve_data.csv'))
    poly_data = pd.read_csv(os.path.join(data_dir, 'poly_roc_curve_data.csv'))
    rbf_data = pd.read_csv(os.path.join(data_dir, 'rbf_roc_curve_data.csv'))
    roc_values = pd.read_csv(os.path.join(data_dir, 'roc_values.csv'))

    # 绘制线性核的ROC曲线
    plt.plot(linear_data['FPR'], linear_data['TPR'], label='linear核')

    # 绘制多项式核的ROC曲线
    plt.plot(poly_data['FPR'], poly_data['TPR'], label='poly核')

    # 绘制RBF核的ROC曲线
    plt.plot(rbf_data['FPR'], rbf_data['TPR'], label='rbf核')

    # 绘制额外的ROC数据（如BP）
    plt.plot(roc_values['FPR'], roc_values['TPR'], label='BP')

    # 设置标签和标题
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('不同模型的ROC曲线')

    # 显示图例
    plt.legend()

    # 保存图片到result目录，格式可以是 'png', 'jpg', 'pdf' 等
    plt.savefig(os.path.join(result_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')

    # 关闭当前图形，以便后续可以重新绘制
    plt.close()


# 定义函数来绘制和保存单独的ROC曲线
def plot_and_save_roc(data_file, label, save_path):
    """
    读取指定的CSV文件并绘制ROC曲线，然后保存图片。

    参数:
    - data_file: CSV文件的路径
    - label: 曲线的标签（如线性核、多项式核等）
    - save_path: 保存图片的路径（包括文件名和格式）
    """
    # 读取CSV文件
    data = pd.read_csv(data_file)

    # 绘制ROC曲线
    plt.plot(data['FPR'], data['TPR'], label=label)

    # 设置标签和标题
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title(f'{label}的ROC曲线')

    # 显示图例
    plt.legend()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 关闭当前图形，以便后续绘图不被干扰
    plt.close()

# 调用函数进行绘制并保存所有曲线到一张图
plot_and_save_all_roc()

# 分别绘制并保存每个单独的曲线到result目录
plot_and_save_roc(os.path.join(data_dir, 'linear_roc_curve_data.csv'), '线性核', os.path.join(result_dir, 'linear_roc_curve.png'))
plot_and_save_roc(os.path.join(data_dir, 'poly_roc_curve_data.csv'), '多项式核', os.path.join(result_dir, 'poly_roc_curve.png'))
plot_and_save_roc(os.path.join(data_dir, 'rbf_roc_curve_data.csv'), 'RBF核', os.path.join(result_dir, 'rbf_roc_curve.png'))
plot_and_save_roc(os.path.join(data_dir, 'roc_values.csv'), 'BP', os.path.join(result_dir, 'bp_roc_curve.png'))
