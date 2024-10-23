import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置字体，确保支持中文
# 如果你有安装SimHei字体（黑体），可以使用这个字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载CSV文件到DataFrame
linear_data = pd.read_csv('E:/GitHub/AI_VI/2.BPSVM/ROC/data/linear_roc_curve_data.csv')
poly_data = pd.read_csv('E:/GitHub/AI_VI/2.BPSVM/ROC/data/poly_roc_curve_data.csv')
rbf_data = pd.read_csv('E:/GitHub/AI_VI/2.BPSVM/ROC/data/rbf_roc_curve_data.csv')
roc_values = pd.read_csv('E:/GitHub/AI_VI/2.BPSVM/ROC/data/roc_values.csv')

# 绘制线性核的ROC曲线
plt.plot(linear_data['FPR'], linear_data['TPR'], label='linear核')

# 绘制多项式核的ROC曲线
plt.plot(poly_data['FPR'], poly_data['TPR'], label='poly核')

# 绘制RBF核的ROC曲线
plt.plot(rbf_data['FPR'], rbf_data['TPR'], label='rbf核')

# 如果需要，可以绘制额外的ROC数据
plt.plot(roc_values['FPR'], roc_values['TPR'], label='BP')

# 设置标签和标题
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真正率 (TPR)')
plt.title('不同核函数的ROC曲线')

# 显示图例
plt.legend()

# 保存图片到指定路径，格式可以是 'png', 'jpg', 'pdf' 等
plt.savefig('E:/GitHub/AI_VI/2.BPSVM/ROC/roc_curve.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.close()
