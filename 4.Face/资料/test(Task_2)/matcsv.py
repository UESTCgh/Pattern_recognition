import scipy.io
import pandas as pd
import os

# 获取当前目录下所有文件
current_dir = os.getcwd()
files = os.listdir(current_dir)

# 遍历所有文件
for file in files:
    # 只处理 .mat 文件
    if file.endswith('.mat'):
        # 载入 .mat 文件
        mat_data = scipy.io.loadmat(file)
        
        # 获取所有变量名（排除元数据）
        variables = [key for key in mat_data.keys() if not key.startswith('__')]
        
        for var in variables:
            # 获取每个变量的数据
            data = mat_data[var]

            # 如果数据是二维数组，转换为DataFrame
            if isinstance(data, (list, tuple)) or len(data.shape) == 2:
                df = pd.DataFrame(data)

                # 输出为 CSV 文件，命名为原文件名 + .csv（不含行号）
                output_file = os.path.splitext(file)[0] + '.csv'
                df.to_csv(output_file, index=False, header=False)

                print(f"{file} 中的变量 {var} 已转换为 {output_file}")
            else:
                print(f"{file} 中的变量 {var} 不适合转换为 DataFrame")
