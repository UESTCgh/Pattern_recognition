# 重新编写 Python 脚本，以读取 'mat.csv' 文件，去除包含 0 的数据，并计算前四列的均值和方差

script_content_updated = """
import pandas as pd

def process_mat_file(file_path):
    try:
        # 读取文件
        df_mat = pd.read_csv(file_path)
        
        # 提取前四列
        df_first_four_columns = df_mat.iloc[:, :4]
        
        # 去除包含 0 的行
        df_non_zero = df_first_four_columns[(df_first_four_columns != 0).all(axis=1)]
        
        # 计算均值和方差
        mean_values = df_non_zero.mean()
        variance_values = df_non_zero.var()
    
        # 创建新的 DataFrame 存储均值和方差
        result_df = pd.DataFrame({
            'Mean': mean_values,
            'Variance': variance_values
        })
    
        # 保存结果到 CSV 文件
        result_output_path = 'mean_variance_results.csv'
        result_df.to_csv(result_output_path, index=False)
    
        print(f"均值和方差计算完成，结果已保存在 '{result_output_path}' 文件中。")

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请确认文件是否存在。")

if __name__ == "__main__":
    # 指定文件路径
    file_path = 'mat.csv'
    process_mat_file(file_path)
"""

# 保存新的代码到 Python 文件
script_file_path_updated = '/mnt/data/calculate_mean_variance_v2.py'

with open(script_file_path_updated, 'w') as file:
    file.write(script_content_updated)

script_file_path_updated
