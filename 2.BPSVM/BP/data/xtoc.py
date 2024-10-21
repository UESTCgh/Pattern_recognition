import pandas as pd

def xlsx_to_csv(xlsx_file, csv_file):
    # 读取xlsx文件
    df = pd.read_excel(xlsx_file)

    # 统计男生和女生的数量
    male_count = df[df['性别 男1女0'] == 1].shape[0]
    female_count = df[df['性别 男1女0'] == 0].shape[0]
    
    # 如果女生数量少于男生，通过复制扩充女生数量与男生一样多
    if female_count < male_count:
        female_df = df[df['性别 男1女0'] == 0]
        copies_needed = (male_count - female_count) // female_count + 1
        expanded_female_df = pd.concat([female_df] * copies_needed, ignore_index=True).iloc[:(male_count - female_count)]
        df = pd.concat([df, expanded_female_df], ignore_index=True)
    
    # 将数据保存到csv文件，处理中文编码问题
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

# 示例用法
xlsx_file = 'data.xlsx'  # 你的xlsx文件路径
csv_file = 'data.csv'     # 输出的csv文件路径

xlsx_to_csv(xlsx_file, csv_file)
