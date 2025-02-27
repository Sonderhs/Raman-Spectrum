import pandas as pd
import numpy as np

# 配置
input_file = "D:\\Python\\Python project\\My_project\\chemical\\code\\prediction.csv"  # 输入 CSV 文件路径
output_file = "D:\\Python\\Python project\\My_project\\chemical\\code\\prediction_pre.csv"  # 输出归一化后文件路径

# 读取 CSV 文件
data = pd.read_csv(input_file, header=None)  # 如果有表头，去掉 header=None


# 定义 Min-Max 归一化函数
def minmax_normalize(row):
    min_val = np.min(row)
    max_val = np.max(row)
    if max_val - min_val == 0:
        return row  # 如果最大值等于最小值，返回原行（避免除以零）
    return (row - min_val) / (max_val - min_val)


# 对每一行进行归一化
normalized_data = data.apply(minmax_normalize, axis=1)

# 保存归一化后的数据到新的 CSV 文件
normalized_data.to_csv(output_file, index=False, header=False)

print(f"归一化完成，结果已保存到 {output_file}")
