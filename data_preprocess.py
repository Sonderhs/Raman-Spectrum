import os
import pandas as pd


def merge_second_columns(input_folder, output_file):
    """
    合并文件夹中所有 CSV 文件的第二列到一个新的 CSV 文件中。

    参数：
    - input_folder: 存放 CSV 文件的文件夹路径
    - output_file: 输出合并后的 CSV 文件路径
    """
    # 获取文件夹中的所有 CSV 文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    if not csv_files:
        print("文件夹中没有找到 CSV 文件")
        return

    # 用于存储每个文件的第二列
    combined_data = pd.DataFrame()

    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)

        # 读取 CSV 文件
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] < 2:
                print(f"文件 {csv_file} 的列数少于两列，跳过...")
                continue

            # 提取第二列，并将其添加到合并的数据中
            combined_data[csv_file] = df.iloc[:, 1]
        except Exception as e:
            print(f"读取文件 {csv_file} 时出错: {e}")

    # 保存合并后的数据到新的 CSV 文件
    combined_data.to_csv(output_file, index=False)
    print(f"所有文件的第二列已合并并保存到 {output_file}")


# 使用示例
input_folder = "D:\\Python\\Python project\\My_project\\chemical\\code\\data\\EHT：BD=0.1：0.1"
output_file = "D:\\Python\\Python project\\My_project\\chemical\\code\\data\\EHT：BD=0.1：0.1合并结果.csv"
merge_second_columns(input_folder, output_file)
