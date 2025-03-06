import os
import numpy as np
import pandas as pd

# 设置输入和输出路径
input_folder = "/home/user2/data/MVI数据/ROI_224_npy"  # 替换为你的.npy文件夹路径
output_folder = "/home/user2/data/MVI数据/ROI_224_processed_npy"  # 替换为输出文件夹路径
output_csv = "/home/user2/data/MVI数据/ROI_224_processed_npy/output_index.csv"  # 保存索引信息的CSV文件路径

# 检查输出路径是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有.npy文件
npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

# 定义滑动窗口参数
window_size = 64
step_size = 32

# 初始化一个空的DataFrame用于保存索引信息
index_df = pd.DataFrame(columns=["file_name", "idx_overall"])

# 初始化索引计数器
idx_overall = 0
# 遍历每个.npy文件
for npy_file in npy_files:
    file_path = os.path.join(input_folder, npy_file)
    data = np.load(file_path)  # 加载.npy文件

    # 假设data是一个3D数组，形状为 (channels, height, width)
    channels, height, width = data.shape

    # 遍历每个可能的裁剪位置
    for y in range(0, height - window_size + 1, step_size):
        for x in range(0, width - window_size + 1, step_size):
            # 裁剪图像
            image_z_minus_1 = data[0, y:y + window_size, x:x + window_size]  # (64, 64)
            image_z_plus_1 = data[2, y:y + window_size, x:x + window_size]  # (64, 64)
            image_he = data[1, y:y + window_size, x:x + window_size]  # (64, 64)

            # 创建image和image_he
            image = np.stack([image_z_minus_1, image_z_plus_1], axis=0)  # (2, 64, 64)
            image_he = image_he[None, :, :]  # (1, 64, 64)

            # 寻找重叠50%的另一个位置
            overlap_x = x + window_size // 2
            overlap_y = y + window_size // 2

            # 确保重叠区域不超出边界
            if overlap_x + window_size > width:
                overlap_x = width - window_size
            if overlap_y + window_size > height:
                overlap_y = height - window_size

            # 裁剪重叠区域图像
            image_pairs_z_minus_1 = data[0, overlap_y:overlap_y + window_size, overlap_x:overlap_x + window_size]
            image_pairs_z_plus_1 = data[2, overlap_y:overlap_y + window_size, overlap_x:overlap_x + window_size]
            image_pairs_he = data[1, overlap_y:overlap_y + window_size, overlap_x:overlap_x + window_size]

            # 创建image_pairs和image_pairs_he
            image_pairs = np.stack([image_pairs_z_minus_1, image_pairs_z_plus_1], axis=0)  # (2, 64, 64)
            image_pairs_he = image_pairs_he[None, :, :]  # (1, 64, 64)

            # 保存结果
            output_data = {
                "image": image,  # (2, 64, 64)
                "image_he": image_he,  # (1, 64, 64)
                "image_pairs": image_pairs,  # (2, 64, 64)
                "image_pairs_he": image_pairs_he,  # (1, 64, 64)
                "idx_overall": idx_overall  # 索引编号
            }

            # 保存为新的.npy文件
            output_filename = f"{idx_overall}.npy"
            np.save(os.path.join(output_folder, output_filename), output_data)

            # 将文件名和idx_overall保存到DataFrame
            new_row = pd.DataFrame({"file_name": [npy_file], "idx_overall": [idx_overall]})
            index_df = pd.concat([index_df, new_row], ignore_index=True)

            idx_overall += 1

    print(f"处理完成：{npy_file}")

# 保存索引信息到CSV文件
index_df.to_csv(output_csv, index=False)
print(f"索引信息已保存到：{output_csv}")