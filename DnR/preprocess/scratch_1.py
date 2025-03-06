import os
import pandas as pd
import numpy as np
from PIL import Image

# 设置文件夹路径
folder_path = "/home/user2/data/MVI数据/ROI_224"  # 替换为你的文件夹路径
save_path = "/home/user2/data/MVI数据/ROI_224_npy"  # 替换为你的保存路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 获取文件夹中所有图片文件
files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 创建一个空的DataFrame
df = pd.DataFrame(columns=['文件名', '病人编号', '标签', '模态'])

# 遍历文件，提取信息并添加到DataFrame
for file in files:
    patient_id = file.split('_')[0][:2]  # 提取病人编号
    label = file.split('_')[0][2:4]      # 提取标签
    modality = file.split('_')[1]        # 提取模态
    new_row = pd.DataFrame({'文件名': [file], '病人编号': [patient_id], '标签': [label], '模态': [modality]})
    df = pd.concat([df, new_row], ignore_index=True)

# 检查每个病人的每个标签和模态的图片数量
def check_images(df):
    result = []
    for patient_id in df['病人编号'].unique():
        for label in df[df['病人编号'] == patient_id]['标签'].unique():  # 遍历每个标签
            for modality in df[(df['病人编号'] == patient_id) & (df['标签'] == label)]['模态'].unique():  # 遍历每个模态
                count = len(df[(df['病人编号'] == patient_id) &
                               (df['标签'] == label) &
                               (df['模态'] == modality)])
                if count != 3:
                    result.append((patient_id, label, modality, count))  # 添加标签到结果
    return result

# 执行检查
missing_images = check_images(df)

# 输出检查结果
print("检查结果：")
if missing_images:
    for item in missing_images:
        print(f"病人编号：{item[0]}, 标签：{item[1]}, 模态：{item[2]}, 图片数量：{item[3]}（应为3张）")
else:
    print("所有病人的每个标签和模态都有3张图片。")

# 将每个病人的每个模态的每个标签的三张图片堆叠成三通道并保存为.npy文件
def stack_and_save_images(df, folder_path, save_path):
    for patient_id in df['病人编号'].unique():
        for label in df[df['病人编号'] == patient_id]['标签'].unique():
            for modality in df[(df['病人编号'] == patient_id) & (df['标签'] == label)]['模态'].unique():
                # 获取对应病人的标签和模态的所有图片
                image_files = df[(df['病人编号'] == patient_id) &
                                 (df['标签'] == label) &
                                 (df['模态'] == modality)]['文件名'].values
                if len(image_files) == 3:
                    # 对图片文件名进行排序（按数字部分排序）
                    image_files = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    # 读取图片并堆叠
                    images = []
                    for img in image_files:
                        image_path = os.path.join(folder_path, img)
                        image = Image.open(image_path)  # 打开图片
                        image = image.resize((256, 256))  # 调整大小到256×256
                        images.append(np.array(image))  # 转换为NumPy数组
                    stacked_image = np.stack(images)  # 堆叠成三通道
                    # 保存为.npy文件
                    output_filename = f"{patient_id}_{modality}_{label}.npy"
                    np.save(os.path.join(save_path, output_filename), stacked_image)
                    print(f"保存文件：{output_filename}")

# 执行堆叠和保存操作
stack_and_save_images(df, folder_path, save_path)