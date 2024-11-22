import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

def save_mat_images(filepath, output_folder):
    # 读取 mat 文件
    mat_data = scipy.io.loadmat(filepath)

    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 计数器，用于命名图像文件
    image_counter = 0

    # 遍历 mat 文件中的所有变量
    for key in mat_data:
        if not key.startswith("__"):  # 忽略系统变量
            data = mat_data[key]
            
            # 检查数据是否是一维的并且长度为 361，或者是二维的并且可以包含 19x19 图像
            if data.ndim == 1 and data.size == 19 * 19:
                # 处理单个一维数组
                try:
                    image_data = data.reshape(19, 19)
                except ValueError:
                    print(f"变量 '{key}' 的数据不能转换为 19x19 形状，跳过。")
                    continue

                # 保存图像到指定文件夹
                image_path = os.path.join(output_folder, f"image_{image_counter:03d}.png")
                plt.imshow(image_data, cmap='gray')
                plt.axis('off')  # 关闭坐标轴
                plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
                plt.close()

                image_counter += 1

            elif data.ndim == 2 and data.shape[1] == 19 * 19:
                # 处理二维数组，每一行都可以转换为 19x19 图像
                for i in range(data.shape[0]):
                    try:
                        image_data = data[i].reshape(19, 19)
                    except ValueError:
                        print(f"变量 '{key}' 中第 {i} 行的数据不能转换为 19x19 形状，跳过。")
                        continue

                    # 保存图像到指定文件夹
                    image_path = os.path.join(output_folder, f"image_{image_counter:03d}.png")
                    plt.imshow(image_data, cmap='gray')
                    plt.axis('off')  # 关闭坐标轴
                    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    image_counter += 1

    if image_counter == 0:
        print("没有找到合适的数组来转换为 19x19 图像。")
    else:
        print(f"成功保存了 {image_counter} 张图像到文件夹 '{output_folder}' 中。")

# 示例用法
filepath = 'train_data.mat'
output_folder = 'train_images'  # 你可以替换为你希望保存图像的路径
save_mat_images(filepath, output_folder)
