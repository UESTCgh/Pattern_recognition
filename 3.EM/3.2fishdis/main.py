import cv2
import os
import joblib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# 定义结果保存目录
RESULT_DIR = 'result'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# 确保保存合并结果的目录存在
COMBINED_DIR = 'result/combined'
if not os.path.exists(COMBINED_DIR):
    os.makedirs(COMBINED_DIR)


def extract_fish_with_grabcut(image_path, save_path):
    """
    使用 GrabCut 提取鱼的区域，并将结果保存。同时保存中间过程和最终结果合并的图片。

    参数:
    image_path: str - 输入图片路径
    save_path: str - 保存最终分割结果的路径
    """
    # 1. 读取图像并转换为 RGB 格式
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. HSV 和 Lab 色彩空间颜色过滤
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # 定义橙色和白色的 HSV 和 Lab 范围
    lower_orange_hsv = np.array([5, 100, 100])
    upper_orange_hsv = np.array([25, 255, 255])
    lower_white_hsv = np.array([0, 0, 200])
    upper_white_hsv = np.array([180, 30, 255])

    lower_orange_lab = np.array([20, 135, 130])
    upper_orange_lab = np.array([255, 180, 175])
    lower_white_lab = np.array([200, 0, 0])
    upper_white_lab = np.array([255, 135, 135])

    # 创建掩码并合并
    mask_orange_hsv = cv2.inRange(hsv, lower_orange_hsv, upper_orange_hsv)
    mask_white_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
    mask_orange_lab = cv2.inRange(lab, lower_orange_lab, upper_orange_lab)
    mask_white_lab = cv2.inRange(lab, lower_white_lab, upper_white_lab)
    combined_mask = cv2.bitwise_or(mask_orange_hsv, mask_white_hsv)
    combined_mask = cv2.bitwise_or(combined_mask, mask_orange_lab)
    combined_mask = cv2.bitwise_or(combined_mask, mask_white_lab)

    # 3. 形态学操作去除噪声
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

    # 4. 应用 Canny 边缘检测以获取轮廓
    edges = cv2.Canny(opened_mask, 100, 200)

    # 5. 查找轮廓并提取最大的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        fish_contour = max(contours, key=cv2.contourArea)
    else:
        print(f"未能找到轮廓: {image_path}")
        return

    # 创建空掩码并绘制最大的轮廓
    contour_mask = np.zeros_like(combined_mask)
    cv2.drawContours(contour_mask, [fish_contour], -1, 255, thickness=cv2.FILLED)

    # 6. 使用形态学操作填充小的孔洞
    filled_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel)

    # 7. 提取鱼的区域并将背景设置为黑色
    fish_result = cv2.bitwise_and(image_rgb, image_rgb, mask=filled_mask)

    # 8. 保存最终的分割结果
    result_bgr = cv2.cvtColor(fish_result, cv2.COLOR_RGB2BGR)  # 转回 BGR 用于保存
    cv2.imwrite(save_path, result_bgr)
    print(f"最终的 GrabCut 扣出结果已保存: {save_path}")

    # 9. 合并各个步骤的图像，保存为 2 行 3 列的合成图
    combined_steps = [
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),  # 原始图片
        combined_mask,  # 合并掩码
        closed_mask,  # 闭操作结果
        opened_mask,  # 开操作结果
        contour_mask,  # 轮廓掩码
        result_bgr  # 最终分割结果
    ]

    # 将所有图像的尺寸统一为相同大小
    resized_steps = [
        cv2.resize(step, (300, 200)) if len(step.shape) == 3 else cv2.cvtColor(cv2.resize(step, (300, 200)),
                                                                               cv2.COLOR_GRAY2BGR) for step in
        combined_steps]

    # 合并图像为 2 行 3 列
    top_row = np.hstack(resized_steps[:3])
    bottom_row = np.hstack(resized_steps[3:])
    combined_image = np.vstack((top_row, bottom_row))

    # 保存合并图像到 combined 目录中
    combined_save_path = os.path.join(COMBINED_DIR, os.path.basename(save_path).replace(".bmp", "_combined_steps.bmp"))
    cv2.imwrite(combined_save_path, combined_image)
    print(f"分割过程合并图已保存到组合文件夹: {combined_save_path}")

def apply_gmm_to_image(gmm_model_path, image_path, output_path, mode='rgb', normalize=True):
    """
    将 GMM 模型应用到输入图像上并保存聚类结果。

    参数:
    gmm_model_path: str - GMM 模型路径
    image_path: str - 输入图片路径
    output_path: str - 输出图片保存路径
    mode: str - 图像模式，支持 'gray' 或 'rgb'
    normalize: bool - 是否对图像数据进行归一化
    """
    # 1. 加载 GMM 模型
    gmm = joblib.load(gmm_model_path)
    image = Image.open(image_path)

    # 2. 获取图像数据
    if mode == 'gray':
        image = image.convert('L')
        image_data = np.array(image)
        reshaped_data = image_data.reshape(-1, 1)
    elif mode == 'rgb':
        image = image.convert('RGB')
        image_data = np.array(image)
        reshaped_data = image_data.reshape(-1, 3)
    else:
        raise ValueError("模式选择错误。请使用 'gray' 或 'rgb'。")

    # 3. 归一化处理
    if normalize:
        reshaped_data = reshaped_data / 255.0

    # 4. 模型预测
    labels = gmm.predict(reshaped_data)

    # 获取聚类均值作为代表颜色
    cluster_colors = gmm.means_
    if normalize:
        cluster_colors = cluster_colors * 255

    # 替换每个像素的颜色为其对应的聚类颜色
    clustered_image_data = cluster_colors[labels]

    # 5. 恢复图像尺寸
    if mode == 'gray':
        clustered_image_data = clustered_image_data.reshape(image_data.shape).astype(np.uint8)
    else:
        clustered_image_data = clustered_image_data.reshape(image_data.shape[0], image_data.shape[1], 3).astype(np.uint8)

    # 6. 保存相关结果
    no_boundaries_output_path = output_path.replace(".jpg", f"_{mode}_no_boundaries.jpg")
    Image.fromarray(clustered_image_data).save(no_boundaries_output_path)
    print(f"聚类结果: {no_boundaries_output_path}")

    # 保存带边界的结果
    labels_image = labels.reshape(image_data.shape[0], image_data.shape[1])
    clustered_image_with_boundaries = np.copy(clustered_image_data)

    unique_labels = np.unique(labels_image)
    for label in unique_labels:
        label_mask = (labels_image == label).astype(np.float64)
        contours = measure.find_contours(label_mask, 0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

    plt.imshow(clustered_image_with_boundaries)
    plt.title("Clustered Image with Boundaries")
    plt.axis('off')
    boundaries_output_path = output_path.replace(".jpg", f"_{mode}_with_boundaries.jpg")
    plt.savefig(boundaries_output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 关闭绘图窗口
    print(f"边界线聚类结果: {boundaries_output_path}")

def process_all_images(input_dir, output_dir, gmm_model_name, image_files, mode='rgb', normalize=True):
    """
    处理目录下的所有图片，应用 GMM 模型并保存结果。

    参数:
    input_dir: str - 输入图片所在的目录
    output_dir: str - 输出结果保存目录
    gmm_model_name: str - GMM 模型文件名
    image_files: list - 待处理图片文件名列表
    mode: str - 图像模式，支持 'gray' 或 'rgb'
    normalize: bool - 是否对图像数据进行归一化
    """
    gmm_model_path = os.path.join('data', gmm_model_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file.replace('.bmp', '_clustered.jpg'))

        try:
            apply_gmm_to_image(gmm_model_path, image_path, output_path, mode, normalize)
        except Exception as e:
            print(f"处理 {image_file} 时出现错误: {e}")

    print("所有图片处理完成并保存到结果目录。")

# 主函数
if __name__ == "__main__":
    data_dir = 'data'
    image_files = ['311.bmp', '313.bmp', '315.bmp', '317.bmp']

    # 使用 GrabCut 提取鱼的区域并保存到 result 目录
    for image_file in image_files:
        image_path = os.path.join(data_dir, image_file)
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        grabcut_save_path = os.path.join(RESULT_DIR, f"{base_name}_grabcut.bmp")

        # 使用 GrabCut 提取鱼的区域
        extract_fish_with_grabcut(image_path, save_path=grabcut_save_path)

    # 使用 GMM 处理 GrabCut 结果并保存
    output_dir_gray = 'result/gray'
    output_dir_rgb = 'result/rgb'
    gmm_model1_name = 'gmm_final_model.pkl'
    gmm_model2_name = 'gmm_rgb_final_model.pkl'

    grabcut_result_files = [f"{base_name}_grabcut.bmp" for base_name in ['311', '313', '315', '317']]
    process_all_images(RESULT_DIR, output_dir_gray, gmm_model1_name, grabcut_result_files, mode='gray', normalize=True)
    process_all_images(RESULT_DIR, output_dir_rgb, gmm_model2_name, grabcut_result_files, mode='rgb', normalize=True)

    process_all_images(data_dir, output_dir_gray, gmm_model1_name, image_files, mode='gray', normalize=True)
    process_all_images(data_dir, output_dir_rgb, gmm_model2_name, image_files, mode='rgb', normalize=True)

