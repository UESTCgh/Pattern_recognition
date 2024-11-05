import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from sklearn.mixture import GaussianMixture
import joblib
import os
import scipy.io as sio
import warnings
from sklearn.exceptions import ConvergenceWarning
from PIL import Image
from skimage import measure

# 设置文件目录
BASE_DIR = "E:/GitHub/AI_VI/3.EM/3.1EM"
DATA_FILE_NAME = "array_sample.mat"
RESULT_DIR = os.path.join(BASE_DIR, "result")
MASK_FILE_NAME = "Mask.mat"
BMP_FILE_NAME = "309.bmp"
MASK_PNG_PATH = os.path.join(RESULT_DIR, "masked_image.png")
MODEL1_PATH = os.path.join(RESULT_DIR, "gmm_final_model.pkl")
MODEL2_PATH = os.path.join(RESULT_DIR, "gmm_rgb_final_model.pkl")
output1_path = os.path.join(RESULT_DIR, "all_rgb.jpg")
output2_path = os.path.join(RESULT_DIR, "all_gray.jpg")
output3_path = os.path.join(RESULT_DIR, "only_rgb.jpg")
output4_path = os.path.join(RESULT_DIR, "only_gray.jpg")

def load_data(file_path):
    """
    加载数据文件并提取需要的数据。

    参数:
        file_path: str, 数据文件的路径

    返回:
        gray_data: ndarray, 一维数据（样本数量, 1）
        rgb_data: ndarray, RGB数据（三维数组：样本数量 x 3）
    """
    data = sio.loadmat(file_path)  # 读取MAT文件
    gray_data = data['array_sample'][:, 0].reshape(-1, 1)  # 提取灰度数据
    rgb_data = data['array_sample'][:, 1:4]  # 提取RGB数据
    return gray_data, rgb_data

def gmm_em_training_save_last(data, n_components=2, max_iter=100, tol=1e-4, init_params='kmeans', save_path=RESULT_DIR):
    """
    训练EM算法的GMM模型，并保存最后一次迭代的结果，包括模型参数、可视化图片和动画GIF。

    参数:
        data: ndarray, 一维数据（样本数量, 1）
        n_components: int, 高斯混合模型的成分数量
        max_iter: int, 最大迭代次数
        tol: float, 收敛阈值
        init_params: str, 初始化参数的方法 ('kmeans' 或 'random')
        save_path: str, 保存模型的文件夹路径
    """
    if not os.path.exists(save_path):  # 创建保存模型的目录
        os.makedirs(save_path)

    # 初始化GMM模型
    gmm = GaussianMixture(n_components=n_components, max_iter=1, tol=tol, init_params=init_params, warm_start=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    frames = []
    prev_means = None

    def update(frame):
        """ 更新图形的函数 """
        ax.clear()
        means, covariances = frames[frame]  # 获取当前帧的均值和方差
        ax.hist(data, bins=30, density=True, alpha=0.5, label="Data Histogram")

        for i, (mean, covariance) in enumerate(zip(means, covariances)):
            ax.axvline(mean, color=f"C{i}", linestyle='--', label=f"Component {i + 1} Mean")
            ax.text(mean, 0.1, f"Mean: {mean:.2f}\nStd: {np.sqrt(covariance):.2f}", color=f"C{i}", fontsize=9, ha='center')

        ax.set_title(f"GMM Clustering - Iteration {frame + 1}")
        ax.set_xlabel("Gray Level")
        ax.set_ylabel("Density")
        ax.legend()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for iteration in range(max_iter):
            gmm.fit(data)
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            print(f"当前迭代 {iteration + 1} 后的均值: {means}, 方差: {covariances}")

            # 提前停止条件 - 如果均值不再显著变化，认为已收敛
            if prev_means is not None and np.allclose(prev_means, means, atol=1e-3):
                print(f"模型在迭代 {iteration + 1} 时收敛，提前停止训练。")
                break
            prev_means = means

            # 保存当前均值和方差
            frames.append((means, covariances))

            # 更新图形并保存当前帧
            update(iteration)
            fig.canvas.draw()
            plt.pause(0.1)

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(frames), repeat=False)
    gif_filename = os.path.join(save_path, "gmm_training_process.gif")
    ani.save(gif_filename, writer=PillowWriter(fps=4))
    print(f"动画已保存: {gif_filename}")

    # 保存最终模型
    model_filename = os.path.join(save_path, "gmm_final_model.pkl")
    joblib.dump(gmm, model_filename)
    print(f"最终模型参数已保存: {model_filename}")

    # 绘制并保存最终迭代的可视化结果
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(data, bins=30, density=True, alpha=0.5, label="Data Histogram")
    for i, (mean, covariance) in enumerate(zip(means, covariances)):
        ax.axvline(mean, color=f"C{i}", linestyle='--', label=f"Component {i + 1} Mean")
        ax.text(mean, 0.1, f"Mean: {mean:.2f}\nStd: {np.sqrt(covariance):.2f}", color=f"C{i}", fontsize=9, ha='center')
    ax.set_title("GMM Clustering - Final Iteration")
    ax.set_xlabel("Gray Level")
    ax.set_ylabel("Density")
    ax.legend()

    final_image_filename = os.path.join(save_path, "gmm_final_result.png")
    plt.savefig(final_image_filename)
    print(f"最终可视化图片已保存: {final_image_filename}")
    plt.close(fig)

def load_and_display_mask(mask_file_path, bmp_file_path):
    """
    读取MAT文件中的掩码数据并显示BMP图像中掩码为1的部分。

    参数:
        mask_file_path: str, 掩码MAT文件路径
        bmp_file_path: str, BMP图像文件路径
    """
    # 加载掩码数据
    mask_data = sio.loadmat(mask_file_path)['Mask']
    # 加载BMP图像
    bmp_image = Image.open(bmp_file_path)
    bmp_array = np.array(bmp_image)

    # 仅保留掩码为1的部分
    masked_image = np.zeros_like(bmp_array)
    masked_image[mask_data == 1] = bmp_array[mask_data == 1]

    # 保存掩码为1的部分的图像
    masked_image_filename = MASK_PNG_PATH
    Image.fromarray(masked_image).save(masked_image_filename)
    print(f"掩码部分的图像已保存: {masked_image_filename}")

    # 显示结果
    plt.figure(figsize=(10, 8))
    plt.imshow(masked_image, cmap='gray')
    plt.title("Masked Image")
    plt.axis('off')
    plt.show()

def display_masked_data_statistics(mask_file_path, bmp_file_path):
    """
    读取掩码数据并计算BMP图像中掩码为1的部分的统计信息。

    参数:
        mask_file_path: str, 掩码MAT文件路径
        bmp_file_path: str, BMP图像文件路径
    """
    # 加载掩码数据
    mask_data = sio.loadmat(mask_file_path)['Mask']
    # 加载BMP图像
    bmp_image = Image.open(bmp_file_path)
    bmp_array = np.array(bmp_image)

    # 提取掩码为1的部分
    masked_data = bmp_array[mask_data == 1]

    # 计算统计信息
    mean_value = np.mean(masked_data)
    std_value = np.std(masked_data)
    max_value = np.max(masked_data)
    min_value = np.min(masked_data)

    # 打印统计信息
    print(f"掩码部分的统计信息:\n均值: {mean_value:.2f}\n标准差: {std_value:.2f}\n最大值: {max_value}\n最小值: {min_value}")


def gmm_rgb_clustering(data, n_components=3, max_iter=100, tol=1e-4, init_params='kmeans', save_path=RESULT_DIR):
    """
    训练EM算法的GMM模型进行RGB数据聚类，并保存最后一次迭代的结果，包括模型参数、动画GIF。

    参数:
        data: ndarray, RGB数据（三维数组：样本数量 x 3）
        n_components: int, 高斯混合模型的成分数量
        max_iter: int, 最大迭代次数
        tol: float, 收敛阈值
        init_params: str, 初始化参数的方法 ('kmeans' 或 'random')
        save_path: str, 保存模型的文件夹路径
    """
    # 创建保存模型的目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 初始化GMM模型
    gmm = GaussianMixture(n_components=n_components, max_iter=1, tol=tol, init_params=init_params, warm_start=True)

    # 初始化3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    frames = []
    prev_means = None

    # 动画更新函数
    def update(frame):
        ax.clear()  # 清空图形
        means, labels = frames[frame]  # 获取当前帧的均值和标签
        colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 预定义颜色用于不同组件

        for i in range(n_components):
            cluster_data = data[labels == i]  # 提取当前组件的数据
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2],
                       color=colors[i % len(colors)], label=f'Component {i + 1}', alpha=0.6, edgecolors='w',
                       linewidth=0.5)

        ax.set_title(f"GMM Clustering - Iteration {frame + 1}", fontsize=15)
        ax.set_xlabel("Red Channel", fontsize=12)
        ax.set_ylabel("Green Channel", fontsize=12)
        ax.set_zlabel("Blue Channel", fontsize=12)
        ax.legend()
        ax.grid(True)

    # 训练模型并实时更新动画
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for iteration in range(max_iter):
            gmm.fit(data)  # 训练GMM模型
            labels = gmm.predict(data)  # 预测每个数据点的标签
            means = gmm.means_  # 获取均值
            covariances = gmm.covariances_  # 获取协方差
            print(f"当前迭代 {iteration + 1} 后的均值:\n{means}\n协方差矩阵如下：\n")
            for i in range(n_components):
                print(f"Component {i + 1} 协方差矩阵如下：\n{covariances[i]}\n")

            # 提前停止条件 - 如果均值不再显著变化，认为已收敛
            if prev_means is not None and np.allclose(prev_means, means, atol=1e-3):
                print(f"模型在迭代 {iteration + 1} 时收敛，提前停止训练。")
                break
            prev_means = means

            # 保存当前均值和标签
            frames.append((means, labels))

            # 更新图形并保存当前帧
            update(iteration)
            fig.canvas.draw()  # 更新图形
            plt.pause(0.1)  # 暂停以更新图形

    # 检查frames是否为空
    if not frames:
        print("没有生成任何帧，可能是因为模型在第一次迭代就收敛。")
        return

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(frames), repeat=False)
    gif_filename = os.path.join(save_path, "gmm_rgb_training_process.gif")
    ani.save(gif_filename, writer=PillowWriter(fps=4))
    print(f"RGB训练的动画已保存: {gif_filename}")

    # 保存最后一次迭代的模型
    model_filename = os.path.join(save_path, "gmm_rgb_final_model.pkl")
    joblib.dump(gmm, model_filename)
    print(f"最终模型参数已保存: {model_filename}")

    # 打印最终的均值和协方差矩阵
    final_means = gmm.means_
    final_covariances = gmm.covariances_
    print(f"最终均值:\n{final_means}\n最终协方差矩阵如下：\n")
    for i in range(n_components):
        print(f"Component {i + 1} 最终协方差矩阵如下：\n{final_covariances[i]}\n")

    # 绘制并保存最终的3D聚类结果
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(n_components):
        cluster_data = data[labels == i]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2],
                   color=colors[i % len(colors)], label=f'Component {i + 1}', alpha=0.6, edgecolors='w', linewidth=0.5)
    ax.set_title("Final GMM Clustering Result", fontsize=15)
    ax.set_xlabel("Red Channel", fontsize=12)
    ax.set_ylabel("Green Channel", fontsize=12)
    ax.set_zlabel("Blue Channel", fontsize=12)
    ax.legend()
    ax.grid(True)

    final_image_filename = os.path.join(save_path, "gmm_rgb_final_result.png")
    plt.savefig(final_image_filename)
    print(f"最终可视化图片已保存: {final_image_filename}")
    plt.show()

def apply_gmm_to_image(gmm_model_path, image_path, output_path, mode, normalize=True):
    """
    将 GMM 模型加载后应用到一张图片上并保存聚类结果，支持灰度和 RGB 图像。

    参数:
    gmm_model_path: str - GMM 模型的路径（.pkl 文件）
    image_path: str - 输入图片的路径
    output_path: str - 输出聚类后图片的路径
    mode: str - 图像模式，支持 'gray'（灰度图）或 'rgb'（RGB 图）
    normalize: bool - 是否对图像数据进行归一化
    """
    # 1.加载 GMM 模型 图片
    gmm = joblib.load(gmm_model_path)
    image = Image.open(image_path)

    # 2.图像分类获取数据
    if mode == 'gray':
        # 转换为灰度图像
        image = image.convert('L')
        image_data = np.array(image)
        reshaped_data = image_data.reshape(-1, 1)  # 展平成一列
    elif mode == 'rgb':
        # 确保图像为 RGB 三通道
        image = image.convert('RGB')
        image_data = np.array(image)
        reshaped_data = image_data.reshape(-1, 3)  # 展平成每行是一个 RGB 颜色值
    else:
        raise ValueError("模式选择错误。请使用 'gray' 或 'rgb'。")

    # 3.归一化处理
    if normalize:
        reshaped_data = reshaped_data / 255.0

    # 4.模型预测
    labels = gmm.predict(reshaped_data)

    # 获取聚类均值作为代表颜色
    cluster_colors = gmm.means_
    if normalize:
        cluster_colors = cluster_colors * 255  # 反归一化
    # 替换每个像素的颜色为其对应的聚类颜色
    clustered_image_data = cluster_colors[labels]

    # 5.恢复图像尺寸
    if mode == 'gray':
        clustered_image_data = clustered_image_data.reshape(image_data.shape).astype(np.uint8)
    else:
        clustered_image_data = clustered_image_data.reshape(image_data.shape[0], image_data.shape[1], 3).astype(np.uint8)

    #6.保存相关结果
    # 保存不加边界线的聚类结果
    no_boundaries_output_path = output_path.replace(".jpg", "_no_boundaries.jpg")
    Image.fromarray(clustered_image_data).save(no_boundaries_output_path)
    print(f"不加边界线的聚类结果已保存: {no_boundaries_output_path}")

    # 绘制边界
    labels_image = labels.reshape(image_data.shape[0], image_data.shape[1])
    clustered_image_with_boundaries = np.copy(clustered_image_data)

    unique_labels = np.unique(labels_image)
    plt.figure(figsize=(10, 8))
    if mode == 'gray':
        plt.imshow(clustered_image_with_boundaries, cmap='gray')
    else:
        plt.imshow(clustered_image_with_boundaries)

    for label in unique_labels:
        label_mask = (labels_image == label).astype(np.float64)
        contours = measure.find_contours(label_mask, 0.5)  # 找到轮廓
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')  # 绘制轮廓

    #保存加边界线的聚类结果
    plt.title("Clustered Image with Boundaries")
    plt.axis('off')
    segmented_image_label_filename = output_path.replace(".jpg", "_with_boundaries.jpg")
    plt.savefig(segmented_image_label_filename, bbox_inches='tight', pad_inches=0.1)
    print(f"加边界线的聚类结果已保存: {segmented_image_label_filename}")
    plt.show()

if __name__ == "__main__":
    # 1. 加载数据文件
    data_file_path = os.path.join(BASE_DIR, DATA_FILE_NAME)
    gray_data, rgb_data = load_data(data_file_path)

    # 2. 训练聚类模型
    # 仅灰度
    gmm_em_training_save_last(data=gray_data, n_components=4, max_iter=100, save_path=RESULT_DIR)
    # RGB数据
    gmm_rgb_clustering(rgb_data, n_components=4, max_iter=100, tol=1e-4, init_params='random', save_path=RESULT_DIR)

    # 3. 加载和显示掩码部分的图像
    mask_file_path = os.path.join(BASE_DIR, MASK_FILE_NAME)
    bmp_file_path = os.path.join(BASE_DIR, BMP_FILE_NAME)
    load_and_display_mask(mask_file_path, bmp_file_path)

    # 显示掩码部分数据的统计信息
    display_masked_data_statistics(mask_file_path, bmp_file_path)

    # 4. 应用模型分割 有无mask rgb和gray
    apply_gmm_to_image(gmm_model_path=MODEL2_PATH, image_path=MASK_PNG_PATH, output_path=output3_path, mode='rgb',normalize=True)
    apply_gmm_to_image(gmm_model_path=MODEL1_PATH, image_path=MASK_PNG_PATH, output_path=output4_path, mode='gray',normalize=True)
    apply_gmm_to_image(gmm_model_path=MODEL2_PATH, image_path=bmp_file_path, output_path=output1_path, mode='rgb',normalize=True)
    apply_gmm_to_image(gmm_model_path=MODEL1_PATH, image_path=bmp_file_path, output_path=output2_path, mode='gray',normalize=True)
