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
Model_Path = "E:/GitHub/AI_VI/3.EM/3.1EM/result/gmm_final_model.pkl"


def load_data(file_path):
    """
    加载数据文件并提取需要的数据。

    参数:
        file_path: str, 数据文件的路径

    返回:
        ndarray, 一维数据（样本数量, 1）
    """
    # 读取MAT文件
    data = sio.loadmat(file_path)
    # 提取数据的第一列作为灰度数据
    gray_data = data['array_sample'][:, 0].reshape(-1, 1)
    return gray_data


def gmm_em_training_save_last(data, n_components=2, max_iter=100, tol=1e-4, init_params='kmeans',
                              save_path="gmm_models"):
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
    # 创建保存模型的目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 初始化GMM模型
    gmm = GaussianMixture(n_components=n_components, max_iter=1, tol=tol, init_params=init_params, warm_start=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    frames = []

    prev_means = None

    # 训练模型
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

            # 绘制当前迭代的直方图和均值、方差位置
            ax.clear()
            ax.hist(data, bins=30, density=True, alpha=0.5, label="Data Histogram")
            for i, (mean, covariance) in enumerate(zip(means, covariances)):
                ax.axvline(mean, color=f"C{i}", linestyle='--', label=f"Component {i + 1} Mean (Iter {iteration + 1})")
                ax.text(mean, 0.1, f"Mean: {mean:.2f}\nStd: {np.sqrt(covariance):.2f}", color=f"C{i}", fontsize=9,
                        ha='center')
            ax.set_title(f"GMM Clustering - Iteration {iteration + 1}")
            ax.set_xlabel("Gray Level")
            ax.set_ylabel("Density")
            ax.legend()

            # 保存当前帧
            frames.append(fig.canvas.copy_from_bbox(fig.bbox))
            plt.pause(0.1)

    # 保存动画为GIF（使用PillowWriter代替imagemagick）
    def animate(i):
        fig.canvas.restore_region(frames[i])

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), repeat=False)
    gif_filename = os.path.join(save_path, "gmm_training_process.gif")
    ani.save(gif_filename, writer=PillowWriter(fps=1))
    print(f"动画已保存: {gif_filename}")

    # 保存最后一次迭代的模型
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
    fig.canvas.draw()
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
    masked_image_filename = os.path.join(RESULT_DIR, "masked_image.png")
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
    print(
        f"掩码部分的统计信息:\n均值: {mean_value:.2f}\n标准差: {std_value:.2f}\n最大值: {max_value}\n最小值: {min_value}")


def segment_masked_data(mask_file_path, bmp_file_path, model_path, n_segments):
    """
    对BMP图像中掩码为1的部分进行像素分割，并绘制边框。

    参数:
        mask_file_path: str, 掩码MAT文件路径
        bmp_file_path: str, BMP图像文件路径
        n_segments: int, 分割的类别数量
    该函数考虑了彩色图像的分割处理，并使用标准化和优化后的GMM模型进行分割，同时在分割后的图像上绘制每个分割区域的边框。
    """

    # 加载掩码数据
    mask_data = sio.loadmat(mask_file_path)['Mask']

    # 加载BMP图像并转换为灰度图像
    bmp_image = Image.open(bmp_file_path).convert('L')
    bmp_array = np.array(bmp_image)

    # 提取掩码为1的部分（灰度图像）
    masked_data = bmp_array[mask_data == 1].reshape(-1, 1)

    # 对掩码数据进行标准化处理
    masked_data = (masked_data - np.mean(masked_data)) / np.std(masked_data)

    # 加载已保存的GMM模型
    gmm = joblib.load(model_path)  # 确保加载的模型
    labels = gmm.predict(masked_data)  # 使用加载的模型进行预测

    # 创建分割后的图像
    segmented_image = np.zeros_like(bmp_array, dtype=np.uint8)
    segmented_image[mask_data == 1] = (labels + 1) * (255 // (n_segments + 1))

    # 保存分割后的图像
    segmented_image_filename = os.path.join(RESULT_DIR, "segmented_image.png")
    Image.fromarray(segmented_image).save(segmented_image_filename)
    print(f"分割后的图像已保存: {segmented_image_filename}")

    # 使用不同的颜色表示每个分割区域
    for i in range(n_segments):
        segmented_image[mask_data == 1][labels == i] = (i + 1) * (255 // (n_segments + 1))

    # 显示分割后的结果并绘制边框
    plt.figure(figsize=(10, 8))
    plt.imshow(segmented_image, cmap='gray')

    # 绘制每个分割区域的内部边框
    unique_labels = np.unique(segmented_image[mask_data == 1])
    for label in unique_labels:
        label_mask = (segmented_image == label)
        contours = measure.find_contours(label_mask.astype(np.float64), 0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='red')

    plt.title("Segmented Masked Image with Internal Labels")
    plt.axis('off')
    plt.show()


def main():
    # 加载数据文件
    data_file_path = os.path.join(BASE_DIR, DATA_FILE_NAME)
    gray_data = load_data(data_file_path)
    n = 3

    # 调用函数，训练并保存最后一次的结果
    gmm_em_training_save_last(data=gray_data, n_components =n, max_iter=50, save_path=RESULT_DIR)

    # 加载和显示掩码部分的图像
    mask_file_path = os.path.join(BASE_DIR, MASK_FILE_NAME)
    bmp_file_path = os.path.join(BASE_DIR, BMP_FILE_NAME)
    load_and_display_mask(mask_file_path, bmp_file_path)

    # 调用函数，显示掩码部分数据的统计信息
    display_masked_data_statistics(mask_file_path, bmp_file_path)

    # 对BMP图像中掩码为1的部分进行像素分割
    segment_masked_data(mask_file_path, bmp_file_path,Model_Path,n_segments = n)


if __name__ == "__main__":
    main()
