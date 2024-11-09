import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.mixture import GaussianMixture
import matplotlib.animation as animation
import scipy.io
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib  # 确保导入 joblib 模块

import matplotlib.font_manager as fm

# 设置中文字体，防止中文显示乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

# 确保模型保存目录存在
MODEL_SAVE_DIR = "trained_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 定义全局变量用于存储模型路径和图像路径
model_path = ""
current_image_path = ""
current_processed_image_path = ""
dataset_path = ""
selected_mode = "gray"  # 默认为灰度模式
n_components = 2  # 聚类数量，默认2
init_params = 'kmeans'  # 初始化方式，默认使用kmeans
max_iterations = 10  # 最大训练轮次，默认10
COMBINED_DIR = "combined"  # 用于保存合成图的目录
os.makedirs(COMBINED_DIR, exist_ok=True)


# 抠图函数，使用你的详细抠图逻辑
def extract_fish_with_grabcut(image_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    # 形态学操作去除噪声
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

    # 应用 Canny 边缘检测以获取轮廓
    edges = cv2.Canny(opened_mask, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        fish_contour = max(contours, key=cv2.contourArea)
    else:
        print(f"未能找到轮廓: {image_path}")
        return

    # 创建空掩码并绘制最大的轮廓
    contour_mask = np.zeros_like(combined_mask)
    cv2.drawContours(contour_mask, [fish_contour], -1, 255, thickness=cv2.FILLED)

    # 形态学操作填充小的孔洞
    filled_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel)

    # 提取鱼的区域并将背景设置为黑色
    fish_result = cv2.bitwise_and(image_rgb, image_rgb, mask=filled_mask)

    # 保存最终的分割结果
    result_bgr = cv2.cvtColor(fish_result, cv2.COLOR_RGB2BGR)
    save_path = save_path.replace(".bmp", ".png")  # 将保存路径修改为 PNG 格式
    cv2.imwrite(save_path, result_bgr)


# 将 GMM 模型应用到输入图像上并保存聚类结果
def apply_gmm_to_image(gmm_model_path, image_path, output_path, mode='gray', normalize=True):
    gmm = joblib.load(gmm_model_path)
    image = Image.open(image_path)

    # 如果选择灰度模式，将图像转换为灰度
    if mode == 'gray':
        image = image.convert('L')  # 将图像转换为灰度模式
        image_data = np.array(image)
        reshaped_data = image_data.reshape(-1, 1)  # 灰度图形状为 (H*W, 1)
    elif mode == 'rgb':
        image = image.convert('RGB')  # 保持 RGB 模式
        image_data = np.array(image)
        reshaped_data = image_data.reshape(-1, 3)  # RGB 图形状为 (H*W, 3)
    else:
        raise ValueError("模式选择错误。请使用 'gray' 或 'rgb'。")

    # 归一化数据（如果需要）
    if normalize:
        reshaped_data = reshaped_data / 255.0

    # 使用 GMM 进行预测
    try:
        labels = gmm.predict(reshaped_data)
    except ValueError:
        raise ValueError("GMM 期望输入的特征数量与当前模式不匹配。请检查模式设置。")

    # 获取聚类的颜色均值
    cluster_colors = gmm.means_
    if normalize:
        cluster_colors = cluster_colors * 255

    # 根据预测结果替换每个像素为其对应的颜色
    clustered_image_data = cluster_colors[labels]

    # 还原图像的形状并保存
    if mode == 'gray':
        clustered_image_data = clustered_image_data.reshape(image_data.shape).astype(np.uint8)
    else:
        clustered_image_data = clustered_image_data.reshape(image_data.shape[0], image_data.shape[1], 3).astype(
            np.uint8)

    # 保存没有边界线的分割结果
    no_boundaries_output_path = output_path.replace(".png", "_no_boundaries.png")
    Image.fromarray(clustered_image_data).save(no_boundaries_output_path)


# 选择模型路径
def load_model():
    global model_path
    model_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pkl")])
    if model_path:
        messagebox.showinfo("模型加载", f"成功加载模型：{model_path}")


# 选择文件函数
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        global current_image_path
        current_image_path = file_path
        load_image(file_path)


# 加载图像显示在界面上
# 修改 load_image 函数以调整图片大小
def load_image(image_path):
    try:
        image = Image.open(image_path)

        # 调整图像大小，设置为一个较大的尺寸，比如宽度 600 像素，高度适配比例
        image = image.resize((600, int(600 * image.height / image.width)), Image.ANTIALIAS)

        img_tk = ImageTk.PhotoImage(image)
        image_label.config(image=img_tk)
        image_label.image = img_tk
    except Exception as e:
        messagebox.showerror("错误", f"加载图像时出错: {str(e)}")

# 直接颜色范围过滤的抠图算法
def generate_nemo_mask(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    light_orange = (5, 100, 120)
    dark_orange = (20, 255, 255)
    light_white = (35, 0, 160)
    dark_light = (255, 160, 255)

    mask1 = cv2.inRange(hsv_img, light_orange, dark_orange)
    mask2 = cv2.inRange(hsv_img, light_white, dark_light)
    mask = np.array((mask1 + mask2) > 0, dtype=mask1.dtype)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ROI = np.zeros(mask.shape, np.uint8)
    ROI[0:166, 50:200] = 1
    mask = mask * ROI

    return mask


def extract_fish_with_color_filter(image_path, save_path):
    img = cv2.imread(image_path)
    mask = generate_nemo_mask(img)
    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    # 保存处理结果
    cv2.imwrite(save_path, segmented_img)

def confirm_grabcut():
    try:
        if not current_image_path:
            messagebox.showerror("错误", "请先加载图片再执行抠图操作！")
            return

        # 获取算法选择
        if isinstance(algorithm_choice, tk.StringVar):
            algorithm = algorithm_choice.get()  # 如果是 tk.StringVar，使用 .get() 获取值
        else:
            algorithm = algorithm_choice  # 如果是字符串，则直接使用

        if algorithm not in ["GrabCut", "Color Filter"]:
            messagebox.showerror("错误", "无效的抠图算法选择！")
            return

        # 设置保存路径，根据算法选择不同的子文件夹
        result_dir = "result"
        if algorithm == "GrabCut":
            save_dir = os.path.join(result_dir, "GrabCut")
        elif algorithm == "Color Filter":
            save_dir = os.path.join(result_dir, "ColorFilter")

        # 创建保存目录（如果目录不存在则创建）
        os.makedirs(save_dir, exist_ok=True)

        # 设置保存文件路径
        save_path = os.path.join(save_dir, os.path.basename(current_image_path).replace(".bmp", "_processed.png"))

        # 根据选择执行不同的抠图方法
        if algorithm == "GrabCut":
            extract_fish_with_grabcut(current_image_path, save_path)
        elif algorithm == "Color Filter":
            extract_fish_with_color_filter(current_image_path, save_path)

        # 更新显示处理后的图像
        load_image(save_path)
        global current_processed_image_path
        current_processed_image_path = save_path

    except Exception as e:
        messagebox.showerror("错误", f"抠图处理时出错: {str(e)}")



# 处理图像进行分割
def confirm_segmentation():
    if not model_path:
        messagebox.showwarning("模型未加载", "请先加载分割模型，然后再进行分割操作。")
        return

    try:
        if current_processed_image_path and model_path:
            # 获取算法选择并确定保存目录
            if isinstance(algorithm_choice, tk.StringVar):
                algorithm = algorithm_choice.get()
            else:
                algorithm = algorithm_choice

            if algorithm == "GrabCut":
                save_dir = os.path.join("result", "GrabCut")
            elif algorithm == "Color Filter":
                save_dir = os.path.join("result", "ColorFilter")
            else:
                messagebox.showerror("错误", "无效的抠图算法选择！")
                return

            # 确保保存目录存在
            os.makedirs(save_dir, exist_ok=True)

            # 根据选择的模式设置保存路径，添加模式后缀
            mode_suffix = "_rgb" if selected_mode == "rgb" else "_gray"
            output_filename = os.path.basename(current_processed_image_path).replace("_processed", f"_segmented{mode_suffix}")
            output_path = os.path.join(save_dir, output_filename)

            # 执行分割操作
            apply_gmm_to_image(model_path, current_processed_image_path, output_path, mode=selected_mode, normalize=True)

            # 确定无边界的输出路径，并检查其保存是否成功
            no_boundaries_output_path = output_path.replace(".png", f"_no_boundaries.png")

            # 确保文件确实被保存到目标路径
            if not os.path.exists(no_boundaries_output_path):
                messagebox.showerror("保存错误", f"分割结果文件未能成功保存：{no_boundaries_output_path}")
                return

            # 加载无边界的分割结果
            segmented_image = Image.open(no_boundaries_output_path)
            display_result(segmented_image)

    except ValueError:
        messagebox.showerror("错误", "GMM 期望输入的特征数量与当前模式不匹配。请设置模式为正确的类型。")
    except Exception as e:
        messagebox.showerror("错误", f"分割处理时出错: {str(e)}")



# 显示处理后的图像
# 修改 display_result 函数，以 600x600 大小显示分割后的图像
def display_result(segmented_image):
    # 调整图像大小，设置为 600x600 像素

    segmented_image = segmented_image.resize((600, int(600 * segmented_image.height / segmented_image.width)), Image.ANTIALIAS)

    img_tk = ImageTk.PhotoImage(segmented_image)
    image_label.config(image=img_tk)
    image_label.image = img_tk


# 更新模式选择的函数
def update_mode(event):
    global selected_mode
    mode = mode_var.get()
    if mode not in ["gray", "rgb"]:
        messagebox.showerror("模式选择错误", "无效的模式选择！请使用 'gray' 或 'rgb'。")
    else:
        selected_mode = mode


# 从MAT文件加载数据
def select_dataset():
    global dataset_path
    # 如果已经加载了数据集路径，则不再需要重新加载
    if dataset_path:
        messagebox.showinfo("数据集已加载", f"数据集路径已存在：{dataset_path}")
        return

    # 选择数据集路径
    dataset_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    if dataset_path:
        messagebox.showinfo("数据集加载", f"成功加载数据集：{dataset_path}")
    else:
        messagebox.showwarning("数据集加载失败", "未选择数据集路径，请重新选择。")


def load_mat_data(data_type):
    try:
        mat = scipy.io.loadmat(dataset_path)
        if data_type == 'gray':
            return mat['array_sample'][:, 0].reshape(-1, 1)  # 提取灰度数据
        else:
            return mat['array_sample'][:, 1:4]  # 提取RGB数据
    except KeyError as e:
        messagebox.showerror("错误", f"数据集中找不到指定的键: {str(e)}")
        raise
    except Exception as e:
        messagebox.showerror("错误", f"加载数据集时出错: {str(e)}")
        raise

# GMM 训练可视化函数
def visualize_gmm_training(data_type, parent_frame):
    try:
        # 检查是否已经加载数据集路径
        if not dataset_path:
            messagebox.showwarning("未加载数据集", "请先选择数据集路径，再进行训练操作。")
            return

        # 清除绘图区域，只清除绘图部分，保留按钮部分
        if hasattr(parent_frame, 'plot_frame'):
            parent_frame.plot_frame.destroy()

        # 创建新的绘图区域
        plot_frame = ttk.Frame(parent_frame)
        plot_frame.pack(pady=10)
        parent_frame.plot_frame = plot_frame

        # 重新生成新的训练可视化区域
        data = load_mat_data(data_type)
        gmm = GaussianMixture(n_components=n_components, max_iter=1, init_params=init_params, warm_start=True)
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d') if data_type == 'rgb' else fig.add_subplot(111)
        frames = []

        # 开始训练
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            stop_iteration = None  # 用于记录早停迭代次数
            for iteration in range(max_iterations):
                try:
                    gmm.fit(data)
                except ValueError as ve:
                    messagebox.showwarning("训练错误", f"在训练过程中遇到错误：{str(ve)}")
                    return

                means = gmm.means_ if data_type == 'rgb' else gmm.means_.flatten()
                covariances = gmm.covariances_ if data_type == 'rgb' else gmm.covariances_.flatten()
                labels = gmm.predict(data)
                frames.append((means, covariances, labels))

                # 早停条件
                if len(frames) > 1 and np.allclose(frames[-1][0], frames[-2][0], atol=1e-3):
                    stop_iteration = iteration + 1
                    break

        # 画图的更新逻辑
        def update(frame):
            ax.clear()
            if data_type == 'rgb':
                colors = ['r', 'g', 'b', 'c', 'm', 'y']
                for i in range(n_components):
                    cluster_data = data[frames[frame][2] == i]
                    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2],
                               color=colors[i % len(colors)], label=f"成分 {i + 1}",
                               alpha=0.6, edgecolors='w', linewidth=0.5)
                ax.set_xlabel("红色通道")
                ax.set_ylabel("绿色通道")
                ax.set_zlabel("蓝色通道")
                ax.set_title(f"GMM 训练 - 迭代 {frame + 1}")
            else:
                ax.hist(data[:, 0], bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black',
                        label="数据直方图")
                means, covariances, _ = frames[frame]
                for i, mean in enumerate(means):
                    ax.axvline(mean, color=f"C{i}", linestyle='--', linewidth=2, label=f"成分 {i + 1} 均值")
                ax.set_xlabel("数值")
                ax.set_ylabel("密度")
                ax.set_title(f"GMM 训练 - 迭代 {frame + 1}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

        ani = animation.FuncAnimation(fig, update, frames=len(frames), repeat=False)

        # 在 Tkinter 窗口中显示动画
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack()
        ani._start()

        # 在显示动画之后再提示早停信息
        canvas.get_tk_widget().after(int(ani.event_source.interval * len(frames)), lambda: (
            messagebox.showinfo("训练完成", f"模型在迭代 {stop_iteration} 时收敛，提前停止训练。")
            if stop_iteration else None))

        # 自动保存训练好的模型到指定的目录中
        model_name = f"gmm_model_{data_type}_{n_components}_components.pkl"
        model_save_path = os.path.join(MODEL_SAVE_DIR, model_name)
        joblib.dump(gmm, model_save_path)
        messagebox.showinfo("模型保存", f"模型已自动保存到：{model_save_path}")

    except Exception as e:
        messagebox.showerror("训练失败", f"在训练过程中遇到问题：{str(e)}")


# 更新抠图算法选择
def update_algorithm_choice(event):
    global algorithm_choice
    algorithm_choice = algorithm_var.get()

# 创建 GUI
root = tk.Tk()
root.title("自动抠图和分割工具")
root.geometry("900x700")
root.configure(bg='#F0F0F0')

# 在全局定义algorithm_choice变量
algorithm_choice = tk.StringVar(value="GrabCut")  # 默认为GrabCut算法

# 使用 Notebook 控件创建类似浏览器标签的界面
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# 创建分割工具页面
segmentation_frame = ttk.Frame(notebook)
notebook.add(segmentation_frame, text='分割工具')

# 创建分割工具的按钮
button_frame = ttk.Frame(segmentation_frame)
button_frame.pack(pady=20)

style = ttk.Style()
style.configure('TButton', font=('Arial', 12), padding=10)

select_file_btn = ttk.Button(button_frame, text="选择图像文件", command=select_file, style='TButton')
select_file_btn.grid(row=0, column=0, padx=5)

load_model_btn = ttk.Button(button_frame, text="加载分割模型", command=load_model, style='TButton')
load_model_btn.grid(row=0, column=1, padx=5)

confirm_grabcut_btn = ttk.Button(button_frame, text="确认抠图", command=confirm_grabcut, style='TButton')
confirm_grabcut_btn.grid(row=0, column=2, padx=5)

confirm_segmentation_btn = ttk.Button(button_frame, text="确认分割", command=confirm_segmentation, style='TButton')
confirm_segmentation_btn.grid(row=0, column=3, padx=5)

# 使按钮分布更加均匀
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)
button_frame.grid_columnconfigure(2, weight=1)
button_frame.grid_columnconfigure(3, weight=1)

# 创建模式选择下拉菜单
mode_var = tk.StringVar(value="gray")
mode_label = ttk.Label(button_frame, text="选择模式:", font=('Arial', 12))
mode_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

mode_menu = ttk.Combobox(button_frame, textvariable=mode_var, values=["gray", "rgb"], font=('Arial', 12), state="readonly")
mode_menu.grid(row=1, column=1, padx=10, pady=10, columnspan=2, sticky="ew")
mode_menu.bind("<<ComboboxSelected>>", update_mode)

# 创建抠图算法选择下拉菜单
algorithm_var = tk.StringVar(value="GrabCut")
algorithm_label = ttk.Label(button_frame, text="选择抠图算法:", font=('Arial', 12))
algorithm_label.grid(row=1, column=3, padx=10, pady=10, sticky="w")

algorithm_menu = ttk.Combobox(button_frame, textvariable=algorithm_var, values=["GrabCut", "Color Filter"], font=('Arial', 12), state="readonly")
algorithm_menu.grid(row=1, column=4, padx=10, pady=10, columnspan=2, sticky="ew")
algorithm_menu.bind("<<ComboboxSelected>>", update_algorithm_choice)


# 创建用于显示处理后图像的区域
image_frame = ttk.Frame(segmentation_frame)
image_frame.pack(pady=20)

image_label = ttk.Label(image_frame)
image_label.pack()

# 创建 GMM 训练工具页面
gmm_training_frame = ttk.Frame(notebook)
notebook.add(gmm_training_frame, text='GMM 训练工具')

# 创建 GMM 训练的按钮
gmm_button_frame = ttk.Frame(gmm_training_frame)
gmm_button_frame.pack(pady=20)

select_dataset_btn = ttk.Button(gmm_button_frame, text="选择数据集路径", command=select_dataset, style='TButton')
select_dataset_btn.grid(row=0, column=0, padx=5)

n_components_label = ttk.Label(gmm_button_frame, text="聚类个数:", font=('Arial', 12))
n_components_label.grid(row=0, column=1, padx=5)

n_components_entry = ttk.Entry(gmm_button_frame, font=('Arial', 12))
n_components_entry.grid(row=0, column=2, padx=5)
n_components_entry.insert(0, "2")


def update_n_components():
    global n_components
    try:
        n_components = int(n_components_entry.get())
    except ValueError:
        messagebox.showerror("输入错误", "请输入有效的聚类个数")


init_params_label = ttk.Label(gmm_button_frame, text="初始化方式:", font=('Arial', 12))
init_params_label.grid(row=0, column=3, padx=5)

init_params_menu = ttk.Combobox(gmm_button_frame, values=["kmeans", "random"], font=('Arial', 12), state="readonly")
init_params_menu.grid(row=0, column=4, padx=5)
init_params_menu.set("kmeans")


def update_init_params(event=None):
    global init_params
    init_params = init_params_menu.get()

init_params_menu.bind("<<ComboboxSelected>>", update_init_params)


max_iterations_label = ttk.Label(gmm_button_frame, text="最大训练轮次:", font=('Arial', 12))
max_iterations_label.grid(row=1, column=0, padx=5)

max_iterations_entry = ttk.Entry(gmm_button_frame, font=('Arial', 12))
max_iterations_entry.grid(row=1, column=1, padx=5)
max_iterations_entry.insert(0, "10")


def update_max_iterations():
    global max_iterations
    try:
        max_iterations = int(max_iterations_entry.get())
    except ValueError:
        messagebox.showerror("输入错误", "请输入有效的最大训练轮次")

# 更新参数按钮
update_params_btn = ttk.Button(gmm_button_frame, text="更新参数",
                               command=lambda: [
                                   update_n_components(),
                                   update_max_iterations(),
                                   update_init_params(),
                                   messagebox.showinfo("参数更新", "参数更新成功！")  # 更新成功提示框
                               ],
                               style='TButton')
update_params_btn.grid(row=1, column=2, padx=5)


gray_training_btn = ttk.Button(gmm_button_frame, text="灰度数据训练",
                               command=lambda: visualize_gmm_training('gray', gmm_training_frame), style='TButton')
gray_training_btn.grid(row=2, column=0, padx=5)

rgb_training_btn = ttk.Button(gmm_button_frame, text="RGB 数据训练",
                              command=lambda: visualize_gmm_training('rgb', gmm_training_frame), style='TButton')
rgb_training_btn.grid(row=2, column=1, padx=5)

# 运行 Tkinter 主循环
root.mainloop()
