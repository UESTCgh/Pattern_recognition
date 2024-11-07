# 自动抠图和分割工具

这是一个用于自动抠图和图像分割的工具，基于 Python 的 Tkinter GUI，结合了 OpenCV 和机器学习模型（如 GMM）。

## 功能

- 选择图片文件进行抠图。
- 使用 GrabCut 算法提取图像中的特定区域（例如鱼的区域）。
- 使用训练好的 GMM（高斯混合模型）进行图像分割，支持灰度模式和 RGB 模式的选择。
- 在界面中显示处理后的图像结果。

## 依赖

确保在运行此程序之前安装以下库：

- Python 3.x
- OpenCV
- NumPy
- Pillow
- scikit-image
- joblib
- matplotlib
- Tkinter（通常与 Python 一起安装）

你可以使用以下命令安装必要的库：

```bash
pip install opencv-python numpy pillow scikit-image joblib matplotlib