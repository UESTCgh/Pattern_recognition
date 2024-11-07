# GMM 图像处理项目

本项目使用高斯混合模型（GMM）对图像进行聚类分析，支持灰度和 RGB 图像的处理。主要功能包括数据加载、模型训练、掩码处理及可视化结果的生成。

## 项目结构

```
E:/GitHub/AI_VI/3.EM/3.1EM
│
├── array_sample.mat             # 输入数据文件，包含需要处理的图像数据
├── Mask.mat                     # 掩码文件，用于提取特定区域
├── 309.bmp                      # 原始 BMP 图像文件
├── result/                      # 输出结果的文件夹
│   ├── masked_image.png         # 掩码处理后的图像
│   ├── gmm_final_model.pkl      # 最终训练得到的 GMM 模型
│   ├── gmm_rgb_final_model.pkl  # RGB 图像的 GMM 模型
│   ├── gmm_training_process.gif # GMM 模型训练过程的动画
│   ├── gmm_final_result.png     # 最终 GMM 聚类结果的图像
│   ├── all_rgb.jpg              # RGB 图像处理结果
│   ├── all_gray.jpg             # 灰度图像处理结果
│   ├── only_rgb.jpg             # 掩码区域的 RGB 聚类结果
│   └── only_gray.jpg            # 掩码区域的灰度聚类结果
│
└── main.py                      # 主程序文件
```

## 环境要求

请确保您的计算机上安装了以下 Python 库：

- numpy
- matplotlib
- scikit-learn
- joblib
- scipy
- Pillow
- scikit-image

可以使用以下命令安装这些依赖项：

```
pip install numpy matplotlib scikit-learn joblib scipy Pillow scikit-image
```

## 使用说明

### 输入文件

- 将数据文件（`array_sample.mat`、`Mask.mat`、`309.bmp`）放入项目目录中。

### 运行程序

1. 运行主程序文件 `main.py`：

   ```
   python main.py
   ```

2. 该程序将执行以下操作：

   - 加载输入数据和掩码。
   - 训练 GMM 模型，应用于原始图像和掩码区域。
   - 保存处理后的结果至 `result/` 目录。

3. 确保修改代码中的 `BASE_DIR` 变量，以保证文件位置的正确性。

## 功能说明

### 数据加载

- **从 MAT 文件中提取数据**：加载 `array_sample.mat` 和 `Mask.mat`，分别提取图像数据和掩码数据。
- **图像格式支持**：支持灰度和 RGB 两种图像格式。

### GMM 模型训练

- **模型参数**：对灰度和 RGB 数据分别训练 GMM 模型，用户可选择成分数量和初始化方法（`kmeans` 或 `random`）。
- **早停机制**：为了提高模型训练效率，GMM 训练中加入了早停机制，即在模型的对数似然值变化小于设定阈值时，停止训练，避免过度拟合。
- **模型存储**：训练后的 GMM 模型存储为 `.pkl` 文件，便于后续加载和使用。

### 掩码处理

- **应用掩码**：从 `Mask.mat` 加载掩码数据，仅保留掩码区域的像素用于后续处理。
- **统计分析**：计算掩码区域的统计信息（均值、标准差、最大值和最小值），用于描述该区域的特性。

### 图像聚类应用

- **应用 GMM 模型**：将训练好的 GMM 模型应用于原始图像，以生成聚类结果。
- **结果保存**：处理后的聚类结果保存为 PNG 或 JPG 格式，便于进一步分析。

### 模型优化方法

- **早停机制**：GMM 训练过程中加入了早停机制，通过设定对数似然值变化的阈值，确保模型在合适的时间点停止训练，从而减少计算时间，避免过拟合。
- **初始化优化**：支持两种初始化方法（`kmeans` 和 `random`），并可以通过多次运行选择较好的初始值，从而提升模型的聚类效果。
- **正则化**：对协方差矩阵进行正则化处理，防止由于数值不稳定性引发的训练失败，提升模型的收敛性。

### 可视化

- **生成训练过程动画**：在模型训练期间记录过程并生成动画 GIF（`gmm_training_process.gif`），以便可视化聚类模型的收敛过程。
- **最终结果展示**：将最终的聚类结果保存为 `gmm_final_result.png`，并生成包含掩码区域的 RGB 和灰度图像聚类结果。

## 示例输出

运行程序后，您将在 `result/` 目录中找到以下输出文件：

- `**masked_image.png**`：应用掩码后的图像。
- `**gmm_final_model.pkl**`：最终训练得到的 GMM 模型文件。
- `**gmm_training_process.gif**`：GMM 模型训练过程的动画，用于查看训练的动态效果。
- `**gmm_final_result.png**`：聚类后的最终结果图像。
- `**all_rgb.jpg**` 和 `**all_gray.jpg**`：整个图像的 RGB 和灰度聚类结果。
- `**only_rgb.jpg**` 和 `**only_gray.jpg**`：仅掩码区域的 RGB 和灰度聚类结果。

## 注意事项

- **输入文件路径**：在运行程序前，请确保所有输入文件（如 `array_sample.mat`、`Mask.mat`、`309.bmp`）都在正确的路径中，并根据需要修改代码中的路径变量。
- **模型参数调整**：GMM 聚类的成分数量可以根据具体图像内容进行调整，以便获得更优的聚类效果。
- **早停设置**：可以调整早停阈值来平衡训练时间和模型精度。

## 许可证

本项目遵循 MIT 许可证，详情请查看 LICENSE 文件。