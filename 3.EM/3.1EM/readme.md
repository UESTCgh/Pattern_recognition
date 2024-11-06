```markdown
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
│   ├── gmm_rgb_final_model.pkl   # RGB 图像的 GMM 模型
│   ├── gmm_training_process.gif  # GMM 模型训练过程的动画
│   ├── gmm_final_result.png      # 最终 GMM 聚类结果的图像
│   ├── all_rgb.jpg               # RGB 图像处理结果
│   ├── all_gray.jpg              # 灰度图像处理结果
│   ├── only_rgb.jpg              # 掩码区域的 RGB 聚类结果
│   └── only_gray.jpg             # 掩码区域的灰度聚类结果
│
└── main.py                      # 主程序文件

```

## 环境要求

确保您的计算机上安装了以下 Python 库：

- numpy
- matplotlib
- scikit-learn
- joblib
- scipy
- Pillow
- scikit-image

可以使用以下命令安装这些依赖项：

```bash
pip install numpy matplotlib scikit-learn joblib scipy Pillow scikit-image
```

## 使用说明

1. 将数据文件（`array_sample.mat`、`Mask.mat`、`309.bmp`）放入指定的目录中。
2. 运行 `main.py` 文件：
   ```bash
   python main.py
   ```
3. 该程序将加载数据，训练 GMM 模型并进行聚类，处理后的结果将保存在 `result/` 目录中。
4. 修改代码的BASE_DIR，保证文件位置正确

## 功能说明

- **数据加载**：从 MAT 文件中提取灰度和 RGB 数据。
- **GMM 模型训练**：对灰度和 RGB 数据分别训练 GMM 模型，支持指定成分数量和初始化方法（kmeans 或 random）。
- **掩码处理**：加载掩码图像，仅保留掩码区域的像素进行后续处理。
- **统计分析**：计算掩码区域的统计信息（均值、标准差、最大值和最小值）。
- **图像聚类应用**：将训练好的 GMM 模型应用于原始图像，生成聚类结果并保存图像。
- **可视化**：生成训练过程的动画 GIF，以及最终的聚类结果可视化图像。

## 示例输出

运行完成后，您将看到生成的各类结果文件，包括聚类图像、模型文件和动画文件，便于进一步分析和展示。

## 许可证

本项目遵循 MIT 许可证，详情请查看 LICENSE 文件。

```

### 说明
- 此 `README.md` 文档包括项目的基本信息、功能描述、使用步骤和环境要求，适合新用户理解项目。
- 您可以根据项目的具体需求和内容进一步调整和补充此文档。
- 如果您需要更多帮助或有其他问题，请随时告诉我！