# AI_VI 模式识别大作业
电子科技大学模式识别课程24级大作业，有的目录没有修改 仅供参考哈哈

执行list.py可以更新directory_structure.md中的目录

environment.yml为项目环境文件 使用下面代码创建

```
conda env create -f environment.yml
```

## 项目结构

```
├── 0.实验报告/
│   └── work/
│       └── 大作业3 to student/
├── 1.bayesian/
├── 2.BPSVM/
│   ├── BP/
│   │   ├── cmake-build-debug/
│   │   ├── data/
│   │   └── 处理结果/
│   ├── BP1/
│   │   ├── cmake-build-debug/
│   │   └── data/
│   ├── ROC/
│   │   ├── data/
│   │   └── result/
│   ├── SVM/
│   │   ├── cmake-build-debug/
│   │   ├── data/
│   │   └── result/
│   └── VS/
│       ├── data/
│       └── result/
│           ├── Model_Compare/
│           └── Model_only/
├── 3.EM/
│   ├── 3.1EM/
│   │   └── result/
│   ├── 3.2fishdis/
│   │   ├── data/
│   │   ├── result/
│   │   └── result1/
│   ├── 3.3display/
│   │   ├── combined/
│   │   ├── result/
│   │   └── trained_models/
│   ├── 实验结果汇总/
│   │   ├── 任务1/
│   │   └── 任务2/
│   └── 资料/
├── 4.Face/
│   ├── 1.randomtree/
│   │   ├── data/
│   │   └── src/
│   ├── 2.CNN/
│   │   ├── data/
│   │   ├── history/
│   │   └── src/
│   ├── 3.CNN_back/
│   │   ├── data/
│   │   ├── history/
│   │   └── src/
│   └── 资料/
│       ├── test(Task_2)/
│       └── train(Task_2)/
└── README.assets/

```



## 1.贝叶斯分类器

### 实验目的

1. 通过贝叶斯分类器实验掌握先验概率、似然函数、后验概率等概念。
2. 学习如何使用最大似然估计和最大后验估计来估计模型参数。
3. 掌握贝叶斯决策的基本步骤和应用场景。

### 实验原理与公式

- 贝叶斯公式用于结合先验知识与样本数据来对样本分类。
- 参数估计中，采用了最大似然（MLE）和最大后验（MAP）的方法。
- 实验中通过男女生的体重和身高数据，生成对应的概率分布进行分类。

**公式说明**：

贝叶斯参数估计、最大后验参数估计及贝叶斯决策相关公式见下图：

![贝叶斯参数估计](README.assets/image-20240928005615233.png)

### 数据来源

- 数据来自网络，包括最新的人体身高体重分布数据。
  - [最新国人体质数据 (qq.com)](https://new.qq.com/rain/a/20201223A066XX00)
  - [第五次国民体质监测公报 (sport.gov.cn)](https://www.sport.gov.cn/n315/n329/c24335066/content.html)

### 实验结论

1. 数据呈现较好的正态分布。
2. 使用最大似然与最大后验两种估计方法获得的参数在相应数据集下有相似性。
3. 贝叶斯决策的准确性与先验的选择密切相关。

### **文书修改**

<img src="README.assets/image-20240928005446019.png" alt="image-20240928005446019" style="zoom: 50%;" />

按照实验里的要求 写实验目的

<img src="README.assets/image-20240928005341490.png" alt="image-20240928005341490" style="zoom:80%;" />

[[公式识别 (simpletex.cn)](https://simpletex.cn/ai/latex_ocr)]()

贝叶斯参数估计

<img src="README.assets/image-20240928005615233.png" alt="image-20240928005615233" style="zoom:50%;" />

最大后验参数估计

<img src="README.assets/image-20240928005710975.png" alt="image-20240928005710975" style="zoom:80%;" />

贝叶斯决策

<img src="README.assets/image-20240928005847873.png" alt="image-20240928005847873" style="zoom: 50%;" />

### <img src="README.assets/image-20240928010013172.png" alt="image-20240928010013172" style="zoom:50%;" />

三点结论：

1、男女生的直方图分布近似符合正态

2、最大似然估计的参数，贝叶斯估计假设先验方差下的参数估计

3、贝叶斯决策，选取什么样的先验分布，给一个测试样本分类的结果

## 2.BP&SVM

### 项目概述

BPSVM（Batch Processing Support Vector Machine）是一个基于支持向量机（SVM）的分类算法实现，旨在处理大规模数据集。该项目利用批处理的方法，提高 SVM 在处理高维数据时的效率和准确性。

### 主要功能

- **高效分类**：使用 SVM 进行数据分类，适用于线性和非线性可分的数据。
- **批处理支持**：优化的批处理算法，减少内存消耗，提高运行效率。
- **模型评估**：提供多种评估指标（如准确率、召回率、F1 分数等）用于验证模型的性能。
- **可视化工具**：通过可视化方法展示分类结果和决策边界。

### 环境要求

### 项目概述

BP&SVM（Batch Processing Support Vector Machine）是一个基于 SVM 的分类算法实现，旨在处理大规模数据集。该部分的实验目标是探索不同分类器的效果，特别是在高维空间中支持向量机的性能。

### 主要功能

- **高效分类**：使用 SVM 进行线性和非线性数据分类。
- **批处理支持**：优化的批处理算法，提高内存利用率。
- **可视化决策边界**：通过图形展示 SVM 分类结果。

### 环境要求

确保您已安装以下库：

- `numpy`、`pandas`、`scikit-learn`、`matplotlib`

使用以下命令进行安装：

```
bash


复制代码
pip install numpy pandas scikit-learn matplotlib
```

### 使用说明

1. 数据集准备为 CSV 格式。
2. 加载数据并进行预处理，创建 SVM 实例，训练并评估模型。

## 3.EM

根据您上传的代码文件，我将为该项目编写一份完整的 `README.md` 文档。以下是建议的内容：

```markdown
# GMM 图像处理项目

本项目使用高斯混合模型（GMM）对图像进行聚类分析，支持灰度和 RGB 图像的处理。主要功能包括数据加载、模型训练、掩码处理及可视化结果的生成。

## 项目结构
3.EM/3.2fishdis
├── data/                          # 输入数据文件夹
│   ├── array_sample.mat           # 输入数据文件，包含图像数据
│   ├── Mask.mat                   # 掩码文件，用于提取特定区域
│   ├── 309.bmp                    # 原始 BMP 图像文件
├── result/                        # 输出结果文件夹
│   ├── gray/                      # 灰度图像的聚类结果
│   ├── rgb/                       # RGB 图像的聚类结果
│   ├── combined/                  # 合并各步骤结果的图像
│   └── gmm_models/                # 保存的 GMM 模型文件
└── main.py                        # 主程序文件

```

### 环境要求

需要安装以下库：

- `numpy`、`matplotlib`、`scikit-learn`、`joblib`、`scipy`、`Pillow`、`scikit-image`

安装命令如下：

```
pip install numpy matplotlib scikit-learn joblib scipy Pillow scikit-image
```

### 使用说明

1. 将数据文件放入 `data/` 目录中。

2. 运行 `main.py` 文件进行处理，执行以下命令：

   ```bash
   python main.py
   ```

3. 结果文件将会输出到 `result/` 目录中，包括灰度和 RGB 图像的聚类结果、以及各个步骤的合并图像。

### 功能描述

- **GrabCut 提取**：首先对输入图像应用 GrabCut 算法提取鱼类区域。
- **GMM 聚类**：利用 GMM 模型进行聚类分析，支持灰度和 RGB 图像。
- **多阶段可视化**：保存分割过程的每个步骤，并最终合并为 2x3 格式的图像，保存在 `combined` 文件夹中。

### 示例输出

- 在 `result/` 文件夹中，保存了每个处理步骤的图像。
- 在 `result/combined` 中，保存了 2x3 格式的每个图像处理过程的合成图，便于观察整个处理流程。

### 实验结果汇总

在 `实验结果汇总/` 目录下，我们保存了所有实验的最终结果，包括不同任务的结果：

- **任务1**：初始 GMM 聚类分析的结果，包括灰度和 RGB 图像处理结果。
- **任务2**：基于鱼类分割的 GMM 结果，并保存了 GrabCut 处理后的图像以及 GMM 聚类的合并结果。

### 实验结论

1. 对不同模式（灰度和 RGB）的聚类效果进行了对比，展示了高斯混合模型在图像分割中的应用。

2. 实验结果显示不同阶段的图片处理效果，帮助深入理解从数据预处理到最终图像分割的过程。

3. 将鱼类区域提取与 GMM 聚类相结合，形成了较为完整的图像分析流程。

   

## 4.人脸识别

基于 CNN 的人脸识别模型训练与测试

### 项目描述

本项目实现了一个卷积神经网络（CNN）用于人脸识别任务，能够在给定的数据集上训练并测试模型性能。该网络采用了多层卷积和全连接层结构，结合了批归一化（Batch Normalization）和 Dropout 层来防止过拟合。数据集采用了增强技术以解决类别不平衡问题，提升模型的泛化能力。

### 文件结构

- `main.py`：主程序文件，包括数据加载、预处理、模型定义、训练、测试等步骤。
- `train_data.mat`：包含训练数据的 `.mat` 文件。
- `train_label.csv`：训练数据的标签文件，格式为 `.csv`。
- `test_data.mat`：包含测试数据的 `.mat` 文件。
- `test_label_manual.mat`：包含测试标签的 `.mat` 文件。
- `final_trained_model.pth`：最终保存的训练模型权重文件。
- `Data_face.png`：数据增强过程中样本的可视化图像。
- `loss.png`：训练损失曲线的图像。
- `ROCPR.png`：模型的 ROC 和 PR 曲线图像。
- `Matrix.png`：模型的混淆矩阵图像。
- `model_predictions.txt`：模型在测试集上的预测结果。

### 环境配置

- **操作系统**：Windows / Linux / macOS
- **Python 版本**：Python 3.7 及以上
- **依赖库**：
  - `numpy`
  - `scipy`
  - `pandas`
  - `torch` (PyTorch)
  - `sklearn` (scikit-learn)
  - `imblearn` (用于数据增强的 SMOTE)
  - `torchvision`
  - `matplotlib`（可选，用于可视化）

可以通过以下命令安装依赖：

```
pip install numpy scipy pandas torch scikit-learn imbalanced-learn torchvision matplotlib
```

### 使用说明

#### 1. 数据准备

将以下数据文件放置在 `../data/` 文件夹中：

- `train_data.mat`：包含训练数据，形状为 (样本数, 特征数)。
- `train_label.csv`：训练数据的标签，-1 表示负类，1 表示正类。
- `test_data.mat`：包含测试数据，形状为 (样本数, 特征数)。
- `test_label_manual.mat`：测试数据的标签，-1 表示负类，1 表示正类。

#### 2. 运行训练和测试脚本

运行 `main.py` 文件，开始训练和测试流程。

```
python main.py
```

- **设备选择**：如果有 GPU 可用，脚本将自动使用 GPU 进行训练。
- **超参数调整**：你可以在脚本顶部的 "超参数设置" 部分修改学习率、训练轮次、权重衰减等。

#### 3. 训练流程

- 使用 `SMOTE` 对训练数据进行过采样，解决类别不平衡问题。
- 进行数据标准化处理（StandardScaler）。
- 支持数据增强（水平镜像翻转和加高斯噪声），可以通过 `apply_data_augmentation` 参数进行开启或关闭。
- 使用 CNN 模型进行训练，模型结构包含三个卷积层，每层后有批归一化和最大池化，最终通过两层全连接层进行分类。
- 训练过程中保存了数据增强的示例图 (`Data_face.png`) 以及训练损失曲线 (`loss.png`)。

#### 4. 测试与评估

- 训练结束后，模型会自动保存为 `final_trained_model.pth` 文件。
- 在测试集上评估模型性能，包括准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1 分数等指标。
- 绘制并保存模型的 ROC 曲线和 PR 曲线 (`ROCPR.png`)。
- 绘制并保存混淆矩阵 (`Matrix.png`)。
- 最终分类报告也会打印在终端上。

### 重要参数

- **学习率 (learning_rate)**：0.01
- **训练轮次 (epochs)**：100
- **批次大小 (batch_size)**：16
- **类权重 (class_weights)**：调整为 `[1, 105.0]`，以解决类别不平衡。
- **L2 正则化 (weight_decay)**：`5e-4`
- **早停 (early_stop_patience)**：`6`，防止过拟合。
- **学习率调度器 (scheduler)**：基于损失函数变化动态调整学习率。

### 模型结构

- **卷积层**：3 层卷积，每层后接批归一化、ReLU 激活、最大池化。
- **全连接层**：2 层全连接层，128 个节点，带 Dropout 层防止过拟合。

### 输出结果

- **模型权重**：`final_trained_model.pth`
- **测试集性能**：包括准确率、精确率、召回率、F1 分数及混淆矩阵。
- **图像文件**：包括数据增强示例 (`Data_face.png`)、训练损失曲线 (`loss.png`)、ROC 和 PR 曲线 (`ROCPR.png`)、混淆矩阵 (`Matrix.png`)。
- **预测结果**：`model_predictions.txt`，记录测试集上的预测结果，标签为 -1 或 1。

### 注意事项

- 需要确保数据集标签中的类别为 0 和 1，因为 CrossEntropyLoss 需要这样的标签格式。
- 请检查 CUDA 是否可用，并确保驱动和 CUDA 版本与 PyTorch 版本兼容。

### 未来改进方向

- ### **超参数优化**：可以使用 Grid Search 或 Random Search 进一步优化模型的超参数。

- **模型改进**：可以尝试更深层的网络或加入残差结构（ResNet）等改进模型性能。

- **数据增强**：增加更多数据增强方式，如旋转、平移、随机裁剪等，以提升模型泛化能力。

## 许可证

本项目遵循 MIT 许可证，详情请查看 LICENSE 文件。
