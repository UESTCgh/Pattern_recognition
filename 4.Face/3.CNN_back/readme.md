- # 人脸识别优化

  基于 CNN 的人脸识别模型训练与测试

  ## 项目描述

  本项目实现了一个卷积神经网络（CNN）用于人脸识别任务，能够在给定的数据集上训练并测试模型性能。该网络采用了多层卷积和全连接层结构，结合了批归一化（Batch Normalization）和 Dropout 层来防止过拟合。数据集采用了增强技术以解决类别不平衡问题，提升模型的泛化能力。

  ## 文件结构

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

  ## 环境配置

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

  ## 使用说明

  ### 1. 数据准备

  将以下数据文件放置在 `../data/` 文件夹中：

  - `train_data.mat`：包含训练数据，形状为 (样本数, 特征数)。
  - `train_label.csv`：训练数据的标签，-1 表示负类，1 表示正类。
  - `test_data.mat`：包含测试数据，形状为 (样本数, 特征数)。
  - `test_label_manual.mat`：测试数据的标签，-1 表示负类，1 表示正类。

  ### 2. 运行训练和测试脚本

  运行 `main.py` 文件，开始训练和测试流程。

  ```
  python main.py
  ```

  - **设备选择**：如果有 GPU 可用，脚本将自动使用 GPU 进行训练。
  - **超参数调整**：你可以在脚本顶部的 "超参数设置" 部分修改学习率、训练轮次、权重衰减等。

  ### 3. 训练流程

  - 使用 `SMOTE` 对训练数据进行过采样，解决类别不平衡问题。
  - 进行数据标准化处理（StandardScaler）。
  - 支持数据增强（水平镜像翻转和加高斯噪声），可以通过 `apply_data_augmentation` 参数进行开启或关闭。
  - 使用 CNN 模型进行训练，模型结构包含三个卷积层，每层后有批归一化和最大池化，最终通过两层全连接层进行分类。
  - 训练过程中保存了数据增强的示例图 (`Data_face.png`) 以及训练损失曲线 (`loss.png`)。

  ### 4. 测试与评估

  - 训练结束后，模型会自动保存为 `final_trained_model.pth` 文件。
  - 在测试集上评估模型性能，包括准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1 分数等指标。
  - 绘制并保存模型的 ROC 曲线和 PR 曲线 (`ROCPR.png`)。
  - 绘制并保存混淆矩阵 (`Matrix.png`)。
  - 最终分类报告也会打印在终端上。

  ## 重要参数

  - **学习率 (learning_rate)**：0.01
  - **训练轮次 (epochs)**：100
  - **批次大小 (batch_size)**：16
  - **类权重 (class_weights)**：调整为 `[1, 105.0]`，以解决类别不平衡。
  - **L2 正则化 (weight_decay)**：`5e-4`
  - **早停 (early_stop_patience)**：`6`，防止过拟合。
  - **学习率调度器 (scheduler)**：基于损失函数变化动态调整学习率。

  ## 模型结构

  - **卷积层**：3 层卷积，每层后接批归一化、ReLU 激活、最大池化。
  - **全连接层**：2 层全连接层，128 个节点，带 Dropout 层防止过拟合。

  ## 输出结果

  - **模型权重**：`final_trained_model.pth`
  - **测试集性能**：包括准确率、精确率、召回率、F1 分数及混淆矩阵。
  - **图像文件**：包括数据增强示例 (`Data_face.png`)、训练损失曲线 (`loss.png`)、ROC 和 PR 曲线 (`ROCPR.png`)、混淆矩阵 (`Matrix.png`)。
  - **预测结果**：`model_predictions.txt`，记录测试集上的预测结果，标签为 -1 或 1。

  ## 注意事项

  - 需要确保数据集标签中的类别为 0 和 1，因为 CrossEntropyLoss 需要这样的标签格式。
  - 请检查 CUDA 是否可用，并确保驱动和 CUDA 版本与 PyTorch 版本兼容。

  ## 未来改进方向

  - **超参数优化**：可以使用 Grid Search 或 Random Search 进一步优化模型的超参数。
  - **模型改进**：可以尝试更深层的网络或加入残差结构（ResNet）等改进模型性能。
  - **数据增强**：增加更多数据增强方式，如旋转、平移、随机裁剪等，以提升模型泛化能力。