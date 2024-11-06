# 实验结果汇总 - README

## 项目简介
该项目主要涉及使用 GrabCut 和高斯混合模型 (GMM) 进行图像分割的实验，旨在对图像中的鱼进行提取和聚类。项目中的代码完成了对多张图像的处理，并保存了最终的结果用于分析和验证。以下是各文件夹的结构和对应的内容描述。

## 描述

### 任务1

使用灰度和RGB分别进行GMM聚类，聚类出两个模型对分割和未分割的图片处理

### 任务2

使用 GrabCut 算法对图像进行处理，提取图像中的鱼 保存训练过程

调用任务1中的函数对图像处理出结果

## 项目结构
项目目录结构如下所示：

```
实验结果汇总/
│
├── 任务1/
│   ├── gray/
│   ├── RGB/
│   └── 模型相关/
│
└── 任务2/
    ├── gray/
    ├── rgb/
    ├── combined/
    ├── 311_grabcut.bmp
    ├── 313_grabcut.bmp
    ├── 315_grabcut.bmp
    └── 317_grabcut.bmp
```

### 目录详细说明

#### 任务1
此任务主要包含对图像的初步聚类和模型训练结果。包括灰度图和 RGB 图像的 GMM 聚类结果。

- **gray/**: 存储基于灰度图像进行高斯混合模型 (GMM) 处理后的结果，包括边界线和无边界线的分割结果。
- **RGB/**: 存储基于 RGB 图像进行 GMM 处理后的结果，同样包括带边界和无边界的聚类结果。
- **模型相关/**: 存储了与模型训练相关的数据和结果，如 `gmm_final_model.pkl` 和 `gmm_rgb_final_model.pkl` 两个模型文件以及训练过程中的可视化结果。
  - `gmm_final_model.pkl`：基于灰度数据训练的高斯混合模型。
  - `gmm_rgb_final_model.pkl`：基于 RGB 数据训练的高斯混合模型。
  - `gmm_final_result.png`：灰度模型训练结果的可视化。
  - `gmm_rgb_final_result.png`：RGB 模型训练结果的可视化。
  - `gmm_rgb_training_process.gif`：RGB 模型训练过程的动态图展示。
  - `gmm_training_process.gif`：灰度模型训练过程的动态图展示。

#### 任务2
此任务中，主要使用 GrabCut 方法对图像中的鱼进行提取，并基于提取后的结果进行 GMM 聚类。

- **gray/**: 存储对从 GrabCut 提取的结果进行灰度聚类的结果。
- **rgb/**: 存储对从 GrabCut 提取的结果进行 RGB 聚类的结果。
- **combined/**:存储图像处理的过程
- **311_grabcut.bmp**、**313_grabcut.bmp**、**315_grabcut.bmp**、**317_grabcut.bmp**: 这些图像是使用 GrabCut 方法提取的鱼的区域，背景为黑色。

## 代码说明

### 主要代码模块

1. **extract_fish_with_grabcut(image_path, save_path)**:
   - 使用 GrabCut 算法对图像进行处理，提取图像中的鱼。
   - 处理后的鱼图像保存到指定路径，背景设为黑色。

2. **apply_gmm_to_image(gmm_model_path, image_path, output_path, mode='rgb', normalize=True)**:
   - 加载预训练的 GMM 模型，对输入图像进行聚类处理。
   - 生成两种结果：无边界聚类图和带边界聚类图。
   - 支持灰度 (`gray`) 和 RGB (`rgb`) 两种模式。

3. **process_all_images(input_dir, output_dir, gmm_model_name, image_files, mode='rgb', normalize=True)**:
   - 对指定目录下的多张图像应用 GMM 模型处理，生成并保存处理后的结果。

### 运行主函数

在 `__main__` 函数中，执行了以下任务：

1. **GrabCut 提取鱼的区域**:
   - 使用 `extract_fish_with_grabcut` 函数对 `data` 目录下的图像进行处理。
   - 将结果保存到 `result` 目录下。

2. **对 GrabCut 结果进行 GMM 聚类处理**:
   - 对使用 GrabCut 提取的图像进行灰度和 RGB 模式的 GMM 聚类处理。
   - 结果分别保存在 `result/gray` 和 `result/rgb` 目录下。

3. **对原始图像进行 GMM 聚类处理**:
   - 使用 GMM 模型直接对原始图像进行灰度和 RGB 模式的聚类处理。
   - 结果保存在与 GrabCut 结果相同的目录中，便于对比。

## 文件命名规则
- **原始图像处理结果**:
  - 如 `311.bmp` 使用 GMM 模型处理后，结果命名为 `311_clustered_gray_no_boundaries.jpg` 或 `311_clustered_rgb_no_boundaries.jpg`。
  - 带有边界的结果后缀为 `_with_boundaries`。
  
- **GrabCut 提取结果**:
  - GrabCut 处理后的图像命名为 `{image_name}_grabcut.bmp`。
  - GrabCut 后再进行 GMM 处理的结果会包含聚类模式和边界信息，例如 `311_grabcut_clustered_gray_no_boundaries.jpg`。

## 使用说明
1. **准备数据**: 将要处理的图片放入 `data` 目录中。
2. **训练 GMM 模型**: 在 `模型相关` 目录中已有训练好的 GMM 模型，可直接使用。如果需要重新训练，请参见相关的模型训练代码。
3. **运行脚本**: 直接运行脚本即可，处理后的结果会自动保存到 `result` 目录及其子目录下。
4. **查看结果**: 结果包含经过 GrabCut 处理后的鱼的提取，以及基于 GMM 模型的进一步聚类，方便对比分析不同方法的效果。

## 注意事项
- 如果路径不存在，代码会自动创建相关目录。
- 代码中使用的文件格式主要为 `.bmp` 和 `.jpg`，可根据需要进行更改。
- GMM 模型文件（如 `gmm_final_model.pkl`）需要保存在 `data` 目录下。
