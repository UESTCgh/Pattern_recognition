import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from PIL import Image, ImageEnhance,ImageOps
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import cv2
from sklearn.mixture import GaussianMixture

#图像预处理函数
def ImageProcess(image):
    # 降噪 (使用高斯滤波)
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    # 降低曝光度
    exposure_factor = 0.7  # 调整曝光度的因子，可以根据需要更改
    exposure_adjusted_image = cv2.convertScaleAbs(RGB_image_311, alpha=exposure_factor, beta=0)

    # 拉伸对比度
    contrast_adjusted_image = cv2.convertScaleAbs(exposure_adjusted_image, alpha=1.5, beta=0)  # 调整alpha值以拉伸对比度

    #转换为HSV
    hsv_image = cv2.cvtColor(contrast_adjusted_image, cv2.COLOR_RGB2HSV)

    return hsv_image

#在hsv空间直接使用颜色阈值进行分割
def Hsvsegment(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    #橙色阈值
    light_orange = (5, 160, 120)
    dark_orange = (20, 255, 255)

    #橙色分割mask
    mask_orange = cv2.inRange(hsv_image,light_orange,dark_orange)

    #白色阈值
    light_white = (35, 0, 165)
    dark_white = (255, 120, 255)

    #切割白色部分
    mask_white = cv2.inRange(hsv_image,light_white,dark_white)

    final_mask = np.array((mask_orange + mask_white) > 0, dtype=mask_orange.dtype)  # 模板合并

    result = cv2.bitwise_and(image,image,mask = final_mask)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)   #膨胀腐蚀，消除孤立区间
    ROI = np.zeros(mask.shape, np.uint8)
    ROI[0:150, 50:200] = 1
    mask_result = mask * ROI

    RGB_result = image * cv2.merge([mask_result, mask_result, mask_result])


    return RGB_result,mask_result

#采用kmeans算法完成分割
def Kmeansegment(image):

    # 将图像转换成NumPy数组
    image_array = np.array(image)
    height, width, channels = image_array.shape

    # 将图像像素转换为特征向量
    pixels = image_array.reshape(-1, channels)

    # 选择聚类数
    k = 8

    # 初始化K均值模型
    kmeans = KMeans(n_clusters=k)

    # 拟合模型
    kmeans.fit(pixels)

    # 获取每个像素点的聚类标签
    labels = kmeans.labels_

    # 将每个像素点的颜色替换为其所属聚类的中心颜色
    segmented_image = kmeans.cluster_centers_[labels].reshape(height, width, channels).astype(np.uint8)

    # 获取每个类的像素数量
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 选择第二多和第三多的两个类
    first_label = unique_labels[np.argsort(counts)[-7]]
    second_label = unique_labels[np.argsort(counts)[-8]]

    # 创建掩码，将选定的两个聚类标签设置为255，其余标签设置为0
    mask_label = np.isin(labels, [first_label, second_label]) * 255

    # 将掩码转换为8位灰度图像
    gray_selected_classes_image = mask_label.reshape(height, width).astype(np.uint8)

    kernel = np.ones((8, 8), np.uint8)
    mask = cv2.morphologyEx(gray_selected_classes_image, cv2.MORPH_CLOSE, kernel)  # 膨胀腐蚀，消除孤立区间

    # 切割区域，消除其他孤立区域
    ROI = np.zeros(gray_selected_classes_image.shape, np.uint8)
    ROI[0:150, 50:200] = 1
    mask_result = mask * ROI  # 膨胀后的mask
    mask_median = gray_selected_classes_image * ROI

    return segmented_image,mask_median,mask_result

#采用EM算法完成分割
def EMGMMsegment(image):
    # 将图像转换成NumPy数组
    image_array = np.array(image)

    # 获取图像的形状
    height, width, channels = image_array.shape

    # 将图像像素转换为特征向量
    pixels = image_array.reshape(-1, channels)
    #选择分量数量 (聚类数)
    n_components = 8

    #初始化GMM模型
    gmm = GaussianMixture(n_components=n_components)

    #拟合模型
    gmm.fit(pixels)

    #获取每个像素点的聚类标签
    labels = gmm.predict(pixels)

    #将每个像素点的颜色替换为其所属聚类的中心颜色
    segmented_image = gmm.means_[labels].reshape(height, width, channels).astype(np.uint8)

    # 获取每个类的像素数量
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 选择第二多和第三多的两个类
    first_label = unique_labels[np.argsort(counts)[-7]]
    second_label = unique_labels[np.argsort(counts)[-8]]

    # 创建掩码，将选定的两个聚类标签设置为255，其余标签设置为0
    mask_label = np.isin(labels, [first_label, second_label]) * 255

    # 将掩码转换为8位灰度图像
    gray_selected_classes_image = mask_label.reshape(height, width).astype(np.uint8)

    kernel = np.ones((8, 8), np.uint8)
    mask = cv2.morphologyEx(gray_selected_classes_image, cv2.MORPH_CLOSE, kernel)  # 膨胀腐蚀，消除孤立区间

    # 切割区域，消除其他孤立区域
    ROI = np.zeros(gray_selected_classes_image.shape, np.uint8)
    ROI[0:150, 50:200] = 1
    mask_result = mask * ROI  # 膨胀后的mask
    mask_median = gray_selected_classes_image * ROI

    return segmented_image, mask_median, mask_result


#显示RGB分量分布情况
def RGBDistribution(image):
    r, g, b = cv2.split(image)  #切割RGB图像
    rows, cols, d = image.shape #获取图像形状
    pixel_colors = (image.reshape(rows * cols, 3) / 255).tolist()  #像素点归一化

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(r.flatten(), g.flatten(), b.flatten(),
                 facecolors=pixel_colors, marker='.')
    axis.set_xlabel('R')
    axis.set_ylabel('G')
    axis.set_zlabel('B')
    plt.show()

#显示HSV空间的分布
def HSVdistribution(image):
    hsv_nemo = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_nemo)
    rows, cols, d = image.shape
    pixel_colors = (image.reshape(rows * cols, 3) / 255).tolist()

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(),
                 facecolors=pixel_colors, marker='.')
    axis.set_xlabel('Hue')
    axis.set_ylabel('Saturation')
    axis.set_zlabel('Value')
    plt.show()



############################读取图片并完成图片与处理#############################
image_311 = cv2.imread('311.bmp')
RGB_image_311 = cv2.cvtColor(image_311,cv2.COLOR_BGR2RGB)
HSV_image_311 = ImageProcess(RGB_image_311)
image_313 = cv2.imread('313.bmp')
RGB_image_313 = cv2.cvtColor(image_313,cv2.COLOR_BGR2RGB)
HSV_image_313 = ImageProcess(RGB_image_313)
image_315 = cv2.imread('315.bmp')
RGB_image_315 = cv2.cvtColor(image_315,cv2.COLOR_BGR2RGB)
HSV_image_315 = ImageProcess(RGB_image_315)
image_317 = cv2.imread('317.bmp')
RGB_image_317 = cv2.cvtColor(image_317,cv2.COLOR_BGR2RGB)
HSV_image_317 = ImageProcess(RGB_image_317)
##############################################################################


########################k-means聚类#############################################

#311结果
segmented_image_311 , mask_median_311 , mask_result_311 = Kmeansegment(HSV_image_311)
plt.figure()
plt.subplot(2,2,1),plt.imshow(RGB_image_311),plt.axis('off'), plt.title('kmeans  311')
plt.subplot(2,2,2),plt.imshow(segmented_image_311),plt.axis('off'), plt.title('k-means-segment')
plt.subplot(2,2,3),plt.imshow(mask_median_311,cmap = 'gray'),plt.axis('off'), plt.title('MASK-median')
plt.subplot(2,2,4),plt.imshow(mask_result_311,cmap = 'gray'),plt.axis('off'), plt.title('MASK-result')
plt.show()

#313结果
segmented_image_313 , mask_median_313 , mask_result_313 = Kmeansegment(HSV_image_313)
plt.figure()
plt.subplot(2,2,1),plt.imshow(RGB_image_313),plt.axis('off'), plt.title('kmeans  313')
plt.subplot(2,2,2),plt.imshow(segmented_image_313),plt.axis('off'), plt.title('k-means-segment')
plt.subplot(2,2,3),plt.imshow(mask_median_313,cmap = 'gray'),plt.axis('off'), plt.title('MASK-median')
plt.subplot(2,2,4),plt.imshow(mask_result_313,cmap = 'gray'),plt.axis('off'), plt.title('MASK-result')
plt.show()

#315结果
segmented_image_315 , mask_median_315 , mask_result_315 = Kmeansegment(HSV_image_315)
plt.figure()
plt.subplot(2,2,1),plt.imshow(RGB_image_315),plt.axis('off'), plt.title('kmeans  315')
plt.subplot(2,2,2),plt.imshow(segmented_image_315),plt.axis('off'), plt.title('k-means-segment')
plt.subplot(2,2,3),plt.imshow(mask_median_315,cmap = 'gray'),plt.axis('off'), plt.title('MASK-median')
plt.subplot(2,2,4),plt.imshow(mask_result_315,cmap = 'gray'),plt.axis('off'), plt.title('MASK-result')
plt.show()

#317结果
segmented_image_317 , mask_median_317 , mask_result_317 = Kmeansegment(HSV_image_317)
plt.figure()
plt.subplot(2,2,1),plt.imshow(RGB_image_317),plt.axis('off'), plt.title('kmeans  317')
plt.subplot(2,2,2),plt.imshow(segmented_image_317),plt.axis('off'), plt.title('k-means-segment')
plt.subplot(2,2,3),plt.imshow(mask_median_317,cmap = 'gray'),plt.axis('off'), plt.title('MASK-median')
plt.subplot(2,2,4),plt.imshow(mask_result_317,cmap = 'gray'),plt.axis('off'), plt.title('MASK-result')
plt.show()


###############################################################################


#########################EM算法，GMM##############################################################

#311结果
segmented_image_311 , mask_median_311 , mask_result_311 = EMGMMsegment(HSV_image_311)
plt.figure()
plt.subplot(2,2,1),plt.imshow(RGB_image_311),plt.axis('off'), plt.title('EM  311')
plt.subplot(2,2,2),plt.imshow(segmented_image_311),plt.axis('off'), plt.title('EM-segment')
plt.subplot(2,2,3),plt.imshow(mask_median_311,cmap = 'gray'),plt.axis('off'), plt.title('MASK-median')
plt.subplot(2,2,4),plt.imshow(mask_result_311,cmap = 'gray'),plt.axis('off'), plt.title('MASK-result')
plt.show()

#313结果
segmented_image_313 , mask_median_313 , mask_result_313 = EMGMMsegment(HSV_image_313)
plt.figure()
plt.subplot(2,2,1),plt.imshow(RGB_image_313),plt.axis('off'), plt.title('EM  313')
plt.subplot(2,2,2),plt.imshow(segmented_image_313),plt.axis('off'), plt.title('EM-segment')
plt.subplot(2,2,3),plt.imshow(mask_median_313,cmap = 'gray'),plt.axis('off'), plt.title('MASK-median')
plt.subplot(2,2,4),plt.imshow(mask_result_313,cmap = 'gray'),plt.axis('off'), plt.title('MASK-result')
plt.show()

#315结果
segmented_image_315 , mask_median_315 , mask_result_315 = EMGMMsegment(HSV_image_315)
plt.figure()
plt.subplot(2,2,1),plt.imshow(RGB_image_315),plt.axis('off'), plt.title('EM  315')
plt.subplot(2,2,2),plt.imshow(segmented_image_315),plt.axis('off'), plt.title('EM-segment')
plt.subplot(2,2,3),plt.imshow(mask_median_315,cmap = 'gray'),plt.axis('off'), plt.title('MASK-median')
plt.subplot(2,2,4),plt.imshow(mask_result_315,cmap = 'gray'),plt.axis('off'), plt.title('MASK-result')
plt.show()

#317结果
segmented_image_317 , mask_median_317 , mask_result_317 = EMGMMsegment(HSV_image_317)
plt.figure()
plt.subplot(2,2,1),plt.imshow(RGB_image_317),plt.axis('off'), plt.title('EM  317')
plt.subplot(2,2,2),plt.imshow(segmented_image_317),plt.axis('off'), plt.title('EM-segment')
plt.subplot(2,2,3),plt.imshow(mask_median_317,cmap = 'gray'),plt.axis('off'), plt.title('MASK-median')
plt.subplot(2,2,4),plt.imshow(mask_result_317,cmap = 'gray'),plt.axis('off'), plt.title('MASK-result')
plt.show()


#############################################################################################################

##################HSV图像分割#######################################################################
RGBDistribution(RGB_image_311)
HSVdistribution(RGB_image_311)

RGB_result_311,Mask_result_311 = Hsvsegment(RGB_image_311)
RGB_result_313,Mask_result_313 = Hsvsegment(RGB_image_313)
RGB_result_315,Mask_result_315 = Hsvsegment(RGB_image_315)
RGB_result_317,Mask_result_317 = Hsvsegment(RGB_image_317)

plt.figure()
plt.subplot(4,3,1),plt.imshow(RGB_image_311),plt.axis('off'), plt.title('311-RGB')
plt.subplot(4,3,2),plt.imshow(RGB_result_311),plt.axis('off'), plt.title('RGB-MASK')
plt.subplot(4,3,3),plt.imshow(Mask_result_311, cmap='gray'),plt.axis('off'), plt.title('MASK')
plt.subplot(4,3,4),plt.imshow(RGB_image_313),plt.axis('off'), plt.title('313-RGB')
plt.subplot(4,3,5),plt.imshow(RGB_result_313),plt.axis('off'), plt.title('RGB-MASK')
plt.subplot(4,3,6),plt.imshow(Mask_result_313, cmap='gray'),plt.axis('off'), plt.title('MASK')
plt.subplot(4,3,7),plt.imshow(RGB_image_315),plt.axis('off'), plt.title('315-RGB')
plt.subplot(4,3,8),plt.imshow(RGB_result_315),plt.axis('off'), plt.title('RGB-MASK')
plt.subplot(4,3,9),plt.imshow(Mask_result_315, cmap='gray'),plt.axis('off'), plt.title('MASK')
plt.subplot(4,3,10),plt.imshow(RGB_image_317),plt.axis('off'), plt.title('317-RGB')
plt.subplot(4,3,11),plt.imshow(RGB_result_317),plt.axis('off'), plt.title('RGB-MASK')
plt.subplot(4,3,12),plt.imshow(Mask_result_317, cmap='gray'),plt.axis('off'), plt.title('MASK')
plt.show()

####################################################################################################