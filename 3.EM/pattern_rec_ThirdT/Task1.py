import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import PIL.Image as Image
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter



#读取数据函数
def DataRead(Test_Data_path,Mask_Data_path):
    data = pd.read_excel(Test_Data_path,header = None)       #读取训练有样本数据
    data_array = np.array(data)
    Gray_Data = data_array[:,0]                              #提取第一列灰度样本数据
    Gray_Data = Gray_Data.reshape(Gray_Data.shape[0],1)
    RGB_Data = data_array[:,1:]                              #提取后三列RGB数据
    Mask_data = pd.read_excel(Mask_Data_path,header = None)  #提取背景数据，用于分割前景
    Mask_data = np.array(Mask_data)

    return Gray_Data,RGB_Data,Mask_data

#绘出直方图函数
def Histogram(data):
    plt.figure()
    plt.hist(data,bins=100,alpha=0.7,color='blue',label='grayData')
    plt.xlabel('GRAYDATA')
    plt.ylabel('numbers')
    plt.title('GRAYDATA_Histogram')
    plt.legend()
    plt.show()

#一元GMM函数
def UnivariateGMM(Prior_Pred_1 , gray_mean_1 , gray_sigma_1 , Prior_Pred_2 , gray_mean_2 , gray_sigma_2 , Gray_Data , gray_test_image_ROI):

    x = np.arange(0, 1, 1 / 1000)
    epochs =30            #最高迭代次数
    ims_Gray_out=[]       #存放聚类结果，用于动图显示
    pdf1_out = []         #存放第一类的类概率密度函数，用于动图显示
    pdf2_out = []         #存放第二类的类概率密度函数，用于动图显示

    #构造一个数组存放每一个样本的过程参数
    graySample_data =np.zeros((Gray_Data.shape[0],4))

    for epoch in range(epochs):
        # E步，更新参数
        for i in range(Gray_Data.shape[0]):
            #计算每一个样本属于每一类的隶属度，即隐函数的值
            Posterior_pred_1 = (Prior_Pred_1 * norm.pdf(Gray_Data[i][0] , gray_mean_1 , gray_sigma_1))/ \
                               ((Prior_Pred_1 * norm.pdf(Gray_Data[i][0] , gray_mean_1 , gray_sigma_1))+
                               (Prior_Pred_2 * norm.pdf(Gray_Data[i][0] , gray_mean_2 , gray_sigma_2)))

            #二分类问题，另一类问题直接用1减
            Posterior_pred_2 = 1-Posterior_pred_1

            #记录每一个样本的各个参数
            graySample_data[i][0] = Posterior_pred_1 * 1.0   #第一类隶属度
            graySample_data[i][1] = Posterior_pred_2 * 1.0   #第二类隶属度
            graySample_data[i][2] = Posterior_pred_1 * Gray_Data[i][0]  # 第一类均值中间值
            graySample_data[i][3] = Posterior_pred_2 * Gray_Data[i][0]  # 第二类均值中间值

        #计算Nk
        num_k_1 = sum(graySample_data)[0]
        num_k_2 = sum(graySample_data)[1]

        #计算类均值
        gray_mean_1 = sum(graySample_data)[2] / num_k_1     #第一类均值
        gray_mean_2 = sum(graySample_data)[3] / num_k_2     #第二类均值

        #计算每一类的标准差
        sum_median_1 = 0.0
        sum_median_2 = 0.0
        for i in range(Gray_Data.shape[0]):
            sum_median_1 = sum_median_1 + graySample_data[i][0] * pow((Gray_Data[i][0] - gray_mean_1) , 2)#第一类标准差分子
            sum_median_2 = sum_median_2 + graySample_data[i][1] * pow((Gray_Data[i][0] - gray_mean_2) , 2)#第二类标准差分子

        gray_sigma_1 = pow((sum_median_1 / num_k_1) , 0.5) #第一类标准差
        gray_sigma_2 = pow((sum_median_2 / num_k_2) , 0.5) #第二类标准差


        #更新先验概率
        print("第",epoch+1,"次迭代 ：")
        Prior_Pred_1 = num_k_1 / (num_k_1 + num_k_2)
        print("第一类       :     先验概率 ： ", Prior_Pred_1,"    均值 ：" ,gray_mean_1 ,"    标准差 ：",gray_sigma_1)
        Prior_Pred_2 = num_k_2 / (num_k_1 + num_k_2)
        print("第二类       :     先验概率 ： ", Prior_Pred_2,"    均值 ：" ,gray_mean_2 ,"    标准差 ：",gray_sigma_2)

        # 计算两类高斯分布的概率密度
        pdf1 = norm.pdf(x, loc=gray_mean_1, scale=gray_sigma_1)
        pdf2 = norm.pdf(x, loc=gray_mean_2, scale=gray_sigma_2)

        # 将PDF数值数据添加到列表中
        pdf1_out.append(pdf1)
        pdf2_out.append(pdf2)



        #M步，根据贝叶斯最大后验概率分类,只看分子

        gray_out = np.zeros_like(gray_test_image_ROI)  #构造一个和测试图像大小相同的零矩阵
        for i in range(gray_test_image_ROI.shape[0]):
            for j in range(gray_test_image_ROI.shape[1]):
                if gray_test_image_ROI[i][j] == 0 :#背景像素点
                    continue
                elif (Prior_Pred_1 * norm.pdf(gray_test_image_ROI[i][j] , gray_mean_1 , gray_sigma_1)) > \
                        (Prior_Pred_2 * norm.pdf(gray_test_image_ROI[i][j] , gray_mean_2 , gray_sigma_2)) :
                    gray_out[i][j] = 100
                else:
                    gray_out[i][j] = 255

        # 显示聚类结果
        gray_out = gray_out/255.0
        ims_Gray_out.append(gray_out)

    # 显示动画
    fig1, ax1 = plt.subplots()
    ax1.set_title('PDF Animation')
    num_iterations =  len(pdf1_out)

    def animate(frame):
        ax1.clear()
        ax1.plot(x, pdf1_out[frame], label='Class 1', color='g')
        ax1.plot(x, pdf2_out[frame], label='Class 2', color='y')
        ax1.set_title(f'Epoch {frame + 1}')
        ax1.legend()

    ani = FuncAnimation(fig1, animate, frames=num_iterations, blit=False)
    plt.show()

    # 准备绘制动画，聚类结果的动画
    fig2,ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_title('Gray segment')

    # 创建一个函数来更新每一帧
    def update(frame):
        ax2.clear()
        ax2.imshow(ims_Gray_out[frame],cmap = 'gray')
        ax2.set_title(f'Epoch {frame + 1}')

    # 创建一个动画对象
    ani = FuncAnimation(fig2, update, frames=len(ims_Gray_out), blit=False)

    # 显示动画
    plt.show()



#三元GMM函数（RGB三通道）
def MultivariateGMM(RGB_Prior_Pred_1 , RGB_mean_1 , RGB_cov_1 , RGB_Prior_Pred_2 , RGB_mean_2 , RGB_cov_2 , RGB_Data , RGB_test_image_ROI) :
    epochs = 30

    # 准备绘制动画
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('RGB segment')
    ims = []

    # 构造一个数组存放每一个样本的过程参数
    RGBSample_data = np.zeros((RGB_Data.shape[0], 8))

    for epoch in range(epochs):
        for i in range(RGB_Data.shape[0]):
            # 计算每一个样本属于每一类的隶属度，即隐函数的值
            Posterior_pred_1 = (RGB_Prior_Pred_1 * multivariate_normal.pdf(RGB_Data[i][:], RGB_mean_1, RGB_cov_1)) / \
                               ((RGB_Prior_Pred_1 * multivariate_normal.pdf(RGB_Data[i][:], RGB_mean_1, RGB_cov_1)) +
                                (RGB_Prior_Pred_2 * multivariate_normal.pdf(RGB_Data[i][:], RGB_mean_2, RGB_cov_2)))

            # 二分类问题，另一类问题直接用1减
            Posterior_pred_2 = 1 - Posterior_pred_1

            # 记录每一个样本的各个参数
            RGBSample_data[i][0] = Posterior_pred_1 * 1.0  # 第一类隶属度
            RGBSample_data[i][1] = Posterior_pred_2 * 1.0  # 第二类隶属度
            RGBSample_data[i][2:5] = Posterior_pred_1 * RGB_Data[i][:]  # 第一类均值中间值
            RGBSample_data[i][5:] = Posterior_pred_2 * RGB_Data[i][:]  # 第二类均值中间值

        # 计算Nk
        num_k_1 = sum(RGBSample_data)[0]
        num_k_2 = sum(RGBSample_data)[1]

        # 计算类均值
        RGB_mean_1 = sum(RGBSample_data)[2:5] / num_k_1  # 第一类均值 1*3的矩阵
        RGB_mean_2 = sum(RGBSample_data)[5:] / num_k_2  # 第二类均值

        # 计算每一类的协方差
        sum_median_1 = np.zeros((3,3))
        sum_median_2 = np.zeros((3,3))
        for i in range(RGB_Data.shape[0]):
            sum_median_1 = sum_median_1 + RGBSample_data[i][0] * np.dot((RGB_Data[i][:] - RGB_mean_1).reshape(3,1) ,
                                                                        (RGB_Data[i][:] - RGB_mean_1).reshape(1,3))  # 第一类协方差分子
            sum_median_2 = sum_median_2 + RGBSample_data[i][1] * np.dot((RGB_Data[i][:] - RGB_mean_2).reshape(3,1) ,
                                                                        (RGB_Data[i][:] - RGB_mean_2).reshape(1,3))  # 第二类协方差分子

        RGB_cov_1 = sum_median_1 / (num_k_1 - 1)
        RGB_cov_2 = sum_median_2 / (num_k_2 - 1)

        # 更新先验概率
        Prior_Pred_1 = num_k_1 / (num_k_1 + num_k_2)
        Prior_Pred_2 = num_k_2 / (num_k_1 + num_k_2)

        print("第",epoch+1,"次迭代 ：")
        print("第一类       :     先验概率 ： ", Prior_Pred_1, "    均值 ：", RGB_mean_1, "    协方差 ：", RGB_cov_1)
        print("第二类       :     先验概率 ： ", Prior_Pred_2, "    均值 ：", RGB_mean_2, "    协方差 ：", RGB_cov_2)

        #M步，根据贝叶斯最大后验概率分类,只看分子
        RGB_out = np.zeros_like(RGB_test_image_ROI)  #构造一个和测试图像大小相同的零矩阵
        for i in range(RGB_test_image_ROI.shape[0]):
            for j in range(RGB_test_image_ROI.shape[1]):
                if np.sum(RGB_test_image_ROI[i][j]) == 0 :#背景像素点
                    continue
                elif (Prior_Pred_1 * multivariate_normal.pdf(RGB_test_image_ROI[i][j] , RGB_mean_1 , RGB_cov_1)) > \
                        (Prior_Pred_2 * multivariate_normal.pdf(RGB_test_image_ROI[i][j] , RGB_mean_2 , RGB_mean_2)) :
                    RGB_out[i][j] = [255 ,0  , 0]
                else:
                    RGB_out[i][j] = [255 ,255 , 255]

        # 显示聚类结果
        RGB_out = RGB_out/255.0
        ims.append(RGB_out)

    # 创建一个函数来更新每一帧
    def update(frame):
        ax.clear()
        ax.imshow(ims[frame])
        ax.set_title(f'Epoch {frame + 1}')

    # 创建一个动画对象
    ani = FuncAnimation(fig, update, frames=len(ims), blit=False)

    # 显示动画
    plt.show()

####################数据读取及与预处理###############################################
Test_Data_path = 'array_sample.xlsx'
Mask_Data_path = 'Mask.xlsx'

#读取训练集数据
Gray_Data,RGB_Data,Mask_Data = DataRead(Test_Data_path,Mask_Data_path)

#读取测试图片
test_image = Image.open('309.bmp')
gray_test_image = np.array(test_image.convert('L'))    #将RGB图像转换为灰度图像
gray_test_image_ROI = (gray_test_image * Mask_Data)/255      #通过Mask分割前景
RGB_Mask = np.array([Mask_Data, Mask_Data, Mask_Data]).transpose(1,2,0)
RGB_test_image_ROI = (test_image * RGB_Mask)/255
##################################################################################

##########################一元高斯#################################################
#绘制灰度分布直方图，观察训练集中灰度分布情况，估计初值
# Histogram(Gray_Data)

#假设两个类别的先验概率一样，都为0.5
gray_Prior_Pred_1 = 0.5
gray_Prior_Pred_2 = 0.5
# 估计假设的第一类的均值和方差
gray_mean_1 = 0.5
gray_sigma_1 = 0.1
# 估计假设的第二类的均值和方差
gray_mean_2 = 0.8
gray_sigma_2 = 0.3

UnivariateGMM(gray_Prior_Pred_1 ,gray_mean_1, gray_sigma_1,gray_Prior_Pred_2 ,gray_mean_2,gray_sigma_2,Gray_Data , gray_test_image_ROI)
#################################################################################

##########################三元高斯################################################

#假设两个类别的先验概率一样，都为0.5
RGB_Prior_Pred_1 = 0.5
RGB_Prior_Pred_2 = 0.5

RGB_mean_1 = np.array([0.5, 0.5, 0.5])
RGB_mean_2 = np.array([0.8, 0.8, 0.8])
RGB_cov_1 = np.array([[0.1, 0.05, 0.04],
                    [0.05, 0.1, 0.02],
                    [0.04, 0.02, 0.1]])
RGB_cov_2 = np.array([[0.1, 0.05, 0.04],
                    [0.05, 0.1, 0.02],
                    [0.04, 0.02, 0.1]])


MultivariateGMM(RGB_Prior_Pred_1 , RGB_mean_1 , RGB_cov_1 , RGB_Prior_Pred_2 , RGB_mean_2 , RGB_cov_2 , RGB_Data , RGB_test_image_ROI)

#################################################################################