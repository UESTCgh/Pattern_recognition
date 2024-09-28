clear;clc
%% 1.处理异常数据
process_data('data.xlsx', 'filtered_data.xlsx');
% 重新加载过滤后的数据
data = readtable('filtered_data.xlsx');

%% 2.画直方图
plot_weight('filtered_data.xlsx');

%% 3.1 求最大似然估计参数
[max_male_params, max_female_params] = max_estimate('filtered_data.xlsx');

%% 3.2 求贝叶斯估计参数 
% 选定方差为1,先验均值，女生59 男生69.6
% 先验参数设置
female_xy_u0 = 59;%kg 女生均值
male_xy_u0 = 69.6;%kg 男生均值
variance = 1;%先验方差
% 计算
[bys_male_mean, bys_male_variance, bys_female_mean, bys_female_variance] = bayesian_estimate('filtered_data.xlsx',female_xy_u0,male_xy_u0,variance);

%% 4.输入样本决策
%样本数据
height = 167;
weight = 52;
%打印决策面
plot_decision('filtered_data.xlsx',height,weight);

clear;