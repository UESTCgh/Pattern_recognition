clear;clc
%% 处理异常数据
process_data('data.xlsx', 'filtered_data.xlsx');

%% 重新加载
data = readtable('filtered_data.xlsx');

%% 画图
plot_weight('filtered_data.xlsx');

%% 求最大似然估计参数
[max_male_params, max_female_params] = max_estimate('filtered_data.xlsx');

%% 求贝叶斯估计参数 选定方差为1,先验均值，女生59 男生69.6
%参数设置
female_xy_u0 = 59;%kg
male_xy_u0 = 69.6;%kg
%计算
[bys_male_mean, bys_male_variance, bys_female_mean, bys_female_variance] = bayesian_estimate('filtered_data.xlsx',female_xy_u0,male_xy_u0);

%% 决策
height = 167;
weight = 52;
plot_decision('filtered_data.xlsx',height,weight);

clear;