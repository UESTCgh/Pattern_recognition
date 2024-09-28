function [bys_male_mean, bys_male_variance, bys_female_mean, bys_female_variance] = bayesian_estimate(input_filename,female_u0,male_u0,prior_variance)
    %% 1.读取Excel文件中的数据
    data = readtable(input_filename);
    % 分别提取男生和女生的体重数据
    male_weights = data.Weight(data.Gender == 1);
    female_weights = data.Weight(data.Gender == 0);

    %% 2 男生的贝叶斯参数估计
    % 2.1 样本均值方差
    % 男生样本数量
    n_male = length(male_weights);  

    % 男生样本均值和方差
    male_mean_sample = mean(male_weights);  
    male_variance_sample = var(male_weights);  

    % 2.2 先验均值方差获取
    % 先验均值 (假设为固定值或输入参数)
    mu0_male_prior = male_u0;

    % 2.3 根据公式计算后验参数估计
    bys_male_mean = (mu0_male_prior / prior_variance + n_male * male_mean_sample / male_variance_sample) / ...
                    (1 / prior_variance + n_male / male_variance_sample);
    bys_male_variance = 1 / (1 / prior_variance + n_male / male_variance_sample);

    %% 3 女生的贝叶斯参数估计
    % 2.1 样本均值方差
    % 女生样本数量
    n_female = length(female_weights);  

    % 女生样本均值和方差
    female_mean_sample = mean(female_weights);  
    female_variance_sample = var(female_weights);  

    % 2.2 先验均值方差获取
    % 先验均值 (假设为固定值或输入参数)
    mu0_female_prior = female_u0;
    % 2.3 根据公式计算后验参数估计
    % 计算女生的后验均值和方差
    bys_female_mean = (mu0_female_prior / prior_variance + n_female * female_mean_sample / female_variance_sample) / ...
                      (1 / prior_variance + n_female / female_variance_sample);
    bys_female_variance = 1 / (1 / prior_variance + n_female / female_variance_sample);

    %% 4.显示计算结果
    fprintf('选取男生先验均值: %.2f, 方差: %.2f，女生先验均值: %.2f, 方差: %.2f\n', mu0_male_prior, prior_variance,mu0_female_prior,prior_variance);
    fprintf('男生的贝叶斯后验估计：均值: %.2f, 方差: %.2f\n', bys_male_mean, bys_male_variance);
    fprintf('女生的贝叶斯后验估计：均值: %.2f, 方差: %.2f\n', bys_female_mean, bys_female_variance);
end
