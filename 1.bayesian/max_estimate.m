function [male_params, female_params] = max_estimate(input_filename)
    % 读取Excel文件
    data = readtable(input_filename);

    % 分别获取男生和女生的体重数据
    male_weight = data.Weight(data.Gender == 1);
    female_weight = data.Weight(data.Gender == 0);

    % 对男生体重进行最大似然估计（假设为正态分布）
    male_mean = mean(male_weight);
    male_std = std(male_weight);
    male_params = [male_mean, male_std];

    % 对女生体重进行最大似然估计（假设为正态分布）
    female_mean = mean(female_weight);
    female_std = std(female_weight);
    female_params = [female_mean, female_std];

    % 显示结果
    fprintf('男生总体的最大似然估计(MLE): 均值 = %.2f, 方差 = %.2f\n', male_mean, male_std);
    fprintf('女生总体的最大似然估计(MLE): 均值 = %.2f, 方差 = %.2f\n', female_mean, female_std);
end
