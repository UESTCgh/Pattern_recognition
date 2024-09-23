function plot_decision(input_filename,s_hight,s_weight)

data = readtable(input_filename);

% 分别获取男生和女生的身高和体重数据
male_data = data(data.Gender == 1, {'Height', 'Weight'});
female_data = data(data.Gender == 0, {'Height', 'Weight'});

% 计算男生和女生的均值向量和协方差矩阵
mu_male = mean(male_data{:,:});  % 男生均值向量
mu_female = mean(female_data{:,:});  % 女生均值向量

sigma_male = cov(male_data{:,:});  % 男生协方差矩阵
sigma_female = cov(female_data{:,:});  % 女生协方差矩阵

% 手动计算多元正态分布PDF
function p = my_mvnpdf(x, mu, sigma)
    d = length(mu);  % 维度 (2维)
    x_mu = x - mu;   % (x - mu)
    p = (1 / ((2*pi)^(d/2) * sqrt(det(sigma)))) * exp(-0.5 * (x_mu / sigma) * x_mu');
end

% 绘制决策面
figure;
hold on;

% 生成网格数据
[x1Grid, x2Grid] = meshgrid(150:1:190, 40:1:80);
XGrid = [x1Grid(:), x2Grid(:)];  % 网格点

% 计算网格上男生和女生的判别值
g_male = arrayfun(@(i) my_mvnpdf(XGrid(i, :), mu_male, sigma_male), 1:size(XGrid, 1));  % 男生联合概率密度
g_female = arrayfun(@(i) my_mvnpdf(XGrid(i, :), mu_female, sigma_female), 1:size(XGrid, 1));  % 女生联合概率密度

% 计算决策面
decision_surface = reshape(g_male - g_female, size(x1Grid));

% 绘制等高线决策面，决策面为等高线值为0的位置
contour(x1Grid, x2Grid, decision_surface, [0 0], 'k', 'LineWidth', 2);

% 绘制男生和女生的散点图
scatter(male_data.Height, male_data.Weight, 'b', 'filled');
scatter(female_data.Height, female_data.Weight, 'r', 'filled');

% 添加标题和图例
title('性别判定的决策面');
xlabel('身高 (cm)');
ylabel('体重 (kg)');
legend('决策分支', '男生', '女生', 'Location', 'best');

hold off;

% 样本身高体重的分类
sample = [s_hight, s_weight];
fprintf('选择身高为%.2fkg，体重为%.2fcm的测试集\n',s_hight,s_weight);

% 计算样本属于男生和女生的概率
p_male = my_mvnpdf(sample, mu_male, sigma_male);  % 男生概率
p_female = my_mvnpdf(sample, mu_female, sigma_female);  % 女生概率

% 分类决策
if p_male > p_female
    fprintf('分类结果为男生\n');
else
    fprintf('分类结果为女生\n');
end

end

