function plot_weight(input_filename)
    %%该函数绘制男生女生的体重直方图
    
    %% 1.分别获取男女生的体重数据
    data = readtable(input_filename);
    % 分别获取男生和女生的体重数据
    male_weight = data.Weight(data.Gender == 1);
    female_weight = data.Weight(data.Gender == 0);

    %% 2.绘制直方图
    figure;
    hold on;
    
    % 男生体重直方图
    histogram(male_weight, 'FaceColor', 'b', 'EdgeColor', 'k', 'FaceAlpha', 0.5);
    % 女生体重直方图
    histogram(female_weight, 'FaceColor', 'r', 'EdgeColor', 'k', 'FaceAlpha', 0.5);
   
    % 图表标题和标签
    title('男女生体重直方图');
    xlabel('体重(kg)');
    ylabel('频数');
    
    % 添加图例
    legend('男生', '女生');

    % 显示网格
    grid on;
    hold off;
end
