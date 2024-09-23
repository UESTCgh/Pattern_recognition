function process_data(input_filename, output_filename)
    % 读取Excel文件
    data_l = readtable(input_filename);

    % 确认列的数量
    num_columns = width(data_l); % 获取数据集列数

    if num_columns == 11
        % 修改所有列名
        data_l.Properties.VariableNames(1:11) = {'Num','Gender','Origin','Height','Weight','Size','50m','Lungs','Color','Sport','Art'};
    else
        error('Number of new column names does not match the number of columns in the dataset.');
    end

    % 过滤性别、喜欢运动和喜欢文学的数据（只保留合法值 0 和 1）
    data = data_l(data_l.Gender == 0 | data_l.Gender == 1, :);
    % data = data(data.Sport == 0 | data.Sport == 1, :);
    % data = data(data.Art == 0 | data.Art == 1, :);

    % 设定异常值的过滤阈值（对身高和体重的异常值进行过滤）

    % 计算身高的均值和标准差
    height_mean = mean(data.Height, 'omitnan');
    height_std = std(data.Height, 'omitnan');

    % 计算体重的均值和标准差
    weight_mean = mean(data.Weight, 'omitnan');
    weight_std = std(data.Weight, 'omitnan');

    % 设置过滤条件，保留在3个标准差范围内的数据
    height_threshold = 3;
    weight_threshold = 3;

    % 过滤掉超出范围的异常值
    data = data(abs(data.Height - height_mean) <= height_threshold * height_std & ...
                abs(data.Weight - weight_mean) <= weight_threshold * weight_std, :);

    % 显示过滤后的数据
    disp(data);

    % 保存过滤后的数据到新的Excel文件
    writetable(data, output_filename);
end
