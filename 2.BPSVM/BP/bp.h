// bp.h
#ifndef BP_H
#define BP_H

#include <vector>
#include <string> // 添加此行以支持 std::string

// BP 神经网络类定义
class BPNeuralNetwork {
public:
    // 构造函数：初始化输入层、隐藏层、输出层节点数以及学习率
    BPNeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate);

    // 前向传播函数：计算网络的输出
    std::vector<double> forward(const std::vector<double>& input);

    // 反向传播函数：根据目标值计算误差并更新权重
    void backward(const std::vector<double>& target, double class_weight); // 添加类别权重参数

    // 保存模型参数到文件
    void saveModel(const std::string& filename);

    // 从文件加载模型参数
    void loadModel(const std::string& filename);

private:
    // 权重和偏置
    std::vector<std::vector<double>> weights_input_hidden; // 输入层到隐藏层的权重
    std::vector<std::vector<double>> weights_hidden_output; // 隐藏层到输出层的权重
    std::vector<double> bias_hidden; // 隐藏层偏置
    std::vector<double> bias_output; // 输出层偏置
    double learning_rate; // 学习率

    // 各层输出
    std::vector<double> input; // 输入层
    std::vector<double> hidden_output; // 隐藏层输出
    std::vector<double> final_output; // 最终输出

    double sigmoid(double x);                   // Sigmoid 激活函数（可选）
    double sigmoid_derivative(double x);        // Sigmoid 导数（可选）
    double relu(double x);
    double relu_derivative(double x);
};

#endif // BP_H
