// bp.cpp
#include "bp.h"
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

// BP 神经网络构造函数，初始化权重和偏置
BPNeuralNetwork::BPNeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate)
        : learning_rate(learning_rate) {
    // 使用随机数生成器初始化权重和偏置，范围为 -1 到 1
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    weights_input_hidden.resize(input_size, std::vector<double>(hidden_size));
    weights_hidden_output.resize(hidden_size, std::vector<double>(output_size));
    bias_hidden.resize(hidden_size);
    bias_output.resize(output_size);

    std::normal_distribution<double> xavier_dist(0.0, sqrt(1.0 / input_size));
    for (auto& row : weights_input_hidden) {
        for (auto& weight : row) {
            weight = xavier_dist(generator);
        }
    }

    for (auto& bias : bias_hidden) {
        bias = distribution(generator);
    }

    for (auto& bias : bias_output) {
        bias = distribution(generator);
    }
}

// 前向传播函数：计算网络的输出
std::vector<double> BPNeuralNetwork::forward(const std::vector<double>& input) {
    this->input = input;

    // 计算隐藏层输出
    hidden_output.resize(bias_hidden.size());
    for (size_t i = 0; i < hidden_output.size(); ++i) {
        double sum = bias_hidden[i];
        for (size_t j = 0; j < input.size(); ++j) {
            sum += input[j] * weights_input_hidden[j][i];
        }
       hidden_output[i] = sigmoid(sum);
       // hidden_output[i] = relu(sum);  // 使用 ReLU 代替 Sigmoid

    }

    // 计算最终输出
    final_output.resize(bias_output.size());
    for (size_t i = 0; i < final_output.size(); ++i) {
        double sum = bias_output[i];
        for (size_t j = 0; j < hidden_output.size(); ++j) {
            sum += hidden_output[j] * weights_hidden_output[j][i];
        }
        final_output[i] = sigmoid(sum);
    }

    return final_output;
}

// 反向传播函数：根据目标值计算误差并更新权重
void BPNeuralNetwork::backward(const std::vector<double>& target, double class_weight) {
    // 计算输出层误差
    std::vector<double> output_error(final_output.size());
    for (size_t i = 0; i < final_output.size(); ++i) {
        output_error[i] = (target[i] - final_output[i]) * sigmoid_derivative(final_output[i]) * class_weight; // 应用类别权重
    }

    // 计算隐藏层误差
    std::vector<double> hidden_error(hidden_output.size(), 0.0);
    for (size_t i = 0; i < hidden_output.size(); ++i) {
        for (size_t j = 0; j < output_error.size(); ++j) {
            hidden_error[i] += output_error[j] * weights_hidden_output[i][j];
        }
        hidden_error[i] *= sigmoid_derivative(hidden_output[i]);
    }

    // 更新权重和偏置（隐藏层到输出层）
    for (size_t i = 0; i < weights_hidden_output.size(); ++i) {
        for (size_t j = 0; j < weights_hidden_output[i].size(); ++j) {
            weights_hidden_output[i][j] += learning_rate * hidden_output[i] * output_error[j];
        }
    }
    for (size_t i = 0; i < bias_output.size(); ++i) {
        bias_output[i] += learning_rate * output_error[i];
    }

    // 更新权重和偏置（输入层到隐藏层）
    for (size_t i = 0; i < weights_input_hidden.size(); ++i) {
        for (size_t j = 0; j < weights_input_hidden[i].size(); ++j) {
            weights_input_hidden[i][j] += learning_rate * input[i] * hidden_error[j];
        }
    }
    for (size_t i = 0; i < bias_hidden.size(); ++i) {
        bias_hidden[i] += learning_rate * hidden_error[i];
    }
}


// Sigmoid 激活函数
double BPNeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Sigmoid 激活函数的导数
double BPNeuralNetwork::sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

double BPNeuralNetwork::relu(double x) {
    return (x > 0) ? x : 0;
}

double BPNeuralNetwork::relu_derivative(double x) {
    return (x > 0) ? 1 : 0;
}

// 保存模型参数到文件
void BPNeuralNetwork::saveModel(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "无法打开文件保存模型参数: " << filename << std::endl;
        return;
    }

    // 保存输入层到隐藏层的权重
    ofs << weights_input_hidden.size() << " " << weights_input_hidden[0].size() << std::endl;
    for (const auto& row : weights_input_hidden) {
        for (const auto& weight : row) {
            ofs << weight << " ";
        }
        ofs << std::endl;
    }

    // 保存隐藏层到输出层的权重
    ofs << weights_hidden_output.size() << " " << weights_hidden_output[0].size() << std::endl;
    for (const auto& row : weights_hidden_output) {
        for (const auto& weight : row) {
            ofs << weight << " ";
        }
        ofs << std::endl;
    }

    // 保存偏置
    ofs << bias_hidden.size() << std::endl;
    for (const auto& bias : bias_hidden) {
        ofs << bias << " ";
    }
    ofs << std::endl;

    ofs << bias_output.size() << std::endl;
    for (const auto& bias : bias_output) {
        ofs << bias << " ";
    }
    ofs << std::endl;

    ofs.close();
    std::cout << "save to:" << filename << std::endl;
}

// 从文件加载模型参数
void BPNeuralNetwork::loadModel(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "无法打开文件加载模型参数: " << filename << std::endl;
        return;
    }

    // 加载输入层到隐藏层的权重
    int hidden_size, input_size;
    ifs >> input_size >> hidden_size;
    weights_input_hidden.resize(input_size, std::vector<double>(hidden_size));
    for (auto& row : weights_input_hidden) {
        for (auto& weight : row) {
            ifs >> weight;
        }
    }

    // 加载隐藏层到输出层的权重
    int output_size;
    ifs >> hidden_size >> output_size;
    weights_hidden_output.resize(hidden_size, std::vector<double>(output_size));
    for (auto& row : weights_hidden_output) {
        for (auto& weight : row) {
            ifs >> weight;
        }
    }

    // 加载偏置
    int hidden_bias_size;
    ifs >> hidden_bias_size;
    bias_hidden.resize(hidden_bias_size);
    for (auto& bias : bias_hidden) {
        ifs >> bias;
    }

    int output_bias_size;
    ifs >> output_bias_size;
    bias_output.resize(output_bias_size);
    for (auto& bias : bias_output) {
        ifs >> bias;
    }

    ifs.close();
    std::cout << "模型参数已从 " << filename << " 加载" << std::endl;
}