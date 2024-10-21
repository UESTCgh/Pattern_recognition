#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <locale>
#include <codecvt>
#include <random>
#include "bp.h"

namespace fs = std::filesystem;

/****************************数据过滤范围********************************/
const double MIN_HEIGHT = 140.0;//身高
const double MAX_HEIGHT = 220.0;
const double MIN_WEIGHT = 30.0;//体重
const double MAX_WEIGHT = 150.0;
const double MIN_SHOE_SIZE = 30.0;//鞋码
const double MAX_SHOE_SIZE = 50.0;
const double MIN_50M_TIME = 5.0;//50m
const double MAX_50M_TIME = 15.0;
const double MIN_VITAL_CAPACITY = 1000.0;//肺活量
const double MAX_VITAL_CAPACITY = 8000.0;

/****************************数据预处理********************************/
// 检查数据是否在合理范围内
bool isValidData(const std::vector<std::string> &data) {
    try {
        // 检查是否有空白数据
        for (const auto &item : data) {
            if (item.empty()) {
                return false;
            }
        }

        double gender = std::stod(data[1]);
        double height = std::stod(data[3]);
        double weight = std::stod(data[4]);
        double shoeSize = std::stod(data[5]);
        double fiftyMeter = std::stod(data[6]);
        double vitalCapacity = std::stod(data[7]);

        return (gender == 0 || gender == 1) &&
               (height >= MIN_HEIGHT && height <= MAX_HEIGHT) &&
               (weight >= MIN_WEIGHT && weight <= MAX_WEIGHT) &&
               (shoeSize >= MIN_SHOE_SIZE && shoeSize <= MAX_SHOE_SIZE) &&
               (fiftyMeter >= MIN_50M_TIME && fiftyMeter <= MAX_50M_TIME) &&
               (vitalCapacity >= MIN_VITAL_CAPACITY && vitalCapacity <= MAX_VITAL_CAPACITY);
    } catch (const std::exception &e) {
        // 如果转换失败，数据无效
        return false;
    }
}

// 解析CSV行
std::vector<std::string> parseCSVLine(const std::string &line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ',')) {
        result.push_back(item);
    }
    return result;
}

// 打开文件并处理数据
bool processFile(const std::string& inputFilePath, const std::string& outputFilePath, const std::string& invalidFilePath) {
    if (!fs::exists(inputFilePath)) {
        std::cerr << "错误：输入文件不存在：" << inputFilePath << std::endl;
        return false;
    }

    std::wifstream inputFile(inputFilePath);
    inputFile.imbue(std::locale(inputFile.getloc(), new std::codecvt_utf8<wchar_t>));
    std::wofstream outputFile(outputFilePath, std::ios::out | std::ios::binary);
    outputFile.imbue(std::locale(outputFile.getloc(), new std::codecvt_utf8<wchar_t>));
    std::wofstream invalidFile(invalidFilePath, std::ios::out | std::ios::binary);
    invalidFile.imbue(std::locale(invalidFile.getloc(), new std::codecvt_utf8<wchar_t>));

    if (!inputFile.is_open()) {
        std::cerr << "输入文件正被使用" << inputFilePath << std::endl;
        return false;
    }

    if (!outputFile.is_open()) {
        std::cerr << "无法创建输出文件" << outputFilePath << std::endl;
        return false;
    }

    if (!invalidFile.is_open()) {
        std::cerr << "无法创建无效数据文件" << invalidFilePath << std::endl;
        return false;
    }

    std::wstring line;
    bool isFirstLine = true;

    while (std::getline(inputFile, line)) {
        // 写入标题行
        if (isFirstLine) {
            outputFile << line << std::endl;
            invalidFile << line << std::endl;
            isFirstLine = false;
            continue;
        }

        // 解析行数据
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::string utf8Line = converter.to_bytes(line);
        std::vector<std::string> data = parseCSVLine(utf8Line);

        // 检查数据完整性和有效性
        if (data.size() < 8) {
            invalidFile << line << std::endl; // 跳过不完整的行并记录
            continue;
        }

        // 检查是否是有效的数据
        if (isValidData(data)) {
            outputFile << line << std::endl;
        } else {
            invalidFile << line << std::endl; // 记录无效的数据
        }
    }

    inputFile.close();
    outputFile.close();
    invalidFile.close();

    std::cout << "数据过滤结果保存在 " << outputFilePath << std::endl;
    std::cout << "无效数据保存在 " << invalidFilePath << std::endl;
    return true;
}

/****************************数据集划分********************************/
// 函数：随机划分CSV文件为训练集和测试集，并存到文件中
void splitCSVFile(const std::string& inputFilePath, const std::string& trainFilePath, const std::string& testFilePath, double train_ratio = 0.7) {
    std::ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        std::cerr << "无法打开输入文件: " << inputFilePath << std::endl;
        return;
    }

    std::vector<std::string> lines;
    std::string line;
    bool isFirstLine = true;
    std::string header;

    // 读取CSV所有行
    while (std::getline(inputFile, line)) {
        if (isFirstLine) {
            header = line;
            isFirstLine = false;
            continue;
        }
        lines.push_back(line);
    }
    inputFile.close();

    // 随机打乱数据
    std::shuffle(lines.begin(), lines.end(), std::default_random_engine(std::random_device{}()));

    // 划分训练集和测试集
    size_t train_size = static_cast<size_t>(lines.size() * train_ratio);
    std::vector<std::string> train_lines(lines.begin(), lines.begin() + train_size);
    std::vector<std::string> test_lines(lines.begin() + train_size, lines.end());

    // 保存训练集到文件
    std::ofstream trainFile(trainFilePath);
    if (trainFile.is_open()) {
        trainFile << header << std::endl;
        for (const auto& train_line : train_lines) {
            trainFile << train_line << std::endl;
        }
        trainFile.close();
    } else {
        std::cerr << "无法创建训练集文件: " << trainFilePath << std::endl;
    }

    // 保存测试集到文件
    std::ofstream testFile(testFilePath);
    if (testFile.is_open()) {
        testFile << header << std::endl;
        for (const auto& test_line : test_lines) {
            testFile << test_line << std::endl;
        }
        testFile.close();
    } else {
        std::cerr << "无法创建测试集文件: " << testFilePath << std::endl;
    }
}

// 从 CSV 文件中读取训练数据集
bool loadData(const std::string& filePath, std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& y) {
    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        std::cerr << "无法打开文件 " << filePath << std::endl;
        return false;
    }

    std::string line;
    bool isFirstLine = true;

    // 读取文件内容并解析
    while (std::getline(inputFile, line)) {
        if (isFirstLine) {
            // 跳过标题行
            isFirstLine = false;
            continue;
        }

        std::vector<std::string> data = parseCSVLine(line);
        if (data.size() >= 8) {
            // 提取特征：身高、体重、鞋码、50m 成绩
            std::vector<double> features = {
                    std::stod(data[3]),  // 身高
                    std::stod(data[4]),  // 体重
                    std::stod(data[5]),  // 鞋码
                    std::stod(data[6])   // 50m 成绩
            };
            X.push_back(features);

            // 提取目标输出：性别 (0 或 1)
            y.push_back({std::stod(data[1])});
        }
    }
    inputFile.close();
    return true;
}

/****************************训练********************************/
// 训练 BP 神经网络的函数
void trainBPNeuralNetwork(BPNeuralNetwork& bpnn, const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int epochs, const std::string& lossFilePath, std::vector<double>& loss_values) {
    std::ofstream lossFile(lossFilePath);
    if (!lossFile.is_open()) {
        std::cerr << "无法创建损失值文件" << std::endl;
        return;
    }
    lossFile << "Epoch,Loss" << std::endl;

    int patience = 20; // 设置耐心值（最大允许无进步轮数）
    double best_loss = std::numeric_limits<double>::max();
    int no_improve_epochs = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < X.size(); ++i) {
            // 前向传播
            std::vector<double> prediction = bpnn.forward(X[i]);
            // 计算损失
            double error = y[i][0] - prediction[0];
            total_loss += error * error;
            // 设置类别权重
            double class_weight = (y[i][0] == 1) ? 0.5 : 0.5;
            // 反向传播
            bpnn.backward(y[i],class_weight);
        }
        total_loss /= X.size();
        loss_values.push_back(total_loss);
        if (epoch % 100 == 0) {
            std::cout << "第 " << epoch << " 轮, 损失: " << total_loss << std::endl;
        }
        lossFile << epoch << "," << total_loss << std::endl;

//        if (total_loss < best_loss) {
//            best_loss = total_loss;
//            no_improve_epochs = 0;
//        } else {
//            no_improve_epochs++;
//        }
//
//        if (no_improve_epochs >= patience) {
//            std::cout << "早停于第 " << epoch << " 轮" << std::endl;
//            break;
//        }
    }
    lossFile.close();
}

void printConfusionMatrix(int tp, int tn, int fp, int fn) {
    std::cout << "混淆矩阵：" << std::endl;
    std::cout << "TP: " << tp << ", FN: " << fn << std::endl; // 真阳性与假阴性
    std::cout << "FP: " << fp << ", TN: " << tn << std::endl; // 假阳性与真阴性
}

/****************************模型评估********************************/
// 使用测试集评估 BP 神经网络，并保存错分样本
double evaluateBPNeuralNetwork(BPNeuralNetwork& bpnn, const std::vector<std::vector<double>>& X_test, const std::vector<std::vector<double>>& y_test, const std::string& misclassifiedFilePath) {
    int correct_predictions = 0;
    std::ofstream misclassifiedFile(misclassifiedFilePath);
    if (!misclassifiedFile.is_open()) {
        std::cerr << "无法创建错分样本文件" << std::endl;
        return 0.0;
    }
    misclassifiedFile << "Features,Actual,Predicted" << std::endl;

    for (size_t i = 0; i < X_test.size(); ++i) {
        std::vector<double> prediction = bpnn.forward(X_test[i]);
        int predicted_label = (prediction[0] >= 0.5) ? 1 : 0;
        int actual_label = static_cast<int>(y_test[i][0]);
        if (predicted_label == actual_label) {
            ++correct_predictions;
        } else {
            // 保存错分样本
            misclassifiedFile << "\"";
            for (size_t j = 0; j < X_test[i].size(); ++j) {
                misclassifiedFile << X_test[i][j];
                if (j < X_test[i].size() - 1) {
                    misclassifiedFile << ",";
                }
            }
            misclassifiedFile << "\"," << actual_label << "," << predicted_label << "\n";
        }
    }
    misclassifiedFile.close();

    double accuracy = static_cast<double>(correct_predictions) / X_test.size();
    std::cout << "模型在测试集上的准确率: " << accuracy * 100 << "%" << std::endl;
    return accuracy;
}

// 交叉验证（不包含训练过程）
void crossValidation(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y,BPNeuralNetwork& bpnn,const std::string& metricsFilePath) {
    double total_se = 0.0;
    double total_sp = 0.0;
    double total_acc = 0.0;
    double total_auc = 0.0;

    // 打开 CSV 文件以保存性能指标
    std::ofstream metricsFile(metricsFilePath, std::ios::app);
    if (!metricsFile.is_open()) {
        std::cerr << "无法打开性能指标文件: " << metricsFilePath << std::endl;
        return;
    }
    metricsFile << "Accuracy,Sensitivity,Specificity,AUC,TP,FP,TN,FN" << std::endl;


    // 评估模型性能
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<double> prediction = bpnn.forward(X[i]);
            int predicted_label = (prediction[0] >= 0.5) ? 1 : 0;
            int actual_label = static_cast<int>(y[i][0]);

            if (predicted_label == 1 && actual_label == 1) {
                tp++;
            } else if (predicted_label == 0 && actual_label == 0) {
                tn++;
            } else if (predicted_label == 1 && actual_label == 0) {
                fp++;
            } else if (predicted_label == 0 && actual_label == 1) {
                fn++;
            }
        }

        printConfusionMatrix(tp, tn, fp, fn);
        // 计算性能指标
        double se = static_cast<double>(tp) / (tp + fn); // Sensitivity
        double sp = static_cast<double>(tn) / (tn + fp); // Specificity
        double acc = static_cast<double>(tp + tn) / (tp + tn + fp + fn); // Accuracy
        double auc = (se + sp) / 2.0; // 近似计算 AUC

        total_se += se;
        total_sp += sp;
        total_acc += acc;
        total_auc += auc;

    // 输出平均性能指标
    std::cout << "SE: " << total_se << std::endl;
    std::cout << "SP: " << total_sp << std::endl;
    std::cout << "ACC: " << total_acc  << std::endl;
    std::cout << "AUC: " << total_auc  << std::endl;

    // 保存每个性能指标到文件
    metricsFile << acc << "," << se << "," << sp << "," << auc << "," << tp << "," << fp << "," << tn << "," << fn << std::endl;
}

/****************************调试用********************************/
// 打印 X 和 Y 向量
void printData(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y) {
    std::cout << "X Data:" << std::endl;
    for (const auto& row : X) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Y Data:" << std::endl;
    for (const auto& val : y) {
        std::cout << val[0] << std::endl; // 假设 y 是一个二元向量
    }
}

int main() {
    /****************************路径********************************/
    std::string inputFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/data.csv";//源数据集路径
    std::string outputFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/filtered_data.csv";//过滤后的数据集路径
    std::string invalidFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/invalid_data.csv";//无效数据路径

    std::string trainFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/train.csv";//训练集路径
    std::string testFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/test.csv";//测试集路径

    std::string lossFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/loss_values.csv";//保存的损失值

    std::string misclassifiedFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/result/misclassified_samples.csv";//模型的错分样本

    std::string modelFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/result/models.txt";//模型参数
    std::string matFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/mat.csv";//评估结果的输出矩阵

    /****************************BP神经网络的参数配置********************************/
    int input_size = 4;       // 输入层节点数为 4
    int hidden_size = 4;      // 隐藏层节点数为 4
    int output_size = 1;      // 输出层节点数为 1 (性别分类：0 或 1)
    double learning_rate = 0.001;//学习率
    int epochs = 2000;//训练轮次

    //1.调用函数预处理文件
    std::cout << "*******************数据集预处理****************** " << std::endl;
    if (!processFile(inputFilePath, outputFilePath, invalidFilePath)) {
        std::cerr << "文件处理失败。" << std::endl;
        return 1;
    }
    else{
        std::cout << "数据集预处理成功！" << std::endl;
    }

    //2.划分数据集
    // 将输入文件随机划分为训练集和测试集，比例为7:3
    splitCSVFile(outputFilePath, trainFilePath, testFilePath);
    std::cout << "按照7:3比例随机划分为训练集和测试集保存到文件中。" << std::endl;

    //2.加载数据集
    // 构建训练数据集，读取 train.csv 文件
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> Y;
    if (!loadData(trainFilePath, X, Y)) {
        return 1;
    }
    //printData(X,Y);
    // 构造测试数据集，读取 test.csv 文件
    std::vector<std::vector<double>> X_test;
    std::vector<std::vector<double>> Y_test;
    if (!loadData(testFilePath, X_test, Y_test)) {
        return 1;
    }
    std::cout << "数据集加载成功！" << std::endl;
    //printData(X_test,Y_test);

    //3.初始化 BP 神经网络
    BPNeuralNetwork bpnn(input_size, hidden_size, output_size, learning_rate);
    std::cout << "BP神经网络初始化完成..." << std::endl;

    //4.训练 BP 神经网络，并记录损失值到文件中,保存模型参数
    std::vector<double> loss_values;
    std::cout << "******************开始训练bp神经网络****************** " << std::endl;
    trainBPNeuralNetwork(bpnn, X, Y, epochs, lossFilePath, loss_values);
    // 保存模型参数
    bpnn.saveModel(modelFilePath);

    //5.根据测试集计算模型准确率
    std::cout << "*****************模型性能交叉验证评估******************" << std::endl;
    evaluateBPNeuralNetwork(bpnn, X_test, Y_test,misclassifiedFilePath);

    //6 使用交叉验证评估模型
    crossValidation(X_test, Y_test, bpnn,matFilePath);

    return 0;
}
