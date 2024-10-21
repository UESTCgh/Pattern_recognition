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

void printConfusionMatrix(int tp, int tn, int fp, int fn) {
    std::cout << "混淆矩阵：" << std::endl;
    std::cout << "TP: " << tp << ", FN: " << fn << std::endl; // 真阳性与假阴性
    std::cout << "FP: " << fp << ", TN: " << tn << std::endl; // 假阳性与真阴性
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
    std::string inputFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/data.csv";//源数据集路径
    std::string outputFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/filtered_data.csv";//过滤后的数据集路径
    std::string invalidFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/invalid_data.csv";//无效数据路径

    std::string trainFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/train.csv";//训练集路径
    std::string testFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/test.csv";//测试集路径

    std::string misclassifiedFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/result/misclassified_samples.csv";//模型的错分样本

    std::string modelFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/result/models.txt";//模型参数
    std::string matFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/mat.csv";//评估结果的输出矩阵

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


    return 0;
}
