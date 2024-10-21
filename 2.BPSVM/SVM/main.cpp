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

/****************************���ݹ��˷�Χ********************************/
const double MIN_HEIGHT = 140.0;//���
const double MAX_HEIGHT = 220.0;
const double MIN_WEIGHT = 30.0;//����
const double MAX_WEIGHT = 150.0;
const double MIN_SHOE_SIZE = 30.0;//Ь��
const double MAX_SHOE_SIZE = 50.0;
const double MIN_50M_TIME = 5.0;//50m
const double MAX_50M_TIME = 15.0;
const double MIN_VITAL_CAPACITY = 1000.0;//�λ���
const double MAX_VITAL_CAPACITY = 8000.0;

/****************************����Ԥ����********************************/
// ��������Ƿ��ں���Χ��
bool isValidData(const std::vector<std::string> &data) {
    try {
        // ����Ƿ��пհ�����
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
        // ���ת��ʧ�ܣ�������Ч
        return false;
    }
}

// ����CSV��
std::vector<std::string> parseCSVLine(const std::string &line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ',')) {
        result.push_back(item);
    }
    return result;
}

// ���ļ�����������
bool processFile(const std::string& inputFilePath, const std::string& outputFilePath, const std::string& invalidFilePath) {
    if (!fs::exists(inputFilePath)) {
        std::cerr << "���������ļ������ڣ�" << inputFilePath << std::endl;
        return false;
    }

    std::wifstream inputFile(inputFilePath);
    inputFile.imbue(std::locale(inputFile.getloc(), new std::codecvt_utf8<wchar_t>));
    std::wofstream outputFile(outputFilePath, std::ios::out | std::ios::binary);
    outputFile.imbue(std::locale(outputFile.getloc(), new std::codecvt_utf8<wchar_t>));
    std::wofstream invalidFile(invalidFilePath, std::ios::out | std::ios::binary);
    invalidFile.imbue(std::locale(invalidFile.getloc(), new std::codecvt_utf8<wchar_t>));

    if (!inputFile.is_open()) {
        std::cerr << "�����ļ�����ʹ��" << inputFilePath << std::endl;
        return false;
    }

    if (!outputFile.is_open()) {
        std::cerr << "�޷���������ļ�" << outputFilePath << std::endl;
        return false;
    }

    if (!invalidFile.is_open()) {
        std::cerr << "�޷�������Ч�����ļ�" << invalidFilePath << std::endl;
        return false;
    }

    std::wstring line;
    bool isFirstLine = true;

    while (std::getline(inputFile, line)) {
        // д�������
        if (isFirstLine) {
            outputFile << line << std::endl;
            invalidFile << line << std::endl;
            isFirstLine = false;
            continue;
        }

        // ����������
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::string utf8Line = converter.to_bytes(line);
        std::vector<std::string> data = parseCSVLine(utf8Line);

        // ������������Ժ���Ч��
        if (data.size() < 8) {
            invalidFile << line << std::endl; // �������������в���¼
            continue;
        }

        // ����Ƿ�����Ч������
        if (isValidData(data)) {
            outputFile << line << std::endl;
        } else {
            invalidFile << line << std::endl; // ��¼��Ч������
        }
    }

    inputFile.close();
    outputFile.close();
    invalidFile.close();

    std::cout << "���ݹ��˽�������� " << outputFilePath << std::endl;
    std::cout << "��Ч���ݱ����� " << invalidFilePath << std::endl;
    return true;
}

/****************************���ݼ�����********************************/
// �������������CSV�ļ�Ϊѵ�����Ͳ��Լ������浽�ļ���
void splitCSVFile(const std::string& inputFilePath, const std::string& trainFilePath, const std::string& testFilePath, double train_ratio = 0.7) {
    std::ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        std::cerr << "�޷��������ļ�: " << inputFilePath << std::endl;
        return;
    }

    std::vector<std::string> lines;
    std::string line;
    bool isFirstLine = true;
    std::string header;

    // ��ȡCSV������
    while (std::getline(inputFile, line)) {
        if (isFirstLine) {
            header = line;
            isFirstLine = false;
            continue;
        }
        lines.push_back(line);
    }
    inputFile.close();

    // �����������
    std::shuffle(lines.begin(), lines.end(), std::default_random_engine(std::random_device{}()));

    // ����ѵ�����Ͳ��Լ�
    size_t train_size = static_cast<size_t>(lines.size() * train_ratio);
    std::vector<std::string> train_lines(lines.begin(), lines.begin() + train_size);
    std::vector<std::string> test_lines(lines.begin() + train_size, lines.end());

    // ����ѵ�������ļ�
    std::ofstream trainFile(trainFilePath);
    if (trainFile.is_open()) {
        trainFile << header << std::endl;
        for (const auto& train_line : train_lines) {
            trainFile << train_line << std::endl;
        }
        trainFile.close();
    } else {
        std::cerr << "�޷�����ѵ�����ļ�: " << trainFilePath << std::endl;
    }

    // ������Լ����ļ�
    std::ofstream testFile(testFilePath);
    if (testFile.is_open()) {
        testFile << header << std::endl;
        for (const auto& test_line : test_lines) {
            testFile << test_line << std::endl;
        }
        testFile.close();
    } else {
        std::cerr << "�޷��������Լ��ļ�: " << testFilePath << std::endl;
    }
}

// �� CSV �ļ��ж�ȡѵ�����ݼ�
bool loadData(const std::string& filePath, std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& y) {
    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        std::cerr << "�޷����ļ� " << filePath << std::endl;
        return false;
    }

    std::string line;
    bool isFirstLine = true;

    // ��ȡ�ļ����ݲ�����
    while (std::getline(inputFile, line)) {
        if (isFirstLine) {
            // ����������
            isFirstLine = false;
            continue;
        }

        std::vector<std::string> data = parseCSVLine(line);
        if (data.size() >= 8) {
            // ��ȡ��������ߡ����ء�Ь�롢50m �ɼ�
            std::vector<double> features = {
                    std::stod(data[3]),  // ���
                    std::stod(data[4]),  // ����
                    std::stod(data[5]),  // Ь��
                    std::stod(data[6])   // 50m �ɼ�
            };
            X.push_back(features);

            // ��ȡĿ��������Ա� (0 �� 1)
            y.push_back({std::stod(data[1])});
        }
    }
    inputFile.close();
    return true;
}

void printConfusionMatrix(int tp, int tn, int fp, int fn) {
    std::cout << "��������" << std::endl;
    std::cout << "TP: " << tp << ", FN: " << fn << std::endl; // �������������
    std::cout << "FP: " << fp << ", TN: " << tn << std::endl; // ��������������
}

/****************************������********************************/
// ��ӡ X �� Y ����
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
        std::cout << val[0] << std::endl; // ���� y ��һ����Ԫ����
    }
}

int main() {
    /****************************·��********************************/
    std::string inputFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/data.csv";//Դ���ݼ�·��
    std::string outputFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/filtered_data.csv";//���˺�����ݼ�·��
    std::string invalidFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/invalid_data.csv";//��Ч����·��

    std::string trainFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/train.csv";//ѵ����·��
    std::string testFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/test.csv";//���Լ�·��

    std::string misclassifiedFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/result/misclassified_samples.csv";//ģ�͵Ĵ������

    std::string modelFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/result/models.txt";//ģ�Ͳ���
    std::string matFilePath = "E:/GitHub/AI_VI/2.BPSVM/SVM/data/mat.csv";//����������������

    //1.���ú���Ԥ�����ļ�
    std::cout << "*******************���ݼ�Ԥ����****************** " << std::endl;
    if (!processFile(inputFilePath, outputFilePath, invalidFilePath)) {
        std::cerr << "�ļ�����ʧ�ܡ�" << std::endl;
        return 1;
    }
    else{
        std::cout << "���ݼ�Ԥ����ɹ���" << std::endl;
    }

    //2.�������ݼ�
    // �������ļ��������Ϊѵ�����Ͳ��Լ�������Ϊ7:3
    splitCSVFile(outputFilePath, trainFilePath, testFilePath);
    std::cout << "����7:3�����������Ϊѵ�����Ͳ��Լ����浽�ļ��С�" << std::endl;

    //2.�������ݼ�
    // ����ѵ�����ݼ�����ȡ train.csv �ļ�
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> Y;
    if (!loadData(trainFilePath, X, Y)) {
        return 1;
    }
    //printData(X,Y);
    // ����������ݼ�����ȡ test.csv �ļ�
    std::vector<std::vector<double>> X_test;
    std::vector<std::vector<double>> Y_test;
    if (!loadData(testFilePath, X_test, Y_test)) {
        return 1;
    }
    std::cout << "���ݼ����سɹ���" << std::endl;
    //printData(X_test,Y_test);


    return 0;
}
