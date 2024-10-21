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

/****************************ѵ��********************************/
// ѵ�� BP ������ĺ���
void trainBPNeuralNetwork(BPNeuralNetwork& bpnn, const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int epochs, const std::string& lossFilePath, std::vector<double>& loss_values) {
    std::ofstream lossFile(lossFilePath);
    if (!lossFile.is_open()) {
        std::cerr << "�޷�������ʧֵ�ļ�" << std::endl;
        return;
    }
    lossFile << "Epoch,Loss" << std::endl;

    int patience = 20; // ��������ֵ����������޽���������
    double best_loss = std::numeric_limits<double>::max();
    int no_improve_epochs = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < X.size(); ++i) {
            // ǰ�򴫲�
            std::vector<double> prediction = bpnn.forward(X[i]);
            // ������ʧ
            double error = y[i][0] - prediction[0];
            total_loss += error * error;
            // �������Ȩ��
            double class_weight = (y[i][0] == 1) ? 0.5 : 0.5;
            // ���򴫲�
            bpnn.backward(y[i],class_weight);
        }
        total_loss /= X.size();
        loss_values.push_back(total_loss);
        if (epoch % 100 == 0) {
            std::cout << "�� " << epoch << " ��, ��ʧ: " << total_loss << std::endl;
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
//            std::cout << "��ͣ�ڵ� " << epoch << " ��" << std::endl;
//            break;
//        }
    }
    lossFile.close();
}

void printConfusionMatrix(int tp, int tn, int fp, int fn) {
    std::cout << "��������" << std::endl;
    std::cout << "TP: " << tp << ", FN: " << fn << std::endl; // �������������
    std::cout << "FP: " << fp << ", TN: " << tn << std::endl; // ��������������
}

/****************************ģ������********************************/
// ʹ�ò��Լ����� BP �����磬������������
double evaluateBPNeuralNetwork(BPNeuralNetwork& bpnn, const std::vector<std::vector<double>>& X_test, const std::vector<std::vector<double>>& y_test, const std::string& misclassifiedFilePath) {
    int correct_predictions = 0;
    std::ofstream misclassifiedFile(misclassifiedFilePath);
    if (!misclassifiedFile.is_open()) {
        std::cerr << "�޷�������������ļ�" << std::endl;
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
            // ����������
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
    std::cout << "ģ���ڲ��Լ��ϵ�׼ȷ��: " << accuracy * 100 << "%" << std::endl;
    return accuracy;
}

// ������֤��������ѵ�����̣�
void crossValidation(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y,BPNeuralNetwork& bpnn,const std::string& metricsFilePath) {
    double total_se = 0.0;
    double total_sp = 0.0;
    double total_acc = 0.0;
    double total_auc = 0.0;

    // �� CSV �ļ��Ա�������ָ��
    std::ofstream metricsFile(metricsFilePath, std::ios::app);
    if (!metricsFile.is_open()) {
        std::cerr << "�޷�������ָ���ļ�: " << metricsFilePath << std::endl;
        return;
    }
    metricsFile << "Accuracy,Sensitivity,Specificity,AUC,TP,FP,TN,FN" << std::endl;


    // ����ģ������
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
        // ��������ָ��
        double se = static_cast<double>(tp) / (tp + fn); // Sensitivity
        double sp = static_cast<double>(tn) / (tn + fp); // Specificity
        double acc = static_cast<double>(tp + tn) / (tp + tn + fp + fn); // Accuracy
        double auc = (se + sp) / 2.0; // ���Ƽ��� AUC

        total_se += se;
        total_sp += sp;
        total_acc += acc;
        total_auc += auc;

    // ���ƽ������ָ��
    std::cout << "SE: " << total_se << std::endl;
    std::cout << "SP: " << total_sp << std::endl;
    std::cout << "ACC: " << total_acc  << std::endl;
    std::cout << "AUC: " << total_auc  << std::endl;

    // ����ÿ������ָ�굽�ļ�
    metricsFile << acc << "," << se << "," << sp << "," << auc << "," << tp << "," << fp << "," << tn << "," << fn << std::endl;
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
    std::string inputFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/data.csv";//Դ���ݼ�·��
    std::string outputFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/filtered_data.csv";//���˺�����ݼ�·��
    std::string invalidFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/invalid_data.csv";//��Ч����·��

    std::string trainFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/train.csv";//ѵ����·��
    std::string testFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/test.csv";//���Լ�·��

    std::string lossFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/loss_values.csv";//�������ʧֵ

    std::string misclassifiedFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/result/misclassified_samples.csv";//ģ�͵Ĵ������

    std::string modelFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/result/models.txt";//ģ�Ͳ���
    std::string matFilePath = "E:/GitHub/AI_VI/2.BPSVM/BP/data/mat.csv";//����������������

    /****************************BP������Ĳ�������********************************/
    int input_size = 4;       // �����ڵ���Ϊ 4
    int hidden_size = 4;      // ���ز�ڵ���Ϊ 4
    int output_size = 1;      // �����ڵ���Ϊ 1 (�Ա���ࣺ0 �� 1)
    double learning_rate = 0.001;//ѧϰ��
    int epochs = 2000;//ѵ���ִ�

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

    //3.��ʼ�� BP ������
    BPNeuralNetwork bpnn(input_size, hidden_size, output_size, learning_rate);
    std::cout << "BP�������ʼ�����..." << std::endl;

    //4.ѵ�� BP �����磬����¼��ʧֵ���ļ���,����ģ�Ͳ���
    std::vector<double> loss_values;
    std::cout << "******************��ʼѵ��bp������****************** " << std::endl;
    trainBPNeuralNetwork(bpnn, X, Y, epochs, lossFilePath, loss_values);
    // ����ģ�Ͳ���
    bpnn.saveModel(modelFilePath);

    //5.���ݲ��Լ�����ģ��׼ȷ��
    std::cout << "*****************ģ�����ܽ�����֤����******************" << std::endl;
    evaluateBPNeuralNetwork(bpnn, X_test, Y_test,misclassifiedFilePath);

    //6 ʹ�ý�����֤����ģ��
    crossValidation(X_test, Y_test, bpnn,matFilePath);

    return 0;
}
