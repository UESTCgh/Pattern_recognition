import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw

# 加载数据
train_data = sio.loadmat('../data/train_data.mat')
train_label = sio.loadmat('../data/train_label.mat')
test_data = sio.loadmat('../data/test_data.mat')

# 提取数据和标签
data = train_data['train_data']
labels = train_label['train_label']
test_data = test_data['test_data']

# 数据标准化到均值为0，方差为1
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data - mean) / (std + 1e-6)  # 避免除以零的问题
data_reshaped = data.reshape(-1, 19, 19, 1)

test_data = (test_data - mean) / (std + 1e-6)
test_data_reshaped = test_data.reshape(-1, 19, 19, 1)

# 将标签转换为适合分类的格式
labels = labels.astype(np.float32)
labels = np.clip(labels, 0, 1)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(data_reshaped, labels, test_size=0.2, random_state=42)

# 数据增强
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,  # 限制旋转角度以防止小尺寸图像失真过大
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,  # 小尺寸图像水平翻转可能有效
    zoom_range=0.1,
    shear_range=0.1
)

# 构建优化后的CNN模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(19, 19, 1), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 动态学习率调度器
def scheduler(epoch, lr):
    if epoch < 15:
        return lr
    else:
        return lr * 0.95

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 设置早停机制
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# 训练模型并记录历史
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
                    epochs=100,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, lr_scheduler])

# 打印模型的结构和评估结果
model.summary()
from sklearn.metrics import classification_report
print('Classification Report on Validation Set:')
class_report = classification_report(y_val, (model.predict(X_val) > 0.5).astype(int), target_names=['Not Face', 'Face'])
print(class_report)

# 保存测试集预测结果到文本文件
def store_txt(labels, result_txt):
    with open(result_txt, "w") as w:
        for index, result in enumerate(labels, 1):
            label = -1 if result == 0 else 1
            w.write(str(index) + " " + str(label) + "\n")

predicted_labels = (model.predict(test_data_reshaped) > 0.5).astype(int)
store_txt(predicted_labels, '../data/task3_result.txt')

# 将测试集转换为图像并贴标签输出
def save_labeled_images(test_data, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, (image, label) in enumerate(zip(test_data, labels)):
        img_array = (image.reshape(19, 19) * 255).astype(np.uint8)
        img = Image.fromarray(img_array, 'L')
        draw = ImageDraw.Draw(img)
        label_text = 'Face' if label == 1 else 'Not Face'
        draw.text((2, 2), label_text, fill='white', anchor='lt')  # 标签颜色设为白色，大小更小，且放在左上角
        img.save(os.path.join(output_dir, f'test_image_{i + 1}.png'))

save_labeled_images(test_data_reshaped, predicted_labels, '../data/test_images_with_labels')

# 绘制训练和验证的损失曲线
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 绘制训练和验证的准确率曲线
plt.figure(figsize=(12, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
