import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Завантаження даних
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Підготовка даних: згортання в 1D масиви та нормалізація
train_X = train_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0
train_X = train_X.reshape(-1, 784)
test_X = test_X.reshape(-1, 784)

# One-hot кодування міток
train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)
test_y = tf.keras.utils.to_categorical(test_y, num_classes=10)

# Конфігурація моделі
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=784))
model.add(Dense(300, activation='relu'))  # Внутрішній шар із 300 нейронів
model.add(Dense(10, activation='softmax'))  # Вихідний шар

# Налаштування моделі
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Навчання моделі
history = model.fit(
    x=train_X,
    y=train_y,
    epochs=10,  # Кількість епох
    batch_size=128,  # Розмір пакету
    validation_split=0.15  # Використання частини тренувальних даних для валідації
)

# Оцінка точності на тестових даних
loss, accuracy = model.evaluate(test_X, test_y)
print(f"Точність на тестових даних: {accuracy:.4f}")

# Прогнози на тестових зображеннях
predicted_y = model.predict(test_X)

# Функція для візуалізації зображень із мітками
def visualize_images(image_data, true_labels, predicted_probs, count=6):
    plt.figure(figsize=(12, 4))
    for idx in range(count):
        plt.subplot(1, count, idx + 1)
        plt.imshow(image_data[idx].reshape(28, 28), cmap='gray')
        actual = np.argmax(true_labels[idx])
        predicted = np.argmax(predicted_probs[idx])
        plt.title(f"Істинне: {actual}\nПередбачене: {predicted}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Відображення прикладів із передбаченнями
visualize_images(test_X, test_y, predicted_y)
