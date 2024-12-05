import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Завантаження та обробка даних
(data_train, labels_train), (data_test, labels_test) = mnist.load_data()

# Нормалізація та зміна розмірності
data_train = data_train[..., np.newaxis].astype('float32') / 255
data_test = data_test[..., np.newaxis].astype('float32') / 255

# Приведення міток до one-hot кодування
labels_train = tf.keras.utils.to_categorical(labels_train, num_classes=10)
labels_test = tf.keras.utils.to_categorical(labels_test, num_classes=10)

# Побудова архітектури LeNet-5
model = Sequential([
    Conv2D(filters=6, kernel_size=5, activation='tanh', padding='same', input_shape=(28, 28, 1)),
    AveragePooling2D(pool_size=2),
    Conv2D(filters=16, kernel_size=5, activation='tanh'),
    AveragePooling2D(pool_size=2),
    Conv2D(filters=120, kernel_size=5, activation='tanh'),
    Flatten(),
    Dense(units=84, activation='tanh'),
    Dense(units=10, activation='softmax')
])

# Налаштування моделі
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Навчання моделі
model.fit(data_train, labels_train, epochs=10, batch_size=128, validation_split=0.1)

# Оцінка на тестових даних
loss, accuracy = model.evaluate(data_test, labels_test)
print(f"Точність моделі на тестовому наборі: {accuracy:.4f}")

# Передбачення
predictions = model.predict(data_test)

# Функція для відображення прикладів зображень
def visualize_samples(images, labels_true, labels_predicted, num_samples=5):
    plt.figure(figsize=(12, 5))
    for idx in range(num_samples):
        plt.subplot(1, num_samples, idx + 1)
        img = images[idx].reshape(28, 28)
        true_class = np.argmax(labels_true[idx])
        pred_class = np.argmax(labels_predicted[idx])
        plt.imshow(img, cmap='gray')
        plt.title(f"Істинне: {true_class}\nПередбачене: {pred_class}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Візуалізація результатів на тестових даних
visualize_samples(data_test, labels_test, predictions)
