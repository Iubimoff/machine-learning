import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# Завантаження та попередня обробка даних
(data_train, labels_train), (data_test, labels_test) = cifar10.load_data()
data_train, data_test = data_train.astype('float32') / 255, data_test.astype('float32') / 255  # Масштабування
labels_train = tf.keras.utils.to_categorical(labels_train, num_classes=10)
labels_test = tf.keras.utils.to_categorical(labels_test, num_classes=10)

# Побудова моделі VGG-13
vgg13_model = Sequential()

# Блок 1
vgg13_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
vgg13_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
vgg13_model.add(MaxPooling2D(pool_size=(2, 2)))

# Блок 2
vgg13_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
vgg13_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
vgg13_model.add(MaxPooling2D(pool_size=(2, 2)))

# Блок 3
vgg13_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
vgg13_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
vgg13_model.add(MaxPooling2D(pool_size=(2, 2)))

# Блок 4
vgg13_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
vgg13_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
vgg13_model.add(MaxPooling2D(pool_size=(2, 2)))

# Блок 5
vgg13_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
vgg13_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
vgg13_model.add(MaxPooling2D(pool_size=(2, 2)))

# Повнозв'язні шари
vgg13_model.add(Flatten())
vgg13_model.add(Dense(4096, activation='relu'))
vgg13_model.add(Dense(4096, activation='relu'))
vgg13_model.add(Dense(10, activation='softmax'))

# Налаштування моделі
vgg13_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Навчання
training_history = vgg13_model.fit(data_train, labels_train, batch_size=64, epochs=10, validation_data=(data_test, labels_test))

# Тестування
loss, accuracy = vgg13_model.evaluate(data_test, labels_test)
print(f"Accuracy on test set: {accuracy:.4f}")

# Функція для відображення результатів
def plot_predictions(num_samples=5):
    sample_indices = np.random.choice(data_test.shape[0], num_samples, replace=False)
    sample_images = data_test[sample_indices]
    sample_labels = labels_test[sample_indices]
    predicted_probs = vgg13_model.predict(sample_images)

    plt.figure(figsize=(12, 4))
    for idx in range(num_samples):
        plt.subplot(1, num_samples, idx + 1)
        plt.imshow(sample_images[idx])
        true_class = np.argmax(sample_labels[idx])
        predicted_class = np.argmax(predicted_probs[idx])
        plt.title(f"True: {true_class}, Pred: {predicted_class}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_predictions()
