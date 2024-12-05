import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# Завантажуємо та нормалізуємо дані CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes=10), tf.keras.utils.to_categorical(y_test, num_classes=10)

# Визначення Inception модуля
def create_inception_block(input_tensor, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    conv_1x1 = layers.Conv2D(filters_1x1, (1, 1), padding="same", activation="relu")(input_tensor)
    
    conv_3x3 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding="same", activation="relu")(input_tensor)
    conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding="same", activation="relu")(conv_3x3)
    
    conv_5x5 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding="same", activation="relu")(input_tensor)
    conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding="same", activation="relu")(conv_5x5)
    
    pooling = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(input_tensor)
    pooling = layers.Conv2D(filters_pool_proj, (1, 1), padding="same", activation="relu")(pooling)

    return layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pooling])

# Побудова моделі GoogLeNet
input_layer = Input(shape=(32, 32, 3))
x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same", activation="relu")(input_layer)
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
x = layers.Conv2D(64, (1, 1), padding="same", activation="relu")(x)
x = layers.Conv2D(192, (3, 3), padding="same", activation="relu")(x)
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

# Додаємо Inception блоки
x = create_inception_block(x, 64, 96, 128, 16, 32, 32)
x = create_inception_block(x, 128, 128, 192, 32, 96, 64)
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

x = create_inception_block(x, 192, 96, 208, 16, 48, 64)
x = create_inception_block(x, 160, 112, 224, 24, 64, 64)
x = create_inception_block(x, 128, 128, 256, 24, 64, 64)
x = create_inception_block(x, 112, 144, 288, 32, 64, 64)
x = create_inception_block(x, 256, 160, 320, 32, 128, 128)
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

x = create_inception_block(x, 256, 160, 320, 32, 128, 128)
x = create_inception_block(x, 384, 192, 384, 48, 128, 128)

# Завершуємо модель
x = layers.GlobalAveragePooling2D()(x)
output_layer = layers.Dense(10, activation="softmax")(x)

# Створення моделі
googlenet_model = models.Model(inputs=input_layer, outputs=output_layer)
googlenet_model.summary()

# Компіляція та тренування моделі
googlenet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = googlenet_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Оцінка точності моделі
test_loss, test_acc = googlenet_model.evaluate(x_test, y_test, verbose=0)
print(f"Тестова точність: {test_acc * 100:.2f}%")

# Прогнозування та візуалізація результатів
predictions = np.argmax(googlenet_model.predict(x_test[:10]), axis=1)
true_labels = np.argmax(y_test[:10], axis=1)

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"True: {true_labels[i]}\nPred: {predictions[i]}")
    plt.axis("off")
plt.show()
