import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# Обробка даних
data_transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=False)

# Реалізація архітектури LeNet-5
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_layer2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv_layer3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc_layer1 = nn.Linear(120, 84)
        self.fc_layer2 = nn.Linear(84, 10)

    def forward(self, input_data):
        x = torch.tanh(self.conv_layer1(input_data))
        x = self.avg_pool(x)
        x = torch.tanh(self.conv_layer2(x))
        x = self.avg_pool(x)
        x = torch.tanh(self.conv_layer3(x))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc_layer1(x))
        output = self.fc_layer2(x)
        return output

# Ініціалізація моделі, функції втрат і оптимізатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lenet_model = LeNet().to(device)
loss_function = nn.CrossEntropyLoss()
adam_optimizer = optim.Adam(lenet_model.parameters(), lr=0.001)

# Навчання моделі
for epoch_num in range(10):
    total_loss = 0.0
    for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

        # Скидання градієнтів
        adam_optimizer.zero_grad()

        # Прямий прохід, обчислення втрат і зворотній прохід
        batch_outputs = lenet_model(batch_images)
        batch_loss = loss_function(batch_outputs, batch_labels)
        batch_loss.backward()

        # Оновлення ваг
        adam_optimizer.step()

        total_loss += batch_loss.item()

    print(f"Епоха {epoch_num + 1}, Втрата: {total_loss / len(train_loader):.4f}")

# Перевірка точності на тестових даних
correct_count = 0
total_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = lenet_model(images)
        _, pred_classes = torch.max(predictions, dim=1)
        total_samples += labels.size(0)
        correct_count += (pred_classes == labels).sum().item()

print(f"Точність на тестових даних: {correct_count / total_samples:.4f}")

# Функція для відображення тестових прикладів
def visualize_images_torch(batch_images, true_labels, pred_classes, count=5):
    plt.figure(figsize=(10, 4))
    for idx in range(count):
        plt.subplot(1, count, idx + 1)
        img_data = batch_images[idx].cpu().numpy().squeeze()
        actual = true_labels[idx].item()
        predicted = pred_classes[idx].item()
        plt.imshow(img_data, cmap='gray')
        plt.title(f"Істинне: {actual}\nПередбачене: {predicted}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Відображення кількох прикладів
sample_data_iter = iter(test_loader)
test_images, test_labels = next(sample_data_iter)
model_outputs = lenet_model(test_images.to(device))
_, sample_preds = torch.max(model_outputs, 1)

# Візуалізація результатів
visualize_images_torch(test_images, test_labels, sample_preds)
