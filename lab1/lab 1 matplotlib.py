import numpy as np
import matplotlib.pyplot as plt
import re

# Задание 1
print("Выполнение задания 1:")

# Вариант 4 Y(x)=x^(1/2)sin(10x), x=[0...5]
x_vals = np.linspace(0, 5, 1000)
y_vals = np.sqrt(x_vals) * np.sin(10 * x_vals)

plt.plot(x_vals, y_vals, label="Y(x)=√x * sin(10x)")
plt.title("График функции Y(x)=√x * sin(10x)")
plt.xlabel("x")
plt.ylabel("Y(x)")
plt.legend()
plt.grid(True)
plt.show()

# Задание 2
print("Выполнение задания 2:")

with open('text.txt', 'r') as file:
    content = file.read()

char_frequency = {}
for char in content:
    if char.isalpha():
        char = char.lower()
        char_frequency[char] = char_frequency.get(char, 0) + 1

letters = np.array(list(char_frequency.keys()))
frequencies = np.array(list(char_frequency.values()))

plt.bar(letters, frequencies, color='skyblue')
plt.xlabel('Буквы')
plt.ylabel('Частота появления')
plt.title('Гистограмма частоты появления букв в тексте')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Задание 3
print("Выполнение задания 3:")

with open('text.txt', 'r') as file:
    content = file.read()

ordinary_sentences = content.count('.') - content.count('...')
question_sentences = content.count('?')
exclamatory_sentences = content.count('!')
ellipsis_sentences = len(re.findall(r'\.\.\.(?!\w)', content))

sentence_types = ['Обычные', 'Вопросительные', 'Окличные', 'Триточка']
sentence_frequencies = [ordinary_sentences, question_sentences, exclamatory_sentences, ellipsis_sentences]

plt.bar(sentence_types, sentence_frequencies, color='lightgreen')
plt.xlabel('Типы предложений')
plt.ylabel('Частота')
plt.title('Гистограмма частоты появления типов предложений')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
