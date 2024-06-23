import numpy as np

# Задание 1
print('Выполнение задания 1:')

arr1 = np.arange(1, 7)
arr2 = np.arange(7, 13)

add = np.add(arr1, arr2)
print(f'Сложение: {add}')

sub = np.subtract(arr1, arr2)
print(f'Вычитание: {sub}')

mul = np.multiply(arr1, arr2)
print(f'Умножение: {mul}')

div = np.divide(arr1, arr2)
print(f'Деление: {div}')

concat_arr = np.r_[arr1, arr2]
print(f'Результат конкатенации: {concat_arr}')

max_elem = concat_arr.max()
print(f'Максимальный элемент: {max_elem}')

min_elem = concat_arr.min()
print(f'Минимальный элемент: {min_elem}')

sum_elems = concat_arr.sum()
print(f'Сумма элементов: {sum_elems}')

prod_elems = concat_arr.prod()
print(f'Произведение элементов: {prod_elems}')

# Задание 2
print('Выполнение задания 2:')

arr3 = np.array([5, 4, 1, 67, 32, 6, 16, 11, 9, 10, 9, 3, 13, 2, 15])

mean_val = arr3.mean()
adjusted_arr = arr3 - mean_val
print(f'Массив с коррекцией: {adjusted_arr}')

sorted_arr = np.sort(adjusted_arr)
print(f'Отсортированный массив по возрастанию: {sorted_arr}')

# Задание 3
print('Выполнение задания 3:')

random_arr = np.random.random(20)
print(f'Одномерный массив: {random_arr}')

reshaped_arr = random_arr.reshape((4, 5))
reshaped_arr += 10
print(f'Двумерный массив с увеличенными элементами: \n{reshaped_arr}')

# Задание 4
print('Выполнение задания 4:')

arr4 = np.random.randint(-15, 16, (5, 5))
print(f'Исходный массив: \n{arr4}')

arr4 = np.where(arr4 < 0, -1, np.where(arr4 > 0, 1, 0))
print(f'Массив после замены: \n{arr4}')

# Задание 5
print('Выполнение задания 5:')

matrix_A = np.array([[2, 3, -1], [4, 5, 2], [-1, 0, 7]])
matrix_B = np.array([[-1, 0, 5], [0, 1, 3], [2, -2, 4]])

operation_result = 2 * (matrix_A + matrix_B) * (2 * matrix_B - matrix_A)
print(f'Результат операций:\n {operation_result}')

# Задание 6
print('Выполнение задания 6:')

matrix_C = np.array([[1, 1, 2, 3],
                     [3, -1, -2, -2],
                     [2, -3, -1, -1],
                     [1, 2, 3, -1]])

vector_D = np.array([1, -4, -6, -4])

solution_vector = np.linalg.solve(matrix_C, vector_D)
print(f'Решение системы уравнений: {solution_vector}')
