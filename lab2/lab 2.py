# Завдання 2
import pandas as pd

data = pd.read_csv('titanic.csv')


# 1
count_gender = data['sex'].value_counts()
print(f"Кількість чоловіків:{count_gender['male']}")
print(f"Кількість жінок: {count_gender['female']}")


# 2
# відсотках (число в інтервалі від 0 до 100, знак відсотка не потрібен), округливши до двох символів.
survivors = data['survived'].mean() * 100
print('Частина виживших пасажирів: {:.2f}'.format(survivors))


# 3
first_class_pass = (data['pclass'] == 1).mean() * 100
print('Частка пасажирів першого класу серед пасажирів: {:.2f}'.format(first_class_pass))


# 4
mean_age = data['age'].mean()
median_age = data['age'].median()
print(f'Середній вік та медіана віку пасажирів: {mean_age} {median_age}')