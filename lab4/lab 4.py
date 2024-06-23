import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

x_values = np.arange(0, 10, 0.1)
y_values = -0.1 * x_values**5 + x_values**4 - 700 * x_values**2

x_train = x_values[::2].reshape(-1, 1)
y_train = y_values[::2]

poly_features = PolynomialFeatures(degree=13)

linear_model = make_pipeline(poly_features, LinearRegression())
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_values.reshape(-1, 1))

plt.figure(figsize=(12, 12))
plt.scatter(x_train, y_train, label='Обучающие данные', color='red')
plt.plot(x_values, y_values, label='Исходная функция', color='green')
plt.plot(x_values, y_pred_linear, label='Полиномиальная регрессия (13 степень)', color='blue')
plt.legend()
plt.title('Полиномиальная регрессия без регуляризации')
plt.show()

mse_linear = mean_squared_error(y_values, y_pred_linear)
print(f'Среднеквадратичная ошибка (без регуляризации): {mse_linear}')

ridge_model = make_pipeline(poly_features, Ridge(alpha=1.0))
ridge_model.fit(x_train, y_train)
y_pred_ridge = ridge_model.predict(x_values.reshape(-1, 1))

plt.figure(figsize=(12, 12))
plt.scatter(x_train, y_train, label='Обучающие данные', color='red')
plt.plot(x_values, y_values, label='Исходная функция', color='green')
plt.plot(x_values, y_pred_ridge, label='Ridge регрессия (L2)', color='blue')
plt.legend()
plt.title('Полиномиальная регрессия с L2 регуляризацией')
plt.show()

mse_ridge = mean_squared_error(y_values, y_pred_ridge)
print(f'Среднеквадратичная ошибка с L2 регуляризацией: {mse_ridge}')

lasso_model = make_pipeline(poly_features, Lasso(alpha=1e-3, max_iter=10000))
lasso_model.fit(x_train, y_train)
y_pred_lasso = lasso_model.predict(x_values.reshape(-1, 1))

plt.figure(figsize=(12, 12))
plt.scatter(x_train, y_train, label='Обучающие данные', color='red')
plt.plot(x_values, y_values, label='Исходная функция', color='green')
plt.plot(x_values, y_pred_lasso, label='Lasso регрессия (L1)', color='blue')
plt.legend()
plt.title('Полиномиальная регрессия с L1 регуляризацией')
plt.show()

mse_lasso = mean_squared_error(y_values, y_pred_lasso)
print(f'Среднеквадратичная ошибка с L1 регуляризацией: {mse_lasso}')

print(f'Среднеквадратичная ошибка (без регуляризации): {mse_linear}')
print(f'Среднеквадратичная ошибка с L2 регуляризацией: {mse_ridge}')
print(f'Среднеквадратичная ошибка с L1 регуляризацией: {mse_lasso}')
