import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

x_vals = np.linspace(0, 6, 200)
y_vals = np.sin(x_vals) + np.sin(6 * x_vals) + np.random.default_rng(seed=1).normal(0, 0.1, x_vals.shape[0])

X_train, X_test, Y_train, Y_test = train_test_split(x_vals.reshape(-1, 1), y_vals, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'hidden_layer_sizes': [(40,), (80,), (40, 40), (80, 80)],
    'activation': ['relu', 'logistic'],
    'solver': ['adam', 'lbfgs'],
    'learning_rate_init': [0.001, 0.005, 0.01],
    'max_iter': [1000, 2000]
}

grid_search = GridSearchCV(
    estimator=MLPRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1
)

grid_search.fit(X_train_scaled, Y_train)
best_params = grid_search.best_params_

model = MLPRegressor(
    **best_params,
    max_iter=5000,
    random_state=42
).fit(X_train_scaled, Y_train)

train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

train_mse, train_r2 = mean_squared_error(Y_train, train_pred), r2_score(Y_train, train_pred)
test_mse, test_r2 = mean_squared_error(Y_test, test_pred), r2_score(Y_test, test_pred)

print(f'Оптимальні гіперпараметри: {best_params}')
print(f'Помилка на тренуванні (MSE): {train_mse}, R2: {train_r2}')
print(f'Помилка на тесті (MSE): {test_mse}, R2: {test_r2}')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train, Y_train, color='purple', label='Train Data')
plt.plot(X_train, train_pred, color='orange', label='Prediction')
plt.title('Тренувальні дані vs Прогноз')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test, Y_test, color='blue', label='Test Data')
plt.plot(X_test, test_pred, color='red', label='Prediction')
plt.title('Тестові дані vs Прогноз')
plt.legend()

plt.tight_layout()
plt.show()
