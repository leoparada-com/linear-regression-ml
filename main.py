import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos de ejemplo
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Características (variable independiente)
y = np.array([2, 4, 5, 4, 5])  # Variable objetivo

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Predecir valores
X_new = np.array([[6]])  # Nuevas características para predecir
predicted_y = model.predict(X_new)

# Visualizar los datos y la línea de regresión
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, model.predict(X), color='red', label='Regresión lineal')
plt.scatter(X_new, predicted_y, color='green', label='Predicción para X_new')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresión Lineal Simple')
plt.legend()
plt.show()
