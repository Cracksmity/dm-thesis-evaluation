"""Ejemplo didáctico: MLP con el dataset Iris (no forma parte del pipeline EOG)."""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# 1. Cargar datos
# =========================
iris = load_iris()
X = iris.data
y = iris.target

# =========================
# 2. Dividir entrenamiento y prueba
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================
# 3. Normalizar datos
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 4. Crear la red neuronal
# =========================
modelo = MLPClassifier(
    hidden_layer_sizes=(8, 6),  # dos capas ocultas: 8 y 6 neuronas
    activation='relu',
    solver='adam',
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42
)

# =========================
# 5. Entrenar
# =========================
modelo.fit(X_train, y_train)

# =========================
# 6. Predecir
# =========================
y_pred = modelo.predict(X_test)

# =========================
# 7. Evaluar
# =========================
print("Exactitud:", accuracy_score(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# =========================
# 8. Graficar curva de pérdida
# =========================
plt.figure(figsize=(8, 5))
plt.plot(modelo.loss_curve_)
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.title("Curva de pérdida durante el entrenamiento")
plt.grid(True)
plt.show()