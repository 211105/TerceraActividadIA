import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# Leer tus datos
df = pd.read_csv('Nose/211105.csv', delimiter=';')

# En tu caso, parece que las columnas 'x1', 'x2', 'x3' son tus características y 'y' es tu objetivo
X = df[['x1', 'x2', 'x3']]
y = df['y']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Listas para almacenar los errores
errors = []

# Rango de valores de K que vamos a explorar (Usar K > 3)
K_range = list(range(4, 21))

# Probar varios valores de K
for K in K_range:
    model = KNeighborsRegressor(n_neighbors=K)
    model.fit(X_train, y_train)

    # Predecir y calcular el error
    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    errors.append(error)

    print(f"K={K}, MSE={error}")

# Graficar la evolución del error
plt.plot(K_range, errors)
plt.xlabel('K')
plt.ylabel('Error cuadrático medio')
plt.title('Evolución del error con diferentes valores de K')
plt.show()

# Encontrar el modelo con el menor error
best_K = K_range[errors.index(min(errors))]
print(f"El mejor modelo es con K={best_K}")
