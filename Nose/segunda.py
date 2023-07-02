import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import tkinter as tk

# Leer los datos del CSV
df = pd.read_csv('Nose/211105.csv', sep=';')

# Eliminar los espacios en blanco de los nombres de las columnas
df.columns = df.columns.str.strip()

# Las columnas de entrada son 'x1', 'x2' y 'x3', y la columna de salida es 'y'
X = df[['x1', 'x2', 'x3']].values
y = df['y'].values

# Contar e imprimir los valores de X
num_values_X = np.prod(X.shape)
print(f'Hay {num_values_X} valores en X. Los valores son:')
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        print(f'Valor en la fila {i + 1}, columna {j + 1}: {X[i, j]}')

# Contar e imprimir los valores de y
num_values_y = len(y)
print(f'\nHay {num_values_y} valores en y. Los valores son:')
for i in range(len(y)):
    print(f'Valor en la fila {i + 1}: {y[i]}')
# Definir el número de folds
k = 5

# Definir el modelo de red neuronal
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Crear el validador cruzado
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Para almacenar los resultados de cada fold
results = []

for train_index, val_index in kf.split(X):
    # Crear un nuevo modelo para este fold
    model = create_model()

    # Entrenar el modelo
    model.fit(X[train_index], y[train_index], epochs=50, verbose=0)

    # Obtener las predicciones del modelo
    y_pred = model.predict(X[val_index]).flatten()

    # Almacenar los valores reales y predichos
    results.append((y[val_index], y_pred))

# Crear ventana de tkinter
root = tk.Tk()

# Crear tabla
table = tk.Frame(root)
table.pack()

# Añadir encabezados de la tabla
tk.Label(table, text="Valor real").grid(row=0, column=0)
tk.Label(table, text="Valor predicho").grid(row=0, column=1)

# Añadir los valores reales y predichos a la tabla
row_counter = 1
for y_real, y_pred in results:
    for j in range(len(y_real)):
        tk.Label(table, text=str(y_real[j])).grid(row=row_counter, column=0)
        tk.Label(table, text=str(y_pred[j])).grid(row=row_counter, column=1)
        row_counter += 1

root.mainloop()
