# Importamos las librerías necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def crear_modelo():
    modelo = Sequential()
    modelo.add(Dense(1, input_dim=3, activation='relu')) 
    modelo.add(Dense(1, activation='linear')) 
    modelo.compile(loss='mean_squared_error', optimizer='adam')
    return modelo

datos = pd.read_csv('Nose/211105.csv', sep=';')

print(datos.head())

datos.columns = datos.columns.str.replace(' ', '')

entradas = datos[['x1', 'x2', 'x3']].values
salidas = datos['y'].values

modelo = KerasRegressor(build_fn=crear_modelo, epochs=50, batch_size=1000, verbose=0)

# Definimos k > 3
validacion_cruzada = KFold(n_splits=3)

errores = []

for indices_entrenamiento, indices_prueba in validacion_cruzada.split(entradas, salidas):
    modelo.fit(entradas[indices_entrenamiento], salidas[indices_entrenamiento])
    prediccion = modelo.predict(entradas[indices_prueba])
    error = mean_squared_error(salidas[indices_prueba], prediccion)
    errores.append(error)
    print(f'MSE: {error}')

indice_mejor_modelo = np.argmin(errores)
print(f'El mejor modelo es el modelo {indice_mejor_modelo+1} con un MSE de {errores[indice_mejor_modelo]}')
