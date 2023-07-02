import csv
import numpy as np
import random
import tkinter as tk
from tkinter import ttk
from sklearn.metrics import mean_squared_error

k = 3  # Define k as 3

datos_x = []
datos_y = []

with open('TerceraActividad/211105.CSV', 'r') as archivo_csv:
    lector_csv = csv.reader(archivo_csv, delimiter=';')
    next(lector_csv)
    for fila in lector_csv:
        valores_x = [float(i) for i in fila[0:3]]
        valores_x.insert(0, 1)
        valor_y = float(fila[3])
        datos_x.append(valores_x)
        datos_y.append(valor_y)

def obtener_winicial(matriz):
    w = np.array([], dtype=float)
    for i in range(int(len(matriz[0]))):
        num = random.randint(-5, 5)
        while num == 0:
            num = random.randint(-5, 5)
        w = np.append(w, num)
    return w

def funcion_activacion(u):
    if u >= 0:
        return 1
    else:
        return 0

def obtener_u(matriz, w):
    u = np.dot(matriz, w)
    return u

def obtener_e(yd, yc):
    e = yd - yc
    return e

def obtener_deltaw(matriz, e, tasa_aprendizaje):
    deltaW = tasa_aprendizaje * np.dot(matriz.T, e)
    return deltaW

def obtener_nuevaw(w, delta_w):
    nueva_w = w + delta_w
    return nueva_w

def obtener_tolerancia_error(e):
    tolerancia = np.sum(np.abs(e))
    return tolerancia

def calcular_mse(yd, yc):
    return mean_squared_error(yd, yc)

def main():
    tolerancia_error = 0
    almacen_w = np.array([])
    almacen_tolerancia_error = np.array([])
    matriz = np.array(datos_x, dtype=float)
    yd = np.array(datos_y, dtype=float)
    w = obtener_winicial(matriz)
    tasa_aprendizaje = 0.00002
    margen_error = 0.5

    root = tk.Tk()
    root.title("Configuración")
    root.geometry("250x100")

    def start_algorithm():
        nonlocal tasa_aprendizaje
        tasa_aprendizaje = float(entry.get())
        root.destroy()

    label = tk.Label(root, text="Tasa de Aprendizaje:")
    label.pack()
    entry = tk.Entry(root, width=20)
    entry.insert(tk.END, "{:.8f}".format(tasa_aprendizaje))
    entry.pack()
    button = tk.Button(root, text="Iniciar", command=start_algorithm)
    button.pack()
    root.mainloop()
    cantidad_iteraciones = 10000

    # Crear un diccionario para almacenar los MSE de los modelos
    modelos_mse = {}

    # Crear múltiples modelos (supongamos K = k)
    for modelo in range(1, k + 1):
        print(f"\nModelo {modelo}:")

        w = obtener_winicial(matriz)
        print("Pesos Iniciales:")
        print(w)

        for i in range(cantidad_iteraciones):
            u = obtener_u(matriz, w)
            yc = np.array([funcion_activacion(x) for x in u])
            e = obtener_e(yd, yc)
            tolerancia = obtener_tolerancia_error(e)
            delta_w = obtener_deltaw(matriz, e, tasa_aprendizaje)
            w = obtener_nuevaw(w, delta_w)
            if tolerancia <= tolerancia_error:
                break

        print("Evolución del error:")
        print(tolerancia)

        print("Pesos Finales:")
        print(w)

        mse = calcular_mse(yd, yc)
        modelos_mse[f"Modelo {modelo}"] = mse
        print("MSE:")
        print(mse)

    # Encontrar el mejor modelo
    mejor_modelo = min(modelos_mse, key=modelos_mse.get)
    print("\nMejor Modelo:")
    print(mejor_modelo)
    print("MSE del mejor modelo:")
    print(modelos_mse[mejor_modelo])

    root = tk.Tk()
    root.title("Resultados")
    treeview = ttk.Treeview(root)
    treeview["columns"] = ("Pesos Iniciales", "Tolerancia", "Pesos Finales")
    treeview.column("#0", width=0, stretch=tk.NO)
    treeview.column("Pesos Iniciales", anchor=tk.W, width=150)
    treeview.column("Tolerancia", anchor=tk.CENTER, width=150)
    treeview.column("Pesos Finales", anchor=tk.W, width=150)
    treeview.heading("#0", text="")
    treeview.heading("Pesos Iniciales", text="Pesos Iniciales")
    treeview.heading("Tolerancia", text="Tolerancia")
    treeview.heading("Pesos Finales", text="Pesos Finales")
    treeview.insert("", tk.END, text="", values=(str(w), str(tolerancia), str(w)))
    treeview.pack()
    root.mainloop()

if __name__ == "__main__":
    main()
