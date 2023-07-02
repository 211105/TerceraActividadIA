import csv
import numpy as np
import random
import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import locale
from sklearn.metrics import confusion_matrix
import seaborn as sns  # Importa la biblioteca seaborn

# Establecer la configuración regional en inglés
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

datos_x = []
datos_y = []

with open('E1.CSV', 'r') as archivo_csv:
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
    global tolerancia
    tolerancia = np.sum(np.abs(e))
    return tolerancia

def main():
    tolerancia_error = 0
    almacen_w = np.array([])
    almacen_tolerancia_error = np.array([])
    matriz = np.array(datos_x, dtype=float)
    yd = np.array(datos_y, dtype=float)
    w = obtener_winicial(matriz)
    winicial = obtener_winicial(matriz)
    almacen_w = np.vstack([w])
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

    for i in range(cantidad_iteraciones):
        u = obtener_u(matriz, w)
        yc = np.array([funcion_activacion(x) for x in u])
        e = obtener_e(yd, yc)
        tolerancia = obtener_tolerancia_error(e)
        almacen_tolerancia_error = np.append(almacen_tolerancia_error, tolerancia)
        delta_w = obtener_deltaw(matriz, e, tasa_aprendizaje)
        w = obtener_nuevaw(w, delta_w)
        almacen_w = np.vstack([w])
        if tolerancia <= tolerancia_error:
            break

    print("Pesos Iniciales:")
    print(winicial)
    print("Tolerancia Final:")
    print(almacen_tolerancia_error[-1])
    print("Pesos Finales:")
    ultimo_peso_final = almacen_w[-1]
    print(ultimo_peso_final)

    etiquetas_predichas = np.array([funcion_activacion(u) for u in obtener_u(matriz, w)])
    etiquetas_reales = np.array(yd, dtype=int)
    matriz_confusion = confusion_matrix(etiquetas_reales, etiquetas_predichas)

    print("Matriz de Confusión:")
    print(matriz_confusion)

    fig, ax = plt.subplots()
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Matriz de Confusión")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()

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
    treeview.insert("", tk.END, text="", values=(str(winicial), str(almacen_tolerancia_error[-1]), str(ultimo_peso_final)))
    treeview.pack()

    arreglo = almacen_tolerancia_error
    indices = np.arange(len(arreglo))
    plt.plot(indices, arreglo)
    plt.xlabel('Índice')
    plt.ylabel('Arreglo')
    plt.title('Gráfico de Arreglo vs Índice')
    plt.show()

    root.mainloop()

if __name__ == "__main__":
    main()
