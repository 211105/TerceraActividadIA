import csv
import numpy as np
import random
import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import locale

# Establecer la configuración regional en inglés
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

datos_x = []
datos_y = []

# Abre el archivo CSV
with open('Primera/211099.csv', 'r') as archivo_csv:
    # Lee el archivo CSV utilizando el lector de CSV
    lector_csv = csv.reader(archivo_csv, delimiter=';')

    # Omite la primera fila que contiene los encabezados
    next(lector_csv)

    # Itera sobre cada fila del archivo CSV
    for fila in lector_csv:
        # Obtiene los valores de x1, x2 y x3 de la fila actual
        valores_x = [float(i) for i in fila[1:4]]

        # Agrega un 1 al inicio de cada elemento de valores_x
        valores_x.insert(0, 1)

        # Obtiene el valor de Y de la última columna
        valor_y = float(fila[4])

        # Agrega los valores de x1, x2 y x3 a la lista de datos_x
        datos_x.append(valores_x)

        # Agrega el valor de Y a la lista de datos_y
        datos_y.append(valor_y)

def obtener_winicial(matriz):
    w = np.array([],dtype=float)
    for i in range(int(len(matriz[0]))):
        num = random.randint(-5, 5)
        while num == 0:
            num = random.randint(-5, 5)
        w = np.append(w, num)
    return w

def funcion_activacion(u):
    return u

def obtener_u(matriz, w):
    u = np.array([])
    for i in range(len(matriz)):
        u_multiplicacion = matriz[i] * w
        u_suma = sum(u_multiplicacion)
        u = np.append(u, u_suma)
    return(u)
        
def obtener_yc(u):
    return u

def obtener_e(yd, yc):
    e = yd - yc
    return e

def obtener_deltaw(matriz,e,tasa_aprendisaje):
    deltaW = tasa_aprendisaje * np.dot(e.T, matriz)
    return deltaW

def obtener_nuevaw(w,delta_W):
    nueva_w = w + delta_W
    return nueva_w

def obtener_tolerancia_error(e):
    suma_cuadrados = np.sum(e**2)
    raiz_suma_cuadrados = np.sqrt(suma_cuadrados)
    return raiz_suma_cuadrados

def main():
    #datos
    tolerancia_error = 1.0
    almacen_w = np.array([])
    almacen_tolerancia_error = np.array([])
    matriz = np.array(datos_x,dtype=float)
    yd = np.array(datos_y, dtype=float)
    w = obtener_winicial(matriz)
    winicial = obtener_winicial(matriz)
    almacen_w = np.vstack([w])
    tasa_apredisaje = 0.00000009
    margen_error = 0.5

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Configuración")
    root.geometry("250x100")

    def start_algorithm():
        nonlocal tasa_apredisaje
        tasa_apredisaje = float(entry.get())
        root.destroy()

    # Create an input field
    label = tk.Label(root, text="Tasa de Aprendizaje:")
    label.pack()
    entry = tk.Entry(root, width=20)
    entry.insert(tk.END, "{:.8f}".format(tasa_apredisaje))
    entry.pack()

    # Create a button to start the algorithm
    button = tk.Button(root, text="Iniciar", command=start_algorithm)
    button.pack()

    # Run the Tkinter event loop
    root.mainloop()

    # Bucle
    while tolerancia_error > margen_error:
        u = obtener_u(matriz, w)
        yc = obtener_yc(u)
        e = obtener_e(yd, yc)
        tolerancia_error = obtener_tolerancia_error(e)
        almacen_tolerancia_error = np.append(almacen_tolerancia_error, tolerancia_error)
        deltaw = obtener_deltaw(matriz,e,tasa_apredisaje)
        w = obtener_nuevaw(w, deltaw)
        almacen_w = np.vstack([w])

    print(winicial)
    print("arreglo tolerancia: ", almacen_tolerancia_error[-1])
    print(almacen_w)

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Resultados")

    # Create a treeview widget
    treeview = ttk.Treeview(root)

    # Define columns
    treeview["columns"] = ("Pesos Iniciales", "Tolerancia", "Pesos Finales")

    # Format columns
    treeview.column("#0", width=0, stretch=tk.NO)
    treeview.column("Pesos Iniciales", anchor=tk.W, width=150)
    treeview.column("Tolerancia", anchor=tk.CENTER, width=150)
    treeview.column("Pesos Finales", anchor=tk.W, width=150)

    # Create headings
    treeview.heading("#0", text="")
    treeview.heading("Pesos Iniciales", text="Pesos Iniciales")
    treeview.heading("Tolerancia", text="Tolerancia")
    treeview.heading("Pesos Finales", text="Pesos Finales")

    # Insert data
    treeview.insert("", tk.END, text="", values=(str(winicial), str(almacen_tolerancia_error[-1]), str(almacen_w)))

    # Pack the treeview
    treeview.pack()

    # Create the plot
    arreglo = almacen_tolerancia_error
    indices = np.arange(len(arreglo))
    plt.plot(indices, arreglo)
    plt.xlabel('Índice')
    plt.ylabel('Arreglo')
    plt.title('Gráfico de Arreglo vs Índice')
    plt.show()

    # Run the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
