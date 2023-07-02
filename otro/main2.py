import tkinter as tk
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk

class Perceptron():
    def __init__(self, aprendizaje, umbral):
        self.aprendizaje = aprendizaje
        self.umbral = umbral
        self.pesos = []
        self.error = 0
        self.X = []
        self.Y = []
        self.ids = []

    def leer_Archivo(self, archivo):
        df = pd.read_csv(archivo)
        x = df.iloc[:, 1:4].values
        y = df.iloc[:, 4].values
        ids = df.iloc[:, 0].values
        x = self.bias(x)
        self.ids = ids
        self.X = x
        self.Y = y
        self.pesos = np.random.rand(len(x[0]))
        global aa
        aa = self.pesos
        print("Pesos iniciales:", self.pesos)

    def bias(self, x):
        x_bias = []
        for i in range(len(x)):
            x_bias.append([1, x[i][0], x[i][1], x[i][2]])
        return x_bias

    def calculo_u(self):
        transpuestaW = np.transpose(self.pesos)
        u = np.linalg.multi_dot([self.X, transpuestaW])
        return u

    def funcion_activacion(self, u):
        return u

    def cal_error(self, ycal):
        error = []
        for i in range(len(ycal)):
            error.append(self.Y[i] - ycal[i])
        return error

    def delta_W(self, e):
        ret = np.transpose(e)
        for i in range(len(self.pesos)):
            dw = np.linalg.multi_dot([ret, self.X]) * self.aprendizaje
        return dw

    def nv_W(self, deltaW):
        nueva_W = self.pesos + deltaW
        self.pesos = nueva_W
        return nueva_W

    def cal_error2(self, error):
        e = 0
        n = len(error)
        for i in range(len(error)):
            e = e + error[i] ** 2
        mse = e / n
        rmse = m.sqrt(mse)
        return rmse


def inicializacion_alg():
    bandera = True
    try:
        aprendizaje = float(entryAprendizaje.get())
        umbral = 1
    except:
        bandera = False

    if bandera:
        nombre = archivo.split("/")
        nombre = nombre[len(nombre) - 1].split(".")
        algoritmo(aprendizaje, umbral, archivo)


def algoritmo(aprendizaje, umbral, archivo):
    global e
    perceptron = Perceptron(aprendizaje, umbral)
    a = str(aprendizaje)
    errores = []
    pesosSesgo = []
    pesosX1 = []
    pesosX2 = []
    pesosX3 = []
    errores_por_epoca = []
    perceptron.leer_Archivo(archivo)
    e = 10 # mandar
    i = 0
    while e > perceptron.umbral:
        u = perceptron.calculo_u()
        ycal = perceptron.funcion_activacion(u)
        error = perceptron.cal_error(ycal)
        deltaW = perceptron.delta_W(error)
        pesosSesgo.append(perceptron.pesos[0])
        pesosX1.append(perceptron.pesos[1])
        pesosX2.append(perceptron.pesos[2])
        pesosX3.append(perceptron.pesos[3])
        perceptron.nv_W(deltaW)
        e = perceptron.cal_error2(error)
        errores_por_epoca.append(error)
        errores.append(e)
        print("Estoy en error resssssssssssssssssssssss",errores)
        i += 1

    print("Pesos finales:", perceptron.pesos)
    print('Cantidad de epocas de entrenamiento:', i)
    print("Maximo error observado:", max(errores))
    pesos = [pesosSesgo, pesosX1, pesosX2, pesosX3]
    print('Estoy en pasos', pesos)  # grafica
    print('Vueltas t:', i)
    graficar_error(errores, a)
    pesos = [pesosSesgo, pesosX1, pesosX2, pesosX3]


def graficar_error(errores, a):
    root = tk.Tk()
    root.title("Evolución de la magnitud del error")

    figure = Figure(figsize=(8, 6), dpi=80)
    plot = figure.add_subplot(111)
    plot.plot(errores, label="TA:" + a, linestyle="-")
    plot.legend()
    plot.set_xlabel("Iteracion")
    plot.set_ylabel("Valor del error")

    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    root.mainloop()


def abrirArchivo():
    global archivo
    archivo = r"211105.CSV"




root = tk.Tk()
root.title("Perceptron")

root.geometry("250x300")
labelTAprendizaje = tk.Label(root, text="TAZA DE APRENDIZAJE")
labelTAprendizaje.place(x=30, y=75)
entryAprendizaje = tk.Entry(root)
entryAprendizaje.place(x=30, y=150, width=200, height=40)
labelMensaje = tk.Label(root, text="")
labelMensaje.place(x=30, y=400, width=621, height=30)
labelNota = tk.Label(root, text="")
labelNota.place(x=30, y=20, width=631, height=61)
buttonEjecutar = tk.Button(root, text="Generar gráficas", command=inicializacion_alg)
buttonEjecutar.place(x=30, y=200, width=201, height=50)
abrirArchivo()
entryAprendizaje.insert(0, "0.00000001")

root.mainloop()
