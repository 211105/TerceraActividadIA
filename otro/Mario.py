import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Datos de ejemplo
y_true = [0, 1, 1, 0, 1, 0, 0]
y_pred = [0, 0, 1, 0, 1, 1, 0]

# Calcular la matriz de confusión
matriz_confusion = confusion_matrix(y_true, y_pred)

# Crear una figura y un eje
fig, ax = plt.subplots()

# Crear la gráfica de matriz de confusión utilizando seaborn
sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

# Configurar etiquetas de los ejes
ax.set_xlabel("Etiquetas Predichas")
ax.set_ylabel("Etiquetas Verdaderas")

# Configurar título de la gráfica
ax.set_title("Matriz de Confusión")

# Mostrar la gráfica
plt.show()
