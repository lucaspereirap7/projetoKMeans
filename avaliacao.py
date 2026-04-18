import time
from sklearn.metrics import silhouette_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calcular_metricas(X, labels):
    return silhouette_score(X, labels)

def exibir_matriz_confusao(y_true, y_pred, titulo="Matriz de Confusão"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(titulo)
    plt.xlabel("Clusters")
    plt.ylabel("Classes Reais")
    plt.show()