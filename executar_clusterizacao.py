import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from kmeans_manual import KMeansManual
from kmeans_biblioteca import executar_kmeans_sklearn
from avaliacao import calcular_metricas, exibir_matriz_confusao
from visualizacao_pca import plotar_clusters_pca

# ------------- Configurações -------------
KS = [3, 5]            
RANDOM_STATE = 42
SAVE_PLOTS = False     
PLOT_FOLDER = "."      
# ------------------------------------------

def carregar_dados():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y

def normalizar(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def medir_tempo(func, *args, **kwargs):
    start = time.perf_counter()
    out = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return out, elapsed

def main():
    X, y = carregar_dados()
    X_scaled, scaler = normalizar(X)

    resultados = {} 
    for k in KS:
        print(f"\n===== K = {k} =====")

        print("\n[KMeans Manual]")
        modelo_manual = KMeansManual(n_clusters=k, random_state=RANDOM_STATE)
        (_, tempo_manual) = medir_tempo(modelo_manual.ajustar, X_scaled)
        rotulos_manual = modelo_manual.rotulos
        score_manual = calcular_metricas(X_scaled, rotulos_manual)
        print(f"Silhouette Score: {score_manual:.4f} | Tempo: {tempo_manual:.4f}s")
        exibir_matriz_confusao(y, rotulos_manual, f"Manual K={k}")

        resultados[("manual", k)] = {
            "labels": rotulos_manual,
            "centers": modelo_manual.centroides,
            "silhouette": score_manual,
            "tempo": tempo_manual
        }

        print("\n[KMeans Sklearn]")
        (ret, tempo_sk) = medir_tempo(executar_kmeans_sklearn, X_scaled, k, RANDOM_STATE)
        rotulos_sk, centros_sk = ret
        score_sk = calcular_metricas(X_scaled, rotulos_sk)
        print(f"Silhouette Score: {score_sk:.4f} | Tempo: {tempo_sk:.4f}s")
        exibir_matriz_confusao(y, rotulos_sk, f"Sklearn K={k}")

        resultados[("sklearn", k)] = {
            "labels": rotulos_sk,
            "centers": centros_sk,
            "silhouette": score_sk,
            "tempo": tempo_sk
        }

    print("\n\n===== Resumo dos Resultados =====")
    for key, val in resultados.items():
        metodo, k = key
        print(f"{metodo.upper():7s} | k={k:2d} | silhouette={val['silhouette']:.4f} | tempo={val['tempo']:.4f}s")

    melhor_chave = max(resultados.items(), key=lambda kv: kv[1]['silhouette'])[0]
    melhor_metodo, melhor_k = melhor_chave
    melhor_silhouette = resultados[melhor_chave]['silhouette']
    print(f"\nMelhor resultado: método='{melhor_metodo}', k={melhor_k}, silhouette={melhor_silhouette:.4f}")

    print("\n===== Gerando PCA para visualização do melhor k =====")
    for metodo in ["sklearn", "manual"]:
        chave = (metodo, melhor_k)
        if chave in resultados:
            labels = resultados[chave]["labels"]
            centers = resultados[chave]["centers"]
            title2d = f"PCA 2D - {metodo.capitalize()} K={melhor_k}"
            title1d = f"PCA 1D - {metodo.capitalize()} K={melhor_k}"
            plotar_clusters_pca(X_scaled, labels, centers, 2, title2d)
            if SAVE_PLOTS:
                plt.savefig(f"{PLOT_FOLDER}/pca2d_{metodo}_k{melhor_k}.png")
            plotar_clusters_pca(X_scaled, labels, centers, 1, title1d)
            if SAVE_PLOTS:
                plt.savefig(f"{PLOT_FOLDER}/pca1d_{metodo}_k{melhor_k}.png")

    print("\nConcluído. Verifique as figuras e os outputs acima para compor o relatório.")

if __name__ == "__main__":
    main()