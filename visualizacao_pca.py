from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plotar_clusters_pca(X, rotulos, centroides, n_componentes=2, titulo="PCA"):
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X)
    centroides_pca = pca.transform(centroides)

    plt.figure(figsize=(8, 6))

    if n_componentes == 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=rotulos, cmap='viridis', s=50, alpha=0.6)
        plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1], c='red', s=200, marker='X')
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
    else:
        plt.scatter(range(len(X_pca)), X_pca[:, 0], c=rotulos, cmap='viridis', s=20)
        plt.scatter(range(len(centroides_pca)), centroides_pca[:, 0], c='red', marker='X', s=100)
        plt.ylabel("Componente Principal 1")

    plt.title(titulo)
    plt.grid(True)
    plt.show()