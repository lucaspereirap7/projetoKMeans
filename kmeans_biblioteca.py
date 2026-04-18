from sklearn.cluster import KMeans

def executar_kmeans_sklearn(X, k, random_state=42):
    modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    modelo.fit(X)
    return modelo.labels_, modelo.cluster_centers_