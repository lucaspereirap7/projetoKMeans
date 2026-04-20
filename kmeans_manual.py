import numpy as np

class KMeansManual:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroides = None
        self.rotulos = None

    def ajustar(self, X):
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroides = X[indices]

        for _ in range(self.max_iter):
            distancias = self._calcular_distancias(X, self.centroides)
            self.rotulos = np.argmin(distancias, axis=1)

            novos_centroides = []
            for i in range(self.n_clusters):
                pontos_cluster = X[self.rotulos == i]
                if len(pontos_cluster) == 0:
                    novos_centroides.append(self.centroides[i])
                else:
                    novos_centroides.append(pontos_cluster.mean(axis=0))

            novos_centroides = np.array(novos_centroides)

            if np.linalg.norm(self.centroides - novos_centroides) < self.tol:
                self.centroides = novos_centroides
                break

            self.centroides = novos_centroides

        return self

    def prever(self, X):
        distancias = self._calcular_distancias(X, self.centroides)
        return np.argmin(distancias, axis=1)

    def _calcular_distancias(self, X, centroides):
        # Retorna uma matriz (n_amostras, n_clusters)
        return np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)