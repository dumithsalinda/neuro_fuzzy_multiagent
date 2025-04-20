import numpy as np
from minisom import MiniSom


class SOMClusterer:
    """
    Utility for clustering agent features using Self-Organizing Maps (SOM).
    """

    def __init__(
        self,
        input_dim,
        som_shape=(5, 5),
        sigma=1.0,
        learning_rate=0.5,
        num_iteration=100,
    ):
        self.som_shape = som_shape
        self.som = MiniSom(
            som_shape[0],
            som_shape[1],
            input_dim,
            sigma=sigma,
            learning_rate=learning_rate,
        )
        self.num_iteration = num_iteration
        self._trained = False

    def fit(self, X):
        X = np.asarray(X)
        self.som.random_weights_init(X)
        self.som.train(X, self.num_iteration, verbose=False)
        self._trained = True

    def predict(self, X):
        """Assigns each sample to a SOM node (cluster)."""
        if not self._trained:
            raise RuntimeError("SOM must be trained first.")
        X = np.asarray(X)
        clusters = []
        for x in X:
            clusters.append(self.som.winner(x))
        return clusters

    def get_weights(self):
        return self.som.get_weights()
