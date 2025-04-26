from src.core.plugins.registration_utils import register_plugin
from minisom import MiniSom


@register_plugin("agent")
class AgentFeatureSOM:
    def __init__(self, x=5, y=5, input_len=3, sigma=1.0, learning_rate=0.5):
        self.som = MiniSom(x, y, input_len, sigma=sigma, learning_rate=learning_rate)
        self.trained = False

    def train(self, data, num_iteration=100):
        self.som.random_weights_init(data)
        self.som.train_random(data, num_iteration)
        self.trained = True

    def assign_clusters(self, data):
        if not self.trained:
            raise RuntimeError("SOM not trained yet!")
        clusters = [self.som.winner(vec) for vec in data]
        return clusters
