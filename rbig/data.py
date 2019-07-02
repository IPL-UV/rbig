import numpy as np
from sklearn.datasets import make_moons


class ToyData:
    def __init__(self, dataset="rbig", n_samples=1000, noise=0.05, random_state=123):
        self.dataset = dataset
        self.n_samples = n_samples
        self.noise = noise
        self.rng = np.random.RandomState(random_state)

    def generate_samples(self):

        if self.dataset == "rbig":
            return self._dataset_rbig()

        elif self.dataset == "moons":
            dataset, y = make_moons(n_samples=self.n_samples, noise=self.noise)
            return dataset
        else:

            return None

    def _dataset_rbig(self):
        seed = 123

        num_samples = 5000
        X = np.abs(2 * self.rng.randn(self.n_samples, 1))
        Y = np.sin(X) + self.noise * self.rng.randn(self.n_samples, 1)
        data = np.hstack((X, Y))

        return data
