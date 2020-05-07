import numpy as np
from sklearn import datasets


class ToyData:
    """Generate demo datasets for RBIG
    
    Parameters
    ----------
    dataset : str, default='rbig'
        datasets to be generated
        Options - {'rbig', 'moons', 'blob'}
    
    n_samples : int, default=1_000
    n_features : int
    noise : float, default=0.05
    """

    def __init__(
        self,
        dataset="rbig",
        n_samples=1_000,
        n_features=2,
        noise=0.05,
        random_state=123,
        clusters=2,
    ):
        self.dataset = dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.clusters = clusters
        self.rng = np.random.RandomState(random_state)

    def generate_samples(self, **kwargs):

        if self.dataset == "rbig":
            X = self._dataset_rbig()

        elif self.dataset == "moons":
            X, _ = datasets.make_moons(
                n_samples=self.n_samples,
                noise=self.noise,
                random_state=self.rng,
                **kwargs,
            )

        elif self.dataset == "blobs":
            X, _ = datasets.make_blobs(
                n_samples=self.n_samples,
                n_features=self.n_features,
                cluster_std=self.noise,
                random_state=self.rng,
                centers=self.clusters,
                **kwargs,
            )

        else:

            raise ValueError(f"Unrecognized dataset: {self.dataset}")

        return X

    def _dataset_rbig(self):
        X = np.abs(2 * self.rng.randn(self.n_samples, 1))
        Y = np.sin(X) + self.noise * self.rng.randn(self.n_samples, 1)
        data = np.hstack((X, Y))

        return data
