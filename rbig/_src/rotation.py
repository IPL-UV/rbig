import numpy as np
from picard import Picard
from sklearn.decomposition import PCA
from scipy.stats import ortho_group


class ICARotation:
    name: str = "ica"

    def __init__(
        self,
        X: np.ndarray,
        random_state=123,
        max_iter=100,
        **kwargs
    ):

        # create pca object
        self.estimator = Picard(
            ortho=True,
            whiten=False,
            extended=None,
            random_state=random_state,
            max_iter=max_iter,
            **kwargs
        ).fit(X)

    def forward(self, X):

        Z = self.estimator.transform(X)

        return Z

    def inverse(self, Z):
        X = self.estimator.inverse_transform(Z)

        return X

    def gradient(self, X):

        X_grad = np.zeros(X.shape[0])

        return X_grad


class PCARotation:
    name: str = "pca"

    def __init__(self, X: np.ndarray, **kwargs):

        # create histogram object
        self.estimator = PCA().fit(X)

    def forward(self, X):

        Z = self.estimator.transform(X)

        return Z

    def inverse(self, Z):
        X = self.estimator.inverse_transform(Z)

        return X

    def gradient(self, X):

        X_grad = np.zeros(X.shape[0])

        return X_grad


class RandomRotation:
    name: str = "ica"

    def __init__(self, X: np.ndarray, **kwargs):

        # create histogram object
        self.rand_ortho_matrix = ortho_group.rvs(X.shape[1])

    def forward(self, X):

        Z = X @ self.rand_ortho_matrix

        return Z

    def inverse(self, Z):
        X = Z @ self.rand_ortho_matrix.T

        return X

    def gradient(self, X):

        X_grad = np.zeros(X.shape[0])

        return X_grad
