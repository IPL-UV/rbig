import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import slogdet, inv


# TODO - Whitening transform - https://github.com/lucastheis/isa/blob/master/code/transforms/whiteningtransform.py


class LinearTransform:
    def __init__(self, rotation="pca", random_state=123, kwargs=None):
        self.rotation = rotation
        self.rng = np.random.RandomState(random_state)
        self.kwargs = kwargs

    def fit(self, X):
        if self.rotation == "pca":
            if self.kwargs is not None:
                self.model = PCA(**self.kwargs)
            else:
                self.model = PCA(random_state=123)
            self.model.fit(X)
            self.R = self.model.components_.T

        else:
            raise ValueError(f"Unrecognized rotation...")

        return self

    def transform(self, X):
        return X @ self.R

    def inverse_transform(self, X):
        return X @ self.R.T

    def logdetjacobian(self, X=None):
        if X is None:
            return slogdet(self.R.T)[1]
        else:
            return slogdet(self.R)[1] + np.zeros([1, X.shape[1]])
