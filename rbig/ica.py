from picard import picard
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from typing import Optional
import numpy as np


class OrthogonalICA(BaseEstimator, TransformerMixin):
    def __init__(self, ortho=True, random_state=123, whiten=False):
        self.ortho = ortho
        self.random_state = random_state
        self.whiten = whiten

    def fit(self, X, y=None):
        """Fit the model to X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
        Returns
        -------
        self
        """
        self._fit(X, compute_sources=False)
        return self

    def _fit(self, X: np.ndarray, compute_sources: bool = False) -> None:

        whitening, unmixing, sources, X_mean, self.n_iter_ = picard(
            X.T,
            ortho=self.ortho,
            random_state=123,
            whiten=self.whiten,
            return_X_mean=True,
            return_n_iter=True,
        )

        if self.whiten:
            self.components_ = np.dot(unmixing, whitening)
            self.mean_ = X_mean
            self.whitening_ = whitening
        else:
            self.components_ = unmixing

        self.mixing_ = np.linalg.pinv(self.components_)

        if compute_sources:
            self.__sources = sources
        return sources

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        if self.whiten:
            X -= self.mean_

        return np.dot(X, self.components_.T)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:

        X = check_array(X)
        X = np.dot(X, self.mixing_.T)
        if self.whiten:
            X += self.mean_

        return X

