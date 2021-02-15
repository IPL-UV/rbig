import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import gen_batches


class Ensemble:
    def __init__(self):
        pass

    def _fit(self, X: np.ndarray) -> BaseEstimator:
        pass

    def _fit_ensemble(self, X: np.ndarray, batch_size: int = 100) -> float:

        Hs = list()
        for idx in gen_batches(X.shape[0], batch_size, 10):
            Hs.append(self._fit(X[idx]))

        return np.mean(Hs)
