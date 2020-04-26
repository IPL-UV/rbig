
from abc import abstractmethod
from sklearn.base import BaseEstimator
import numpy as np


class PDFEstimator(BaseEstimator):

    @abstractmethod
    def pdf(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        return np.log(self.pdf(X))

    @abstractmethod
    def cdf(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def ppf(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def entropy(self, X: np.ndarray) -> float:
        raise NotImplementedError
