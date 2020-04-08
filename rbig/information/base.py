from abc import abstractmethod
from sklearn.base import BaseEstimator
import numpy as np


class PDFEstimator(BaseEstimator):
    @abstractmethod
    def logpdf(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
