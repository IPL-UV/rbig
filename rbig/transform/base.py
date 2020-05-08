from typing import Tuple, Optional, Union
from abc import abstractmethod
import numpy as np
from numpy.random import RandomState
from sklearn.base import BaseEstimator, TransformerMixin
from rbig.density.base import PDFEstimator


class BaseTransform(BaseEstimator, TransformerMixin):
    def log_abs_det_jacobian(self, X: np.ndarray) -> float:
        raise NotImplementedError


class DensityMixin(object):
    """Mixin for :func:`sample` that returns the """

    @abstractmethod
    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return the mean log likelihood (or log(det(Jacobian))).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.
        y : None, default=None
            Not used but kept for compatibility.
        Returns
        -------
        log_likelihood : float
            Mean log likelihood data points in X.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        n_samples: int = 1,
        random_state: Optional[Union[RandomState, int]] = None,
    ) -> np.ndarray:
        """Take samples from the base distribution.
        
        Parameters 
        ----------
        n_samples : int, default=1
            the number of samples
        
        random_state: default=1
            the random state
        
        Returns
        -------
        samples : np.ndarray, (n_samples, n_features)
            the generated samples
        """
        raise NotImplementedError

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Return the mean log likelihood (or log(det(Jacobian))).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.
        y : None, default=None
            Not used but kept for compatibility.
        Returns
        -------
        log_likelihood : float
            Mean log likelihood data points in X.
        """
        return np.mean(self.score_samples(X, y))
