from sklearn.base import BaseEstimator, TransformerMixin
from rbig.base import UniformMixin, DensityMixin, ScoreMixin, GaussMixin
from typing import Optional, Union, Callable
from scipy import stats
from sklearn.utils import check_random_state, check_array
from rbig.density import Histogram
from rbig.transform import InverseGaussCDF
import numpy as np


class HistogramGaussianization(BaseEstimator, TransformerMixin, ScoreMixin):
    """This performs a univariate transformation on a datasets.
    
    Assuming that the data is independent across features, this
    applies a transformation on each feature independently. The inverse 
    transformation is the marginal cdf applied to each of the features
    independently and the inverse transformation is the marginal inverse
    cdf applied to the features independently.
    """

    def __init__(
        self,
        nbins: Optional[Union[int, str]] = "auto",
        alpha: float = 1e-5,
        log: bool = True,
    ) -> None:
        self.nbins = nbins
        self.alpha = alpha
        self.log = log

    def fit(self, X, y=None):

        # Uniformization
        self.uniformer = Histogram(nbins=self.nbins, alpha=self.alpha)
        self.uniformer.fit(X)

        return self

    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        return_jacobian: bool = False,
    ) -> np.ndarray:

        # 1. Uniformization
        Xu = self.uniformer.transform(X)

        # 2. Gaussianization
        Xg = InverseGaussCDF().transform(Xu)

        if not return_jacobian:
            return Xg
        else:
            Xu_der = self.abs_det_jacobian(X, log=True)
            Xg_der = InverseGaussCDF().abs_det_jacobian(Xu, log=True)
            return Xg, Xg_der + Xu_der

    def inverse_transform(self, X, y=None):

        # 1. Inverse Gaussianization
        X_trans = InverseGaussCDF().inverse_transform(X)

        # 2. Inverse Uniformization
        X_trans = self.uniformer.inverse_transform(X_trans)

        return X_trans

    def abs_det_jacobian(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, log: Optional[bool] = True
    ) -> np.ndarray:

        # Transformation
        Xu = self.uniformer.transform(X)

        # ==========================
        # Uniformization
        # ===========================

        # Log Probability Uniformer
        if log is None:
            log = self.log

        if log == True:
            Xu_der = self.uniformer.log_abs_det_jacobian(X)
            Xg_der = InverseGaussCDF().log_abs_det_jacobian(Xu)
            return Xu_der + Xg_der
        elif log == False:
            Xu_der = self.uniformer.abs_det_jacobian(X)
            Xg_der = InverseGaussCDF().abs_det_jacobian(Xu)
            return Xu_der * Xg_der
        else:
            raise ValueError("Unrecognized command for log.")

    def score_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:

        return self.abs_det_jacobian(X, log=True)
