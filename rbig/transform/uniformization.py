from typing import Callable, Optional, Tuple, Union, Dict

import numpy as np
from numpy.random import RandomState
from scipy import stats
from rbig.density.histogram import ScipyHistogram, QuantileHistogram
from rbig.density.kde import KDEFFT, KDEScipy, KDESklearn, KDEEpanechnikov

# Base classes
from sklearn.utils import check_array, check_random_state
from sklearn.base import clone

from rbig.transform.base import DensityMixin, BaseTransform
import matplotlib.pyplot as plt

# from rbig.density.base
from rbig.utils import (
    get_domain_extension,
    bin_estimation,
    make_interior,
    check_input_output_dims,
)

BOUNDS_THRESHOLD = 1e-7


class HistogramUniformization(BaseTransform, DensityMixin):
    def __init__(
        self,
        bins: Union[int, str] = "auto",
        alpha: float = 1e-5,
        prob_tol: float = 1e-7,
        n_quantiles: Optional[int] = None,
        support_extension: Union[float, int] = 10,
        subsample: Optional[int] = 10_000,
        random_state: int = 10,
        kwargs: Dict = {},
    ) -> None:
        self.bins = bins
        self.alpha = alpha
        self.prob_tol = prob_tol
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension
        self.subsample = subsample
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        # fit superclass
        if self.n_quantiles is not None:
            self.estimator_ = QuantileHistogram(
                bins=self.bins,
                alpha=self.alpha,
                n_quantiles=self.n_quantiles,
                subsample=self.subsample,
                random_state=self.random_state,
                support_extension=self.support_extension,
            ).fit(X)
        else:
            self.estimator_ = ScipyHistogram(
                bins=self.bins,
                alpha=self.alpha,
                prob_tol=self.prob_tol,
                support_extension=self.support_extension,
            ).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Forward transform which is the CDF function of
        the samples

            z  = P(x)
            P() - CDF
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            incomining samples

        Returns
        -------
        X : np.ndarray
            transformed data
        """
        X = check_array(X, ensure_2d=False, copy=True)

        return self.estimator_.cdf(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform which is the Inverse CDF function
        applied to the samples

            x = P^-1(z)
            P^-1() - Inverse CDF (PPF)

        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
        
        Returns
        -------
        X : np.ndarray, (n_samples, 1)
            Transformed data
        """
        X = check_array(X, ensure_2d=False, copy=True)

        return self.estimator_.ppf(X)

    def score_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Returns the log likelihood. It
        calculates the mean of the log probability.
        """
        # log_prob = self.log_abs_det_jacobian(X, y).sum(axis=1).reshape(-1, 1)

        # check_input_output_dims(log_prob, (X.shape[0], 1), "Histogram", "Log Prob")

        # return log_prob
        raise NotImplementedError

    def log_abs_det_jacobian(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> float:
        """Returns the log likelihood. It
        calculates the mean of the log probability.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
         incoming samples
        
        Returns
        -------
        X_jacobian : (n_samples, n_features),
            the mean of the log probability
        """
        X = check_array(X, ensure_2d=False, copy=True)

        return self.estimator_.logpdf(X)

    def sample(
        self,
        n_samples: int = 1,
        random_state: Optional[Union[RandomState, int]] = None,
    ) -> np.ndarray:
        """Generate random samples from this.
        
        Parameters
        ----------
        n_samples : int, default=1
            The number of samples to generate. 
        
        random_state : int, RandomState,None, Optional, default=None
            The int to be used as a seed to generate the random 
            uniform samples.
        
        Returns
        -------
        X : np.ndarray, (n_samples, )
        """
        #
        rng = check_random_state(random_state)

        U = rng.rand(n_samples, self.n_features_)

        X = self.inverse_transform(U)
        return X


class KDEUniformization(HistogramUniformization):
    def __init__(
        self,
        bw_method: str = "scott",
        n_quantiles: int = 1_000,
        support_extension: Union[float, int] = 10,
        method: str = "fft",
        algorithm: str = "kd_tree",
        kernel: str = "gaussian",
        metric: str = "euclidean",
        norm: int = 2,
        kwargs: Dict = {},
    ) -> None:
        self.bw_method = bw_method
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension
        self.method = method
        self.algorithm = algorithm
        self.metric = metric
        self.kernel = kernel
        self.norm = norm
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        # fit superclass
        if self.method in ["knn"]:
            self.estimator_ = KDESklearn(
                n_quantiles=self.n_quantiles,
                algorithm=self.algorithm,
                kernel=self.kernel,
                metric=self.metric,
                support_extension=self.support_extension,
            ).fit(X)
        elif self.method in ["fft"]:
            self.estimator_ = KDEFFT(
                bw_method=self.bw_method,
                n_quantiles=self.n_quantiles,
                support_extension=self.support_extension,
            ).fit(X)
        elif self.method in ["exact"]:
            self.estimator_ = KDEScipy(
                bw_method=self.bw_method,
                n_quantiles=self.n_quantiles,
                support_extension=self.support_extension,
            ).fit(X)
        elif self.method in ["epa"]:
            self.estimator_ = KDEEpanechnikov(
                kernel=self.kernel,
                bw_method=self.bw_method,
                n_quantiles=self.n_quantiles,
                support_extension=self.support_extension,
                norm=self.norm,
            ).fit(X)
        else:
            raise ValueError(f"Unrecognized method: {self.method}")
        return self


class MarginalUniformization(BaseTransform, DensityMixin):
    def __init__(self, uni_transformer) -> None:
        self.uni_transformer = uni_transformer

    def fit(self, X: np.ndarray) -> None:
        X = check_array(X, ensure_2d=True, copy=True)

        transforms = []

        for feature_idx in range(X.shape[1]):
            transformer = clone(self.uni_transformer)
            transforms.append(transformer.fit(X[:, feature_idx]))

        self.transforms_ = transforms

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, ensure_2d=True, copy=True)

        for feature_idx in range(X.shape[1]):

            X[:, feature_idx] = (
                self.transforms_[feature_idx].transform(X[:, feature_idx]).squeeze()
            )

        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, ensure_2d=True, copy=True)

        for feature_idx in range(X.shape[1]):

            X[:, feature_idx] = (
                self.transforms_[feature_idx]
                .inverse_transform(X[:, feature_idx])
                .squeeze()
            )

        return X

    def log_abs_det_jacobian(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> float:
        X = check_array(X, ensure_2d=True, copy=True)
        for feature_idx in range(X.shape[1]):

            X[:, feature_idx] = (
                self.transforms_[feature_idx]
                .log_abs_det_jacobian(X[:, feature_idx],)
                .squeeze()
            )
        return X
