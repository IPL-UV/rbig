from typing import Callable, Optional, Union, Dict

import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn.utils import check_array, check_random_state

from rbig.transform.base import DensityMixin, BaseTransform

# from rbig.density import Histogram
from rbig.transform.gauss_icdf import InverseGaussCDF
from rbig.transform.histogram import ScipyHistogramUniformization
from rbig.transform.kde import ScipyKDEUniformization, SklearnKDEUniformization


class HistogramGaussianization(BaseTransform, DensityMixin):
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
        support_extension: Union[int, float] = 10,
        kwargs: Dict = {},
    ) -> None:
        self.nbins = nbins
        self.alpha = alpha
        self.support_extension = support_extension
        self.kwargs = kwargs

    def fit(self, X, y=None):

        # Uniformization
        self.marg_uniformer_ = ScipyHistogramUniformization(
            nbins=self.nbins,
            alpha=self.alpha,
            support_extension=self.support_extension,
            kwargs=self.kwargs,
        )
        self.marg_uniformer_.fit(X)

        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:

        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        # 1. Marginal Uniformization
        X = self.marg_uniformer_.transform(X)

        # 2. Marginal Gaussianization
        X = InverseGaussCDF().transform(X)

        return X

    def log_abs_det_jacobian(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        # marginal uniformization
        # print("X: ", X.min(), X.max())
        Xu_der = self.marg_uniformer_.log_abs_det_jacobian(X)
        # print("Xu Jacobian:", Xu_der.min(), Xu_der.max())
        X = self.marg_uniformer_.transform(X)
        # print("X_u:", X.min(), X.max())

        # inverse CDF gaussian
        # X = InverseGaussCDF().transform(X)
        # print("Xg:", X.min(), X.max())

        Xg_der = InverseGaussCDF().log_abs_det_jacobian(X)
        # print("Xg jacobian:", Xg_der.min(), Xg_der.max())
        # print(f"#Nans: {np.count_nonzero(~np.isnan(Xg_der))} / {Xg_der.shape[0]}")

        return Xu_der + Xg_der

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        # 1. Inverse Gaussianization
        X = InverseGaussCDF().inverse_transform(X)

        # 2. Inverse Uniformization
        X = self.marg_uniformer_.inverse_transform(X)

        return X

    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Returns the log determinant abs jacobian of the inputs.
        
        Parameters
        ----------
        X : np.ndarray
            Inputs to be transformed
        
        y: Not used, only for compatibility
        
        """

        # Marginal Gaussianization Transformation
        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        x_logprob = stats.norm().logpdf(self.transform(X))

        return (x_logprob + self.log_abs_det_jacobian(X)).sum(axis=1)

    def sample(
        self, n_samples: int = 1, random_state: Optional[Union[RandomState, int]] = None
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


class KDEGaussianization(BaseTransform, DensityMixin):
    """This performs a univariate transformation on a datasets.
    
    Assuming that the data is independent across features, this
    applies a transformation on each feature independently. The inverse 
    transformation is the marginal cdf applied to each of the features
    independently and the inverse transformation is the marginal inverse
    cdf applied to the features independently.
    """

    def __init__(
        self,
        method: str = "exact",
        bw_estimator: str = "scott",
        algorithm: str = "kd_tree",
        kernel: str = "gaussian",
        metric: str = "euclidean",
        n_quantiles: int = 1_000,
        support_extension: Union[int, float] = 10,
        kwargs: Optional[Dict] = {},
    ) -> None:
        self.method = method
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension
        self.bw_estimator = bw_estimator
        self.algorithm = algorithm
        self.kernel = kernel
        self.metric = metric
        self.kwargs = kwargs

    def fit(self, X, y=None):

        # Uniformization
        if self.method == "exact":
            self.marg_uniformer_ = ScipyKDEUniformization(
                n_quantiles=self.n_quantiles,
                support_extension=self.support_extension,
                bw_estimator=self.bw_estimator,
            )
        elif self.method == "knn":
            self.marg_uniformer_ = SklearnKDEUniformization(
                n_quantiles=self.n_quantiles,
                support_extension=self.support_extension,
                algorithm=self.algorithm,
                kernel=self.kernel,
                metric=self.metric,
                kwargs=self.kwargs,
            )
        else:
            raise ValueError(f"Unrecognized method: {self.method}")

        self.marg_uniformer_.fit(X)

        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:

        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        # 1. Marginal Uniformization
        X = self.marg_uniformer_.transform(X)

        # 2. Marginal Gaussianization
        X = InverseGaussCDF().transform(X)

        return X

    def log_abs_det_jacobian(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        # marginal uniformization
        # print("X: ", X.min(), X.max())
        Xu_der = self.marg_uniformer_.log_abs_det_jacobian(X)
        # print("Xu Jacobian:", Xu_der.min(), Xu_der.max())
        X = self.marg_uniformer_.transform(X)
        # print("X_u:", X.min(), X.max())

        # inverse CDF gaussian
        # X = InverseGaussCDF().transform(X)
        # print("Xg:", X.min(), X.max())

        Xg_der = InverseGaussCDF().log_abs_det_jacobian(X)
        # print("Xg jacobian:", Xg_der.min(), Xg_der.max())
        # print(f"#Nans: {np.count_nonzero(~np.isnan(Xg_der))} / {Xg_der.shape[0]}")

        return Xu_der + Xg_der

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        # 1. Inverse Gaussianization
        X = InverseGaussCDF().inverse_transform(X)

        # 2. Inverse Uniformization
        X = self.marg_uniformer_.inverse_transform(X)

        return X

    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Returns the log determinant abs jacobian of the inputs.
        
        Parameters
        ----------
        X : np.ndarray
            Inputs to be transformed
        
        y: Not used, only for compatibility
        
        """

        # Marginal Gaussianization Transformation
        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        x_logprob = stats.norm().logpdf(self.transform(X))

        return (x_logprob + self.log_abs_det_jacobian(X)).sum(axis=1)

    def sample(
        self, n_samples: int = 1, random_state: Optional[Union[RandomState, int]] = None
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
