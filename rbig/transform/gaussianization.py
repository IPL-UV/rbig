from typing import Callable, Optional, Union, Dict

import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn.utils import check_array, check_random_state
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

from rbig.transform.base import DensityMixin, BaseTransform

# from rbig.density import Histogram
from rbig.transform.gauss_icdf import InverseGaussCDF

# from rbig.transform.histogram import ScipyHistogramUniformization
from rbig.utils import get_domain_extension, get_support_reference
import warnings
from sklearn.base import clone


class Gaussianization(BaseTransform, DensityMixin):
    """class to take a univariate Gaussianization
    
    This class composes a uniform transformer and a Inverse Gauss CDF
    transformation to make a Gaussianization transformation

    Parameters
    ----------
    uni_transformer : BaseTransform
        any base transformation that transforms data to a 
        uniform distribution.
    
    """

    def __init__(self, uni_transformer) -> None:
        self.uni_transformer = uni_transformer

    def fit(self, X: np.ndarray) -> None:
        """Fits the uniform transformation to the data
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            the input data to be transformed
        
        Returns
        -------
        self : instance of self
        """
        # check inputs
        X = check_array(X, ensure_2d=False, copy=True)

        # fit uniformization to data
        self.uni_transformer.fit(X)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Forward transformation of X.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            the data to be transformed to the gaussian domain
        
        Returns
        -------
        Xtrans : np.ndarray, (n_samples, n_features)
            the transformed Gaussianized data
        """
        # check inputs
        X = check_array(X, ensure_2d=False, copy=True)

        # transform data to uniform domain
        X = self.uni_transformer.transform(X)

        # transform data to gaussian domain
        X = InverseGaussCDF().transform(X)

        # return gaussianized variable
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Performs the inverse transformation to original domain
        
        This transforms univariate Gaussian data to the original
        domain of the fitted transformation.

        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Gaussian data
        
        Returns
        -------
        X : np.ndarray, (n_samples, n_features)
            the data in the original data domain.
        """
        # check inputs
        X = check_array(X, ensure_2d=False, copy=True)

        # transform data to uniform domain
        X = InverseGaussCDF().inverse_transform(X)

        # transform data to origin domain
        X = self.uni_transformer.inverse_transform(X)

        # return data from original domain
        return X

    def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Calculates the log-det-jacobian of the transformation
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            the data to be transformed
        
        Returns
        -------
        Xslogdet : np.ndarray, (n_samples, n_features)
            the log det-jacobian for each sample
        """
        # check array
        X = check_array(X.reshape(-1, 1), ensure_2d=True, copy=True)

        # find uniform probability
        u_log_prob = self.uni_transformer.log_abs_det_jacobian(X)

        # transform data into gaussian domain
        X_g = self.transform(X)

        # find gaussian probability
        g_log_prob = stats.norm().logpdf(X_g.squeeze())

        # return combined log det-jacobian
        return u_log_prob - g_log_prob


class MarginalGaussianization(BaseTransform, DensityMixin):
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
                .log_abs_det_jacobian(X[:, feature_idx])
                .squeeze()
            )
        return X


# class QuantileGaussianization(QuantileTransformer, DensityMixin):
#     def __init__(
#         self,
#         n_quantiles: int = 1_000,
#         support_extension: Union[float, int] = 10,
#         subsample: int = 1e5,
#         random_state: Optional[int] = None,
#     ) -> None:
#         super().__init__(
#             n_quantiles=n_quantiles,
#             output_distribution="normal",
#             subsample=subsample,
#             random_state=random_state,
#             copy=True,
#         )
#         self.support_extension = support_extension

#     def _dense_fit(self, X, random_state):
#         """Compute percentiles for dense matrices.
#         Parameters
#         ----------
#         X : ndarray, shape (n_samples, n_features)
#             The data used to scale along the features axis.
#         """
#         if self.ignore_implicit_zeros:
#             warnings.warn(
#                 "'ignore_implicit_zeros' takes effect only with"
#                 " sparse matrix. This parameter has no effect."
#             )

#         n_samples, n_features = X.shape
#         references = self.references_ * 100

#         self.quantiles_ = []
#         for col in X.T:
#             # extend the domain for the feature
#             new_support = get_support_reference(
#                 support=col, extension=self.support_extension
#             )

#             if self.subsample < n_samples:
#                 subsample_idx = random_state.choice(
#                     n_samples, size=self.subsample, replace=False
#                 )
#                 col = col.take(subsample_idx, mode="clip")

#             # interp
#             # col_new = np.interp(col,)

#             quantiles = np.nanpercentile(col, references)
#             quantiles = np.interp(new_support, quantiles, references)

#             self.quantiles_.append(quantiles)
#         self.quantiles_ = np.transpose(self.quantiles_)
#         # Due to floating-point precision error in `np.nanpercentile`,
#         # make sure that quantiles are monotonically increasing.
#         # Upstream issue in numpy:
#         # https://github.com/numpy/numpy/issues/14685
#         self.quantiles_ = np.maximum.accumulate(self.quantiles_)

#     def log_abs_det_jacobian(
#         self, X: np.ndarray, y: Optional[np.ndarray] = None
#     ) -> float:
#         """Returns the log likelihood. It
#         calculates the mean of the log probability.

#         Parameters
#         ----------
#         X : np.ndarray, (n_samples, 1)
#          incoming samples

#         Returns
#         -------
#         X_jacobian : (n_samples, n_features),
#             the mean of the log probability
#         """
#         X = check_array(X, ensure_2d=True, copy=True)

#         n_samples = X.shape[0]

#         # forward transformation

#         # log pdf of a
#         ldj = -stats.norm().logpdf(self.transform(X))

#         return ldj

#     # def logpdf()
#     def score_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
#         """Returns the log likelihood. It
#         calculates the mean of the log probability.
#         """
#         X = check_array(X, ensure_2d=True, copy=False)

#         # forward transformation
#         x_logprob = -stats.norm().logpdf(self.transform(X))

#         return x_logprob.sum(axis=-1)

#     def sample(
#         self, n_samples: int = 1, random_state: Optional[Union[RandomState, int]] = None
#     ) -> np.ndarray:
#         """Generate random samples from this.

#         Parameters
#         ----------
#         n_samples : int, default=1
#             The number of samples to generate.

#         random_state : int, RandomState,None, Optional, default=None
#             The int to be used as a seed to generate the random
#             uniform samples.

#         Returns
#         -------
#         X : np.ndarray, (n_samples, )
#         """
#         #
#         rng = check_random_state(random_state)

#         X_gauss = rng.randn(n_samples, self.quantiles_.shape[1])

#         X = self.inverse_transform(X_gauss)
#         return X


# class PowerGaussianization(PowerTransformer, DensityMixin):
#     def __init__(
#         self,
#         standardize: bool = True,
#         support_extension: Union[float, int] = 10,
#         copy: bool = True,
#     ) -> None:
#         super().__init__(
#             method="yeo-johnson", standardize=standardize, copy=copy,
#         )
#         self.support_extension = support_extension

#     def log_abs_det_jacobian(
#         self, X: np.ndarray, y: Optional[np.ndarray] = None
#     ) -> float:
#         """Returns the log likelihood. It
#         calculates the mean of the log probability.

#         Parameters
#         ----------
#         X : np.ndarray, (n_samples, 1)
#          incoming samples

#         Returns
#         -------
#         X_jacobian : (n_samples, n_features),
#             the mean of the log probability
#         """
#         X = check_array(X, ensure_2d=True, copy=True)

#         n_samples = X.shape[0]

#         # forward transformation
#         X = self.transform(X)

#         # log pdf of a
#         ldj = stats.norm().logpdf(X)

#         return ldj

#     def score_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
#         """Returns the log likelihood. It
#         calculates the mean of the log probability.
#         """

#         # forward transformation
#         log_prob = self.log_abs_det_jacobian(X, y).sum(axis=1)

#         return log_prob

#     def sample(
#         self, n_samples: int = 1, random_state: Optional[Union[RandomState, int]] = None
#     ) -> np.ndarray:
#         """Generate random samples from this.

#         Parameters
#         ----------
#         n_samples : int, default=1
#             The number of samples to generate.

#         random_state : int, RandomState,None, Optional, default=None
#             The int to be used as a seed to generate the random
#             uniform samples.

#         Returns
#         -------
#         X : np.ndarray, (n_samples, )
#         """
#         #
#         rng = check_random_state(random_state)

#         X_gauss = rng.randn(n_samples, self.quantiles_.shape[1])

#         X = self.inverse_transform(X_gauss)
#         return X


# class HistogramGaussianization(BaseTransform, DensityMixin):
#     """This performs a univariate transformation on a datasets.

#     Assuming that the data is independent across features, this
#     applies a transformation on each feature independently. The inverse
#     transformation is the marginal cdf applied to each of the features
#     independently and the inverse transformation is the marginal inverse
#     cdf applied to the features independently.
#     """

#     def __init__(
#         self,
#         nbins: Optional[Union[int, str]] = "auto",
#         alpha: float = 1e-5,
#         support_extension: Union[int, float] = 10,
#         kwargs: Dict = {},
#     ) -> None:
#         self.nbins = nbins
#         self.alpha = alpha
#         self.support_extension = support_extension
#         self.kwargs = kwargs

#     def fit(self, X, y=None):

#         # Uniformization
#         self.marg_uniformer_ = ScipyHistogramUniformization(
#             nbins=self.nbins,
#             alpha=self.alpha,
#             support_extension=self.support_extension,
#             kwargs=self.kwargs,
#         )
#         self.marg_uniformer_.fit(X)

#         return self

#     def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:

#         # check inputs
#         X = check_array(X, ensure_2d=True, copy=False)

#         # 1. Marginal Uniformization
#         X = self.marg_uniformer_.transform(X)

#         # 2. Marginal Gaussianization
#         X = InverseGaussCDF().transform(X)

#         return X

#     def log_abs_det_jacobian(
#         self, X: np.ndarray, y: Optional[np.ndarray] = None
#     ) -> np.ndarray:

#         # marginal uniformization
#         # print("X: ", X.min(), X.max())
#         Xu_der = self.marg_uniformer_.log_abs_det_jacobian(X)
#         # print("Xu Jacobian:", Xu_der.min(), Xu_der.max())
#         X = self.marg_uniformer_.transform(X)
#         # print("X_u:", X.min(), X.max())

#         # inverse CDF gaussian
#         # X = InverseGaussCDF().transform(X)
#         # print("Xg:", X.min(), X.max())

#         Xg_der = InverseGaussCDF().log_abs_det_jacobian(X)
#         # print("Xg jacobian:", Xg_der.min(), Xg_der.max())
#         # print(f"#Nans: {np.count_nonzero(~np.isnan(Xg_der))} / {Xg_der.shape[0]}")

#         return Xu_der + Xg_der

#     def inverse_transform(
#         self, X: np.ndarray, y: Optional[np.ndarray] = None
#     ) -> np.ndarray:

#         # check inputs
#         X = check_array(X, ensure_2d=True, copy=False)

#         # 1. Inverse Gaussianization
#         X = InverseGaussCDF().inverse_transform(X)

#         # 2. Inverse Uniformization
#         X = self.marg_uniformer_.inverse_transform(X)

#         return X

#     def score_samples(
#         self, X: np.ndarray, y: Optional[np.ndarray] = None
#     ) -> np.ndarray:
#         """Returns the log determinant abs jacobian of the inputs.

#         Parameters
#         ----------
#         X : np.ndarray
#             Inputs to be transformed

#         y: Not used, only for compatibility

#         """

#         # Marginal Gaussianization Transformation
#         # check inputs
#         X = check_array(X, ensure_2d=True, copy=False)

#         x_logprob = stats.norm().logpdf(self.transform(X))

#         return (x_logprob + self.log_abs_det_jacobian(X)).sum(axis=1)

#     def sample(
#         self, n_samples: int = 1, random_state: Optional[Union[RandomState, int]] = None
#     ) -> np.ndarray:
#         """Generate random samples from this.

#         Parameters
#         ----------
#         n_samples : int, default=1
#             The number of samples to generate.

#         random_state : int, RandomState,None, Optional, default=None
#             The int to be used as a seed to generate the random
#             uniform samples.

#         Returns
#         -------
#         X : np.ndarray, (n_samples, )
#         """
#         #
#         rng = check_random_state(random_state)

#         U = rng.rand(n_samples, self.n_features_)

#         X = self.inverse_transform(U)
#         return X


# class KDEGaussianization(BaseTransform, DensityMixin):
#     """This performs a univariate transformation on a datasets.

#     Assuming that the data is independent across features, this
#     applies a transformation on each feature independently. The inverse
#     transformation is the marginal cdf applied to each of the features
#     independently and the inverse transformation is the marginal inverse
#     cdf applied to the features independently.
#     """

#     def __init__(
#         self,
#         method: str = "exact",
#         bw_estimator: str = "scott",
#         algorithm: str = "kd_tree",
#         kernel: str = "gaussian",
#         metric: str = "euclidean",
#         n_quantiles: int = 1_000,
#         support_extension: Union[int, float] = 10,
#         kwargs: Optional[Dict] = {},
#     ) -> None:
#         self.method = method
#         self.n_quantiles = n_quantiles
#         self.support_extension = support_extension
#         self.bw_estimator = bw_estimator
#         self.algorithm = algorithm
#         self.kernel = kernel
#         self.metric = metric
#         self.kwargs = kwargs

#     def fit(self, X, y=None):

#         # Uniformization
#         if self.method == "exact":
#             self.marg_uniformer_ = ScipyKDEUniformization(
#                 n_quantiles=self.n_quantiles,
#                 support_extension=self.support_extension,
#                 bw_estimator=self.bw_estimator,
#             )
#         elif self.method == "knn":
#             self.marg_uniformer_ = SklearnKDEUniformization(
#                 n_quantiles=self.n_quantiles,
#                 support_extension=self.support_extension,
#                 algorithm=self.algorithm,
#                 kernel=self.kernel,
#                 metric=self.metric,
#                 kwargs=self.kwargs,
#             )
#         else:
#             raise ValueError(f"Unrecognized method: {self.method}")

#         self.marg_uniformer_.fit(X)

#         return self

#     def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:

#         # check inputs
#         X = check_array(X, ensure_2d=True, copy=False)

#         # 1. Marginal Uniformization
#         X = self.marg_uniformer_.transform(X)

#         # 2. Marginal Gaussianization
#         X = InverseGaussCDF().transform(X)

#         return X

#     def log_abs_det_jacobian(
#         self, X: np.ndarray, y: Optional[np.ndarray] = None
#     ) -> np.ndarray:

#         # marginal uniformization
#         # print("X: ", X.min(), X.max())
#         Xu_der = self.marg_uniformer_.log_abs_det_jacobian(X)
#         # print("Xu Jacobian:", Xu_der.min(), Xu_der.max())
#         X = self.marg_uniformer_.transform(X)
#         # print("X_u:", X.min(), X.max())

#         # inverse CDF gaussian
#         # X = InverseGaussCDF().transform(X)
#         # print("Xg:", X.min(), X.max())

#         Xg_der = InverseGaussCDF().log_abs_det_jacobian(X)
#         # print("Xg jacobian:", Xg_der.min(), Xg_der.max())
#         # print(f"#Nans: {np.count_nonzero(~np.isnan(Xg_der))} / {Xg_der.shape[0]}")

#         return Xu_der + Xg_der

#     def inverse_transform(
#         self, X: np.ndarray, y: Optional[np.ndarray] = None
#     ) -> np.ndarray:

#         # check inputs
#         X = check_array(X, ensure_2d=True, copy=False)

#         # 1. Inverse Gaussianization
#         X = InverseGaussCDF().inverse_transform(X)

#         # 2. Inverse Uniformization
#         X = self.marg_uniformer_.inverse_transform(X)

#         return X

#     def score_samples(
#         self, X: np.ndarray, y: Optional[np.ndarray] = None
#     ) -> np.ndarray:
#         """Returns the log determinant abs jacobian of the inputs.

#         Parameters
#         ----------
#         X : np.ndarray
#             Inputs to be transformed

#         y: Not used, only for compatibility

#         """

#         # Marginal Gaussianization Transformation
#         # check inputs
#         X = check_array(X, ensure_2d=True, copy=False)

#         x_logprob = stats.norm().logpdf(self.transform(X))

#         return (x_logprob + self.log_abs_det_jacobian(X)).sum(axis=1)

#     def sample(
#         self, n_samples: int = 1, random_state: Optional[Union[RandomState, int]] = None
#     ) -> np.ndarray:
#         """Generate random samples from this.

#         Parameters
#         ----------
#         n_samples : int, default=1
#             The number of samples to generate.

#         random_state : int, RandomState,None, Optional, default=None
#             The int to be used as a seed to generate the random
#             uniform samples.

#         Returns
#         -------
#         X : np.ndarray, (n_samples, )
#         """
#         #
#         rng = check_random_state(random_state)

#         U = rng.rand(n_samples, self.n_features_)

#         X = self.inverse_transform(U)
#         return X
