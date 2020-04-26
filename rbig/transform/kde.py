from typing import Callable, Optional, Tuple, Union, Dict

import numpy as np
from numpy.random import RandomState
from scipy import stats
from rbig.information.histogram import ScipyHistogram

# Base classes
from sklearn.utils import check_array, check_random_state

from rbig.transform.base import DensityMixin, BaseTransform
from rbig.density.kde import KDEScipy, KDESklearn
from rbig.utils import (
    get_domain_extension,
    bin_estimation,
    make_interior,
    check_input_output_dims,
)

BOUNDS_THRESHOLD = 1e-7


class ScipyKDEUniformization(BaseTransform, DensityMixin):
    def __init__(
        self,
        n_quantiles: int = 1_000,
        support_extension: Union[int, float] = 10,
        bw_estimator: str = "scott",
    ) -> None:
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension
        self.bw_estimator = bw_estimator

    def fit(self, X: np.ndarray) -> None:
        """Finds an empirical distribution based on the
        histogram approximation.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
        
        Returns
        -------
        self : instance of self
        """
        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        self.n_samples_, self.n_features_ = X.shape

        # initialize histogram
        marginal_transforms = list()

        #
        # Loop through features
        for feature in X.T:

            estimator = KDEScipy(
                bw_method=self.bw_estimator,
                n_quantiles=self.n_quantiles,
                support_extension=self.support_extension,
            )

            # save marginal transform
            marginal_transforms.append(estimator.fit(feature))

        # calculate the rv-based on the histogram
        self.marginal_transforms_ = marginal_transforms

        return self

    def _transform(self, X: np.ndarray, inverse: bool = False) -> np.ndarray:

        for feature_idx in range(X.shape[1]):

            # transform each column
            if inverse:
                X[:, feature_idx] = self.marginal_transforms_[feature_idx].ppf(
                    X[:, feature_idx]
                )
            else:
                X[:, feature_idx] = self.marginal_transforms_[feature_idx].cdf(
                    X[:, feature_idx]
                )
        return X

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
        X = check_array(X, ensure_2d=True, copy=True)

        n_samples = X.shape[0]

        X = self._transform(X, inverse=False)

        check_input_output_dims(X, (n_samples, self.n_features_), "KDE", "Foward")
        return X

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
        X = check_array(X, ensure_2d=True, copy=True)

        n_samples = X.shape[0]

        X = self._transform(X, inverse=True)

        check_input_output_dims(
            X, (n_samples, self.n_features_), "Histogram", "Inverse"
        )

        return X

    def score_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Returns the log likelihood. It
        calculates the mean of the log probability.
        """
        log_prob = self.log_abs_det_jacobian(X, y).sum(axis=1).reshape(-1, 1)

        check_input_output_dims(log_prob, (X.shape[0], 1), "Histogram", "Log Prob")

        return log_prob

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
        X = check_array(X, ensure_2d=True, copy=True)

        n_samples = X.shape[0]

        log_scores = list()

        for feature_idx in range(X.shape[1]):

            # transform each column
            iscore = self.marginal_transforms_[feature_idx].logpdf(X[:, feature_idx])
            log_scores.append(iscore)

        # print(f"hstack: {np.hstack(log_scores).shape}")
        # print(f"vstack: {np.vstack(log_scores).shape}")
        log_scores = np.vstack(log_scores).T

        assert log_scores.shape == (
            n_samples,
            self.n_features_,
        ), f"KDE Scipy: Jacobian lost dims, {X.shape}"

        check_input_output_dims(
            log_scores, (X.shape[0], self.n_features_), "Histogram", "Jacobian"
        )

        return log_scores

    def _clip_infs(self, f: Callable) -> Tuple[float, float]:
        clip_min = f(BOUNDS_THRESHOLD - np.spacing(1))
        clip_max = f(1 - (BOUNDS_THRESHOLD - np.spacing(1)))

        return clip_min, clip_max

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


class SklearnKDEUniformization(ScipyKDEUniformization):
    def __init__(
        self,
        algorithm: str = "kd_tree",
        kernel: str = "gaussian",
        metric: str = "euclidean",
        n_quantiles: int = 1_000,
        support_extension: Union[int, float] = 10,
        kwargs: Optional[Dict] = {},
    ) -> None:
        super().__init__(n_quantiles=n_quantiles, support_extension=support_extension)
        self.algorithm = algorithm
        self.kernel = kernel
        self.metric = metric
        self.kwargs = kwargs

    def fit(self, X: np.ndarray) -> None:
        """Finds an empirical distribution based on the
        histogram approximation.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
        
        Returns
        -------
        self : instance of self
        """
        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        self.n_samples_, self.n_features_ = X.shape

        # initialize histogram
        marginal_transforms = list()

        #
        # Loop through features
        for feature in X.T:

            estimator = KDESklearn(
                algorithm=self.algorithm,
                kernel=self.kernel,
                metric=self.metric,
                kwargs=self.kwargs,
                n_quantiles=self.n_quantiles,
                support_extension=self.support_extension,
            )

            # save marginal transform
            marginal_transforms.append(estimator.fit(feature[:, None]))

        # calculate the rv-based on the histogram
        self.marginal_transforms_ = marginal_transforms

        return self
