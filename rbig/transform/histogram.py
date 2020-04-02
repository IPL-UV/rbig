from typing import Tuple, Optional, Union, Callable
import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn.utils import check_array, check_random_state

# Base classes
from sklearn.base import BaseEstimator
from rbig.base import DensityMixin, DensityTransformerMixin

from sklearn.utils import check_array, check_random_state
from numpy.random import RandomState
from typing import Optional, Union, Callable
from rbig.utils import check_input_output_dims


BOUNDS_THRESHOLD = 1e-7


class MarginalHistogramTransform(BaseEstimator, DensityTransformerMixin, DensityMixin):
    def __init__(
        self, nbins: Optional[Union[int, str]] = "auto", alpha: float = 1e-5
    ) -> None:
        self.nbins = nbins
        self.alpha = alpha

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

            # calculate histogram
            hist, edges = np.histogram(feature, bins=self.nbins)

            # add some regularization
            hist = hist.astype(np.float64)
            hist += self.alpha

            # save marginal transform
            marginal_transforms.append(stats.rv_histogram((hist, edges)))

        # calculate the rv-based on the histogram
        self.marginal_transforms_ = marginal_transforms

        return self

    def _transform(self, X: np.ndarray, inverse: bool = False) -> np.ndarray:

        for feature_idx in range(X.shape[1]):

            # transform each column
            if inverse:
                X[:, feature_idx] = self._transform_feature(
                    X[:, feature_idx],
                    self.marginal_transforms_[feature_idx].ppf,
                    self.marginal_transforms_[feature_idx]._hcdf,
                    inverse=inverse,
                )
            else:
                X[:, feature_idx] = self._transform_feature(
                    X[:, feature_idx],
                    self.marginal_transforms_[feature_idx].cdf,
                    self.marginal_transforms_[feature_idx]._hcdf,
                    inverse=inverse,
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

        check_input_output_dims(X, (n_samples, self.n_features_), "Histogram", "Foward")
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

    def _transform_feature(
        self,
        X: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        quantiles: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """Internal function that handles the boundary issues that can occur
        when doing the transformations. Wraps the transform and inverse
        transform functions.
        """
        if inverse == True:
            # get boundaries
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
        else:
            # get boundaries
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1

        # find indices for upper and lower bounds
        lower_bounds_idx = X == lower_bound_x
        upper_bounds_idx = X == upper_bound_x

        # mask out infinite values
        isfinite_mask = ~np.isnan(X)
        X_col_finite = X[isfinite_mask]

        X[isfinite_mask] = f(X_col_finite)

        # set bounds
        X[upper_bounds_idx] = upper_bound_y
        X[lower_bounds_idx] = lower_bound_y

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

        log_scores = list()

        for feature_idx in range(X.shape[1]):

            # transform each column
            X[:, feature_idx] = np.clip(
                X[:, feature_idx],
                self.marginal_transforms_[feature_idx].a + 10 * np.spacing(1),
                self.marginal_transforms_[feature_idx].b - 10 * np.spacing(1),
            )

            iscore = self.marginal_transforms_[feature_idx].logpdf(X[:, feature_idx])
            log_scores.append(iscore)

        # print(f"hstack: {np.hstack(log_scores).shape}")
        # print(f"vstack: {np.vstack(log_scores).shape}")
        log_scores = np.vstack(log_scores).T

        assert log_scores.shape == (
            self.n_samples_,
            self.n_features_,
        ), f"Histogram: Jacobian lost dims, {X.shape}"

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
