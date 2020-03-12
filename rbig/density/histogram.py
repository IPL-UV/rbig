from typing import Tuple, Optional, Union, Callable
import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn.utils import check_array, check_random_state

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
from numpy.random import RandomState
from typing import Optional, Union, Callable
from rbig.base import UniformMixin, DensityMixin, ScoreMixin

TOL = 1e100


class Histogram(UniformMixin, DensityMixin, ScoreMixin):
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
        X : np.ndarray, (n_samples, 1)
        
        Returns
        -------
        self : instance of self
        """
        # calculate histogram
        # if self.nbins is None:
        #     self.nbins = int(np.sqrt(X.shape[0]))
        X = check_array(X, ensure_2d=False, copy=True)

        hist = np.histogram(X, bins=self.nbins)

        # calculate the rv-based on the histogram
        self.marg_u_transform = stats.rv_histogram(hist)
        return self

    def transform(self, X: np.ndarray, return_jacobian: bool = False) -> np.ndarray:
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

        X_trans = self._transform(X, self.marg_u_transform.cdf, inverse=False)

        # Return Jacobian
        if not return_jacobian:
            return X_trans
        else:
            return X_trans, self.log_abs_det_jacobian(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform which is the Inverse CDF function
        applied to the samples

            x = P^-1(z)
            P^-1() - Inverse CDF, PPF

        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
        
        Returns
        -------
        X : np.ndarray, (n_samples, 1)
            Transformed data
        """
        X = check_array(X, ensure_2d=False, copy=True)

        return self._transform(X, self.marg_u_transform.ppf, inverse=True)

    def _transform(
        self,
        X: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
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
            lower_bound_y = self.marg_u_transform._hcdf[0]
            upper_bound_y = self.marg_u_transform._hcdf[-1]
        else:
            # get boundaries
            lower_bound_x = self.marg_u_transform._hcdf[0]
            upper_bound_x = self.marg_u_transform._hcdf[-1]
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

    def abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """The log determinant jacobian of the distribution.
        
            dz/dx = pdf
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
        
        Returns
        -------
        log_det : np.ndarray, (n_samples, 1)
        """
        X_prob = self.marg_u_transform.pdf(X)

        # Add regularization of PDF estimates
        X_prob[X_prob <= 0.0] = self.alpha
        return X_prob

    # def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
    #     """The log determinant jacobian of the distribution.

    #     log_det = log dz/dx = log pdf

    #     Parameters
    #     ----------
    #     X : np.ndarray, (n_samples, 1)

    #     Returns
    #     -------
    #     log_det : np.ndarray, (n_samples, 1)
    #     """
    #     return np.log(self.abs_det_jacobian(X))

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """Calculates the log probability
        
        log prob = p(z) * absdet J
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            input samples
        
        Returns
        -------
        x_prob : np.ndarray, (n_samples, 1)
        """
        return self.abs_det_jacobian(X)

    # def logpdf(self, X: np.ndarray) -> np.ndarray:
    #     """Calculates the log probability

    #     log prob = log pdf p(z) + log det

    #     Parameters
    #     ----------
    #     X : np.ndarray, (n_samples, 1)
    #         input samples

    #     Returns
    #     -------
    #     x_prob : np.ndarray, (n_samples, 1)
    #     """
    #     return self.log_abs_det_jacobian(X)

    def score_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Returns the log likelihood. It
        calculates the mean of the log probability.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
         incoming samples
        
        Returns
        -------
        nll : float,
            the mean of the log probability
        """
        return self.log_abs_det_jacobian(X)

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

        U = rng.rand(n_samples, 1)

        X = self.inverse_transform(U)
        return X
