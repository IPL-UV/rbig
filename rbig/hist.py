from typing import Tuple, Optional, Union, Callable
import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn.utils import check_array, check_random_state


from sklearn.utils import check_array, check_random_state
from numpy.random import RandomState
from typing import Optional, Union, Callable
from .base import UniformMixin

TOL = 1e100


class Histogram(UniformMixin):
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

        return self._transform(X, self.marg_u_transform.cdf, inverse=False)

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

    def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """The log determinant jacobian of the distribution.
        
        log_det = log dz/dx = log pdf
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
        
        Returns
        -------
        log_det : np.ndarray, (n_samples, 1)
        """
        return np.log(self.abs_det_jacobian(X))

    def prob(self, X: np.ndarray) -> np.ndarray:
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

    def logprob(self, X: np.ndarray) -> np.ndarray:
        """Calculates the log probability
        
        log prob = log pdf p(z) + log det
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            input samples
        
        Returns
        -------
        x_prob : np.ndarray, (n_samples, 1)
        """
        return self.log_abs_det_jacobian(X)

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Returns the negative log likelihood. It
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
        return -self.log_abs_det_jacobian(X).mean()

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


class MarginalUniformization:
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

        return self._transform(X, self.marg_u_transform.cdf, inverse=False)

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

    def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """The log determinant jacobian of the distribution.
        
        log_det = log dz/dx = log pdf
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
        
        Returns
        -------
        log_det : np.ndarray, (n_samples, 1)
        """
        return np.log(self.abs_det_jacobian(X))

    def prob(self, X: np.ndarray) -> np.ndarray:
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

    def logprob(self, X: np.ndarray) -> np.ndarray:
        """Calculates the log probability
        
        log prob = log pdf p(z) + log det
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            input samples
        
        Returns
        -------
        x_prob : np.ndarray, (n_samples, 1)
        """
        return self.log_abs_det_jacobian(X)

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Returns the negative log likelihood. It
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
        return -self.log_abs_det_jacobian(X).mean()

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


def main():
    from scipy import stats
    import matplotlib.pyplot as plt

    seed = 123
    n_samples = 1000
    a = 10

    # initialize data distribution
    data_dist = stats.gamma(a=a)

    # get some samples
    X_samples = data_dist.rvs(size=n_samples)
    # X_samples = np.array([1.0, 2.0, 1.0])

    # initialize HistogramClass
    nbins = int(np.sqrt(n_samples))
    bounds = None
    hist_clf = HistogramUniformization(nbins=nbins)

    # fit to data
    hist_clf.fit(X_samples)

    # ========================
    # Transform Data Samples
    # ========================

    # transform data
    Xu = hist_clf.transform(X_samples)

    fig, ax = plt.subplots()
    ax.hist(Xu, nbins)
    ax.set_xlabel(r"$F(x)$")
    ax.set_ylabel(r"$p(u)$")
    plt.show()

    # ========================
    # Generate Uniform Samples
    # ========================
    u_samples = stats.uniform().rvs(size=1000)
    X_approx = hist_clf.inverse_transform(u_samples)

    fig, ax = plt.subplots()
    ax.hist(X_approx, nbins)
    ax.set_xlabel(r"$F^{-1}(u)$")
    ax.set_ylabel(r"$p(x_d)$")
    plt.show()

    # ========================
    # Generate Uniform Samples
    # ========================
    X_approx = hist_clf.sample(1000)

    fig, ax = plt.subplots()
    ax.hist(X_approx, nbins)
    ax.set_xlabel(r"$F^{-1}(u)$")
    ax.set_ylabel(r"$p(x_d)$")
    plt.show()
    # ========================
    # Evaluate Probability
    # ========================
    print(X_samples[:10].shape)
    x_prob = hist_clf.prob(X_samples[:10])
    data_prob = data_dist.pdf(X_samples[:10])
    print(x_prob)
    print(data_prob)

    # ========================
    # Evaluate Log Probability
    # ========================
    x_score = hist_clf.score(X_samples)
    data_score = -np.log(data_dist.pdf(X_samples)).mean()
    print(x_score, data_score)

    return None


if __name__ == "__main__":
    main()

