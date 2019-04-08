import numpy as np
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from sklearn.utils import check_random_state
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import column_or_1d
import scipy.stats
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class HistogramUnivariateDensity(BaseEstimator, TransformerMixin):
    def __init__(self, bins="auto", bounds=0.1, alpha=1e-6):
        self.bins = bins
        self.bounds = bounds
        self.alpha = alpha

    def fit(self, X, y=None, histogram_params=None):

        # Check X
        X = self.check_X(X)

        # Check Bounds
        bounds = check_bounds(X, self.bounds)

        # fit numpy histogram
        hist, bin_edges = np.histogram(X, bins=self.bins, range=bounds)

        # ========================
        # Regularization
        # ========================
        hist = np.array(hist, dtype=float)
        hist += self.alpha

        # Convert to scipy RV
        self.rv = scipy.stats.rv_histogram((hist, bin_edges))
        return self

    def transform(self, X):
        X = self.check_X(X)
        return self.rv.cdf(X.ravel()).reshape((-1, 1))

    def inverse_transform(self, X):
        X = self.check_X(X)
        return self.rv.ppf(X.ravel()).reshape((-1, 1))

    def logdetjacobian(self, X):
        X = self.check_X(X)
        return self.rv.logpdf(X.ravel()).reshape((-1, 1))

    def get_support(self, X):
        # Assumes density is univariate
        return np.array([[self.rv.a, self.rv.b]])

    def sample(self, n_samples=1, random_state=None):

        rng = check_random_state(random_state)
        return np.array(self.rv.rvs(size=n_samples, random_state=rng)).reshape(
            (n_samples, 1)
        )

    def entropy(self, X):
        X = self.check_X(X)
        return self.rv.entropy(X.ravel()).reshape((-1, 1))

    def check_X(self, X, inverse=False):

        bounds = check_bounds(X, self.bounds)

        if np.any(X <= bounds[0]) or np.any(X >= bounds[1]):
            warnings.warn(
                BoundaryWarning(
                    "Input to random variable function has at least one value outside of bounds "
                    "but all inputs should be in (bounds[0], bounds[1]) exclusinve. Bounding "
                    "values away from bounds[0] and bounds[1]."
                )
            )

            X = make_interior(X, bounds)
        return X


class InverseCDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        pass

    def inverse_transform(self, X, y=None):
        pass

    def logdetjacobian(self, X, y=None):
        pass


def check_bounds(X=None, bounds=([-np.inf, np.inf]), extend=True):
    """Checks the bounds. Since we are going from an unbound domain to
    a bounded domain (Random Dist to Uniform Dist) we are going to have
    a problem with defining the boundaries. This function will either 
    have set boundaries or extend the boundaries with with a percentage.
    
    Parameters
    ----------
    X : array-like, default=None
    
    bounds : int or array-like [low, high]
    
    extend : bool, default=True
    
    Returns
    -------
    bounds : array-like
    
    References
    ---------
    https://github.com/davidinouye/destructive-deep-learning/blob/master/ddl/univariate.py#L506
    """

    default_support = np.array([-np.inf, np.inf])

    # Case I - Extension
    if np.isscalar(bounds):

        if X is None:
            # If no X, return default support (unbounded domain)
            return default_support
        else:
            # extend the domain by x percent
            percent_extension = bounds

            # Get the min and max for the current domain
            domain = np.array([np.min(X), np.max(X)])

            # Get the mean value of the domain
            center = np.mean(domain)

            # Extend the domain on either sides
            domain = (1 + percent_extension) * (domain - center) + center

            return domain

    # Case II - Directly compute
    else:
        domain = column_or_1d(bounds).copy()

        if domain.shape[0] != 2:
            raise ValueError(
                "Domain should either be a two element array-like"
                " or a scalar indicating percentage extension of domain."
            )

        return domain


def make_interior(X, bounds, eps=None):
    """Scale/Shift data to fit in the open interval given by bounds.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix
    
    bounds : array-like, shape (2,)
        Minimum and maximum of bounds.
    
    eps : float, optional
        Epsilon for clipping, defaults to ``np.info(X.dtype).eps``
        
    Returns
    -------
    X : array, shape (n_samples, n_features)
        Data matrix after possible modification
    """
    X = check_floating(X)

    if eps is None:
        eps = np.finfo(X.dtype).eps

    left = bounds[0] + np.abs(bounds[0] * eps)
    right = bounds[1] - np.abs(bounds[1] * eps)
    return np.minimum(np.maximum(X, left), right)


def check_floating(X):
    if not np.issubdtype(X.dtype, np.floating):
        X = np.array(X, dtype=np.float)
    return X


def make_interior_probability(X, eps=None):
    """Convert data to probability values in the open interval between 0 and 1.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.
    eps : float, optional
        Epsilon for clipping, defaults to ``np.info(X.dtype).eps``
    Returns
    -------
    X : array, shape (n_samples, n_features)
        Data matrix after possible modification.
    """
    X = check_floating(X)
    if eps is None:
        eps = np.finfo(X.dtype).eps
    return np.minimum(np.maximum(X, eps), 1 - eps)


def make_finite(X):
    """Make the data matrix finite by replacing -infty and infty.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.
    Returns
    -------
    X : array, shape (n_samples, n_features)
        Data matrix as numpy array after checking and possibly replacing
        -infty and infty with min and max of floating values respectively.
    """
    X = check_floating(X)
    return np.minimum(np.maximum(X, np.finfo(X.dtype).min), np.finfo(X.dtype).max)


def make_positive(X):
    """Make the data matrix positive by clipping to +epsilon if not positive.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.
    Returns
    -------
    X : array, shape (n_samples, n_features)
        Data matrix as numpy array after checking and possibly replacing
        non-positive numbers to +epsilon.
    """
    X = check_floating(X)
    return np.maximum(X, np.finfo(X.dtype).tiny)


class BoundaryWarning(DataConversionWarning):
    """Warning that data is on the boundary of the required set.
    Warning when data is on the boundary of the domain or range and
    is converted to data that lies inside the boundary. For example, if
    the domain is (0,inf) rather than [0,inf), values of 0 will be made
    a small epsilon above 0.
    """


def get_data(seed=123, n_samples=1000, noise=0.25):
    rng = np.random.RandomState(seed=seed)

    X = np.abs(2 * rng.randn(n_samples, 1))
    Y = np.sin(X) + noise * rng.randn(n_samples, 1)

    return X, Y


def example():
    X, _ = get_data()

    seed = 123
    rng = np.random.RandomState(seed=seed)

    num_samples = 1000
    noise = 0.25

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(X, marker=".", linestyle="")
    ax.set_title("Original Data")
    plt.show()

    # Parameters
    alpha = 1e-6  # regularization parameter
    bounds = 0.01  # percentage extension
    bins = "auto"  # number of bins or bins estimator parameter

    histogram_params = None

    # Initialize Univariate Normalization
    uni_model = HistogramUnivariateDensity(alpha=alpha, bounds=bounds, bins=bins)

    # transform
    uni_model.fit(X)

    # Initialize Model
    uni_model = HistogramUnivariateDensity(alpha=alpha, bounds=bounds, bins=bins)

    # Fit Model to Data
    uni_model.fit(X)

    # Transform Data
    X_uni_approx = uni_model.transform(X)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(X_uni_approx, marker=".", linestyle="")
    ax.set_title("Data in Uniform Domain")
    plt.show()

    # Inverse Transform Data
    X_uni = rng.uniform(0, 1, (1000, 1))
    X_approx = uni_model.inverse_transform(X_uni)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(X_approx, marker=".", linestyle="")
    ax.set_title("Data Generated data domain")
    plt.show()

    # Generate samples from domain
    X_uni_samples = uni_model.sample(n_samples=1000)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(X_uni_samples, marker=".", linestyle="")
    ax.set_title("Samples from Uniform Domain")
    plt.show()

    # Calculate Log Jacobian
    mll = uni_model.logdetjacobian(X).sum()
    print(mll)

    # Calculate Entropy
    ent = uni_model.entropy(X).sum()
    print(ent)
    return None


def main():
    pass


if __name__ == "__main__":
    example()
    pass
