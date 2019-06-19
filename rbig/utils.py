import numpy as np
import warnings
from sklearn.utils import check_random_state
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import column_or_1d


class BoundaryWarning(DataConversionWarning):
    """Warning that data is on the boundary of the required set.
    Warning when data is on the boundary of the domain or range and
    is converted to data that lies inside the boundary. For example, if
    the domain is (0,inf) rather than [0,inf), values of 0 will be made
    a small epsilon above 0.
    """

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


def bin_estimation(X, rule="scott"):

    n_samples = X.shape[0]

    if rule == "sqrt":
        nbins = np.sqrt(n_samples)
    elif rule == "scott":
        nbins = (3.49 * np.std(X)) / np.cbrt(n_samples)
    elif rule == "sturge":
        nbins = 1 + np.log2(n_samples)
    elif rule == "rice":
        nbins = 2 * np.cbrt(n_samples)
    else:
        raise ValueError(f"Unrecognized rule: {rule}")

    return int(np.ceil(nbins))
