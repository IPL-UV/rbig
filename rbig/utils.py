import warnings
from typing import Tuple, Union

import numpy as np
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_random_state
from sklearn.utils.validation import column_or_1d


def make_interior_log_prob(X: np.ndarray):

    # remove numbers that are close to zero
    X[X <= -np.inf] = -np.finfo(X.dtype).eps

    return X


def get_domain_extension(
    data: np.ndarray, extension: Union[float, int],
) -> Tuple[float, float]:

    if isinstance(extension, float):
        pass
    elif isinstance(extension, int):
        extension /= 100
    else:
        raise ValueError(f"Unrecognized type extension: {type(extension)}")

    domain = np.abs(data.max() - data.min())

    domain_ext = extension * domain

    lower_bound = data.min() - domain_ext
    upper_bound = data.max() + domain_ext

    return lower_bound, upper_bound


def get_support_reference(
    support: np.ndarray, extension: Union[float, int], n_quantiles: int = 1_000
) -> np.ndarray:

    lb, ub = get_domain_extension(support, extension)

    # get new support
    new_support = np.linspace(lb, ub, n_quantiles, endpoint=True)

    return new_support


def interp_support():
    return None


def check_input_output_dims(
    input: np.ndarray, dims: Tuple[int, int], method: str, transform: str
) -> None:
    assert input.shape == (
        dims[0],
        dims[1],
    ), f"{method.capitalize()}: {transform.capitalize()} lost dims, {input.shape} =/= {dims}"
    return None


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


def make_cdf_monotonic(cdf):
    """
    Take a cdf and just sequentially readjust values to force monotonicity
    There's probably a better way to do this but this was in the original
    implementation. We just readjust values that are less than their predecessors
    Parameters
    ----------
    cdf : ndarray
      The values of the cdf in order (1d)
    """
    # laparra's version
    corrected_cdf = cdf.copy()
    for i in range(1, len(corrected_cdf)):
        if corrected_cdf[i] <= corrected_cdf[i - 1]:
            if abs(corrected_cdf[i - 1]) > 1e-14:
                corrected_cdf[i] = corrected_cdf[i - 1] + 1e-14
            elif corrected_cdf[i - 1] == 0:
                corrected_cdf[i] = 1e-80
            else:
                corrected_cdf[i] = corrected_cdf[i - 1] + 10 ** (
                    np.log10(abs(corrected_cdf[i - 1]))
                )
    return corrected_cdf
