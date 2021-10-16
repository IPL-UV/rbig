import numpy as np
from typing import Union, Tuple


def get_support_reference(
    support: np.ndarray, extension: Union[float, int], n_quantiles: int = 1_000
) -> np.ndarray:

    lb, ub = get_domain_extension(support, extension)

    # get new support
    new_support = np.linspace(lb, ub, n_quantiles, endpoint=True)

    return new_support


def get_domain_extension(
    data: np.ndarray,
    extension: Union[float, int],
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


def check_floating(X):
    if not np.issubdtype(X.dtype, np.floating):
        X = np.array(X, dtype=np.float)
    return X


def make_interior_uniform_probability(X, eps=None):
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


def make_interior_log_prob(X: np.ndarray, eps=None):
    X = check_floating(X)
    if eps is None:
        eps = np.finfo(X.dtype).eps
    # remove numbers that are close to zero
    X[X <= -np.inf] = -np.finfo(X.dtype).eps

    return X


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

    X[X < left] = left
    X[X > right] = right

    # assert np.min(X) >= left
    # assert np.max(X) <= right

    return X
