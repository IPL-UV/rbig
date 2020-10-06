from typing import Optional, Tuple, Union

import numpy as np


def get_support_reference(
    support: np.ndarray, extension: Union[float, int], n_quantiles: int = 1_000
) -> np.ndarray:

    lb, ub = get_domain_extension(support, extension)

    # get new support
    new_support = np.linspace(lb, ub, n_quantiles, endpoint=True)

    return new_support


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


def generate_batches(n_samples, batch_size):
    """A generator to split an array of 0 to n_samples
    into an array of batch_size each.

    Parameters
    ----------
    n_samples : int
        the number of samples

    batch_size : int,
        the size of each batch


    Returns
    -------
    start_index, end_index : int, int
        the start and end indices for the batch

    Source:
        https://github.com/scikit-learn/scikit-learn/blob/master
        /sklearn/utils/__init__.py#L374
    """
    start_index = 0

    # calculate number of batches
    n_batches = int(n_samples // batch_size)

    for _ in range(n_batches):

        # calculate the end coordinate
        end_index = start_index + batch_size

        # yield the start and end coordinate for batch
        yield start_index, end_index

        # start index becomes new end index
        start_index = end_index

    # special case at the end of the segment
    if start_index < n_samples:

        # yield the remaining indices
        yield start_index, n_samples
