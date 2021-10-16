from typing import Union
import numpy as np
from scipy.stats import rv_histogram


def univariate_entropy(X: np.ndarray, bins: Union[int, str]="auto", correction: bool = True) -> np.ndarray:

    # Get histogram (use default bin estimation)
    hist = np.histogram(
        a=X,
        bins=bins,
        range=(X.min(), X.max()),
    )

    # Calculate differential entropy
    H = rv_histogram(hist).entropy()

    if correction:
        H += 0.5 * (np.sum(hist[0] > 0) - 1) / np.sum(hist[0])

    return H


def entropy_marginal(data: np.ndarray, bins="auto", correction: bool = True) -> np.ndarray:
    """Calculates the marginal entropy (the entropy per dimension) of a
    multidimensional dataset. Uses histogram bin counnts. Also features
    and option to add the Shannon-Miller correction.

    Parameters
    ----------
    data : array, (n_samples x d_dimensions)

    bin_est : str, (default='standard')
        The bin estimation method.
        {'standard', 'sturge'}

    correction : bool, default=True

    Returns
    -------
    H : array (d_dimensions)

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    """
    d_dimensions = data.shape[1]

    H = np.zeros(d_dimensions)

    for idim, i_data in enumerate(data.T):

        H[idim] = univariate_entropy(i_data, bins=bins, correction=correction)

    return H
