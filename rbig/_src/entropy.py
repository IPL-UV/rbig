from typing import Union
import numpy as np
from scipy.stats import rv_histogram
from astropy.stats import histogram as astro_hist


def entropy_univariate(
    X: np.ndarray, bins: Union[int, str] = "auto", correction: bool = True
) -> np.ndarray:

    # Get histogram (use default bin estimation)
    # create histogram
    if bins in ["blocks", "knuth"]:
        hist = astro_hist(X, bins=bins, range=(X.min(), X.max()))
    else:
        hist = np.histogram(X, bins=bins, range=(X.min(), X.max()))
    # hist = np.histogram(
    #     a=X,
    #     bins=bins,
    #     range=(X.min(), X.max()),
    # )

    # Calculate differential entropy
    H = rv_histogram(hist).entropy()

    if correction:
        H += 0.5 * (np.sum(hist[0] > 0) - 1) / np.sum(hist[0])

    return H


def entropy_marginal(
    data: np.ndarray, bins="auto", correction: bool = True
) -> np.ndarray:
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

        H[idim] = entropy_univariate(i_data, bins=bins, correction=correction)

    return H


def entropy_rbig(
    X: np.ndarray,
    bins: str = "auto",
    alpha: float = 1e-10,
    bound_ext: float = 0.3,
    eps: float = 1e-10,
    rotation: str = "PCA",
    zero_tolerance: int = 60,
    max_layers: int = 1_000,
):
    from rbig._src.total_corr import rbig_total_corr

    # total correlation using RBIG, TC(X)
    tc_rbig = rbig_total_corr(
        X=X,
        bins=bins,
        alpha=alpha,
        bound_ext=bound_ext,
        eps=eps,
        rotation=rotation,
        zero_tolerance=zero_tolerance,
        max_layers=max_layers,
    )

    # marginal entropy using rbig, H(X)
    Hx = entropy_marginal(X)

    # Multivariate entropy, H(X) - TC(X)
    return Hx.sum() - tc_rbig
