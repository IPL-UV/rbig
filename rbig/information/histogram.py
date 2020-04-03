from typing import Union, Optional, Dict
import numpy as np
from scipy import stats


def hist_entropy(
    X: np.ndarray,
    bins: Union[str, int] = "auto",
    correction: bool = True,
    hist_kwargs: Optional[Dict] = {},
) -> float:
    """Calculates the entropy using the histogram of a univariate dataset.
    Option to do a Miller Maddow correction.
    
    Parameters
    ----------
    X : np.ndarray, (n_samples)
        the univariate input dataset
    
    bins : {str, int}, default='auto'
        the number of bins to use for the histogram estimation
    
    correction : bool, default=True
        implements the Miller-Maddow correction for the histogram
        entropy estimation.
    
    hist_kwargs: Optional[Dict], default={}
        the histogram kwargs to be used when constructing the histogram
        See documention for more details:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

    Returns
    -------
    H_hist_entropy : float
        the entropy for this univariate histogram

    Example
    -------
    >> from scipy import stats
    >> from pysim.information import histogram_entropy
    >> X = stats.gamma(a=10).rvs(1_000, random_state=123)
    >> histogram_entropy(X)
    array(2.52771628)
    """
    # get histogram
    hist_counts = np.histogram(X, bins=bins, **hist_kwargs)

    # create random variable
    hist_dist = stats.rv_histogram(hist_counts)

    # calculate entropy
    H = hist_dist.entropy()

    # MLE Estimator with Miller-Maddow Correction
    if correction == True:
        H += 0.5 * (np.sum(hist_counts[0] > 0) - 1) / hist_counts[0].sum()

    return H
