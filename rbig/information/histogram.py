from typing import Union, Optional, Dict
import numpy as np
from scipy import stats
from rbig.information.base import PDFEstimator
from rbig.utils import make_interior_log_prob, make_interior_probability
from sklearn.utils import check_array
from rbig.utils import get_domain_extension


def hist_entropy(
    X: np.ndarray,
    bins: Union[str, int] = "auto",
    correction: bool = True,
    support_extension: Union[int, float] = 10,
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

    # get support bounds
    support_bounds = get_domain_extension(X, support_extension)

    # get histogram
    hist_counts = np.histogram(X, bins=bins, range=support_bounds, **hist_kwargs)

    # create random variable
    hist_dist = stats.rv_histogram(hist_counts)

    # calculate entropy
    H = hist_dist.entropy()

    # MLE Estimator with Miller-Maddow Correction
    if correction is True:
        H += 0.5 * (np.sum(hist_counts[0] > 0) - 1) / hist_counts[0].sum()

    return H


class ScipyHistogram(PDFEstimator):
    def __init__(
        self,
        bins: Optional[int] = None,
        alpha: float = 1e-5,
        prob_tol: float = 1e-7,
        support_extension: Union[float, int] = 10,
        kwargs: Dict = {},
    ) -> None:
        self.bins = bins
        self.alpha = alpha
        self.prob_tol = prob_tol
        self.support_extension = support_extension
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        X = check_array(X, ensure_2d=True, copy=True)

        # get support bounds
        support_bounds = get_domain_extension(X, self.support_extension)

        # calculate histogram
        print("Support Bounds:", support_bounds)
        hist = np.histogram(
            X.squeeze(), bins=self.bins, range=support_bounds, **self.kwargs
        )

        # needed for the entropy calculation
        self.hist_counts_ = hist[0]

        # fit model
        self.estimator_ = stats.rv_histogram(hist)
        print("Histogram Support:", self.estimator_.support())

        # Add some noise to the pdfs to prevent zero probability
        self.estimator_._hpdf += self.alpha

        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:

        X_prob = self.estimator_.pdf(X.squeeze())

        X_prob = make_interior_probability(X_prob, eps=self.prob_tol)
        return X_prob

    def logpdf(self, X: np.ndarray) -> np.ndarray:

        return np.log(self.pdf(X))

    def cdf(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.cdf(X.squeeze())

    def ppf(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.ppf(X.squeeze())

    def entropy(self, correction: bool = True) -> float:
        # calculate entropy
        H = self.estimator_.entropy()

        # MLE Estimator with Miller-Maddow Correction
        if correction is True:
            H += 0.5 * (np.sum(self.hist_counts_ > 0) - 1) / self.hist_counts_.sum()

        return H
