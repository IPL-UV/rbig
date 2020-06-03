from typing import Union, Optional, Dict, Tuple
import numpy as np
from scipy import stats
from rbig.density.base import PDFEstimator
from rbig.utils import make_interior_log_prob, make_interior_probability
from sklearn.utils import check_array
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_random_state
import logging
from rbig.density.empirical import estimate_empirical_cdf
from rbig.utils import get_support_reference, get_domain_extension

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def hist_est_pdf(
    X: np.ndarray,
    bins: Union[int, str] = "auto",
    alpha: float = 1e-5,
    support_extension: Union[float, int] = 10,
    support_bounds: Optional[Tuple[float, float]] = None,
    kwargs: Dict = {},
) -> np.ndarray:
    X = check_array(X, ensure_2d=False, copy=True)

    # get support bounds
    if support_bounds is None:
        support_bounds = get_domain_extension(X, support_extension)

    # calculate histogram
    hist = np.histogram(X.squeeze(), bins=bins, range=support_bounds, **kwargs)

    hpdf = np.asarray(hist[0], dtype=np.float64)

    # add zeros for regularization (no zero probability)
    hpdf += alpha
    hbins = np.asarray(hist[1], dtype=np.float64)
    hbin_widths = hbins[1:] - hbins[:-1]
    hpdf = hpdf / float(np.sum(hpdf * hbin_widths))
    hpdf = np.hstack([alpha, hpdf])
    return hpdf, hbins


class ScipyHistogram(PDFEstimator):
    """Univariate histogram density estimator.
    A light wrapper around the scipy `stats.rv_histogram` function
    which calculates the empirical distribution. After this has
    been fitted, this method will have the standard density
    functions like pdf, logpdf, cdf, ppf. All of these are
    necessary for the transformations.

    Parameters
    ----------
    bins : Union[int, str], default='auto'
        the number of bins to estimated the histogram function.
        see `np.histogram` for more options.
    alpha : float, default=1e-5
        the amount of regularization to add to the estimated PDFs
        so that there are no zero probabilities.
    prob_tol : float, default=1e-7
        this controls how much we clip any data in the outside bounds
    support_extension : int, default=10
        the amount to extend the support of the fitted data X. Affects
        the PDF,CDF, and PPF functions. Extending the support will allow
        more data to be interpolated.
    kwargs : Dict[str,Any], default={}
        any extra kwargs to be passed into the `np.histogram` estimator.
    
    Attributes
    ----------
    estimator_ : PDFEstimator
        a fitted histogram estimator with a PDFEstimator base class.
        This method will have pdf, logpdf, cdf and ppf functions.
    hist_counts : List[int], 
        the histogram counts. Used for the entropy estimation with the
        correction
    """

    def __init__(
        self,
        bins: Union[int, str] = "auto",
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
        """Used to fit the empirical scipy dist to data.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            the data to be fitted
        y : np.ndarray, (n_samples)
            not used, for compatibility only
        Returns
        -------
        self : instance of self
        """
        X = check_array(X.reshape(-1, 1), ensure_2d=True, copy=True)

        # get support bounds
        support_bounds = get_domain_extension(X, self.support_extension)

        # calculate histogram
        hist = np.histogram(
            X.squeeze(), bins=self.bins, range=support_bounds, **self.kwargs
        )

        # needed for the entropy calculation
        self.hist_counts_ = hist[0]

        # fit model
        self.estimator_ = stats.rv_histogram(hist)

        # Add some noise to the pdfs to prevent zero probability
        self.estimator_._hpdf += self.alpha

        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """Probability density function for new data.
        Estimates the PDF for inputs.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            data to be estimated
        
        Returns
        -------
        Xpdf : np.ndarray, (n_samples)
            pdf of input data
        """
        X_prob = self.estimator_.pdf(X.squeeze())

        # X_prob = make_interior_probability(X_prob, eps=self.prob_tol)
        return X_prob

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        """Probability log density function for new data.
        Estimates the Log PDF for inputs.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            data to be estimated
        
        Returns
        -------
        Xlpdf : np.ndarray, (n_samples)
            log pdf of input data
        """
        return self.estimator_.logpdf(X.squeeze())

    def cdf(self, X: np.ndarray) -> np.ndarray:
        """Probability cumulative density function for new data.
        Estimates the CDF for inputs.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            data to be estimated
        
        Returns
        -------
        Xcdf : np.ndarray, (n_samples)
            cdf of input data
        """
        return self.estimator_.cdf(X.squeeze())

    def ppf(self, X: np.ndarray) -> np.ndarray:
        """Probability quantile function for new data.
        Estimates the PPF for inputs.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            data to be estimated
        
        Returns
        -------
        Xppf : np.ndarray, (n_samples)
            ppf of input data
        """
        return self.estimator_.ppf(X.squeeze())

    def entropy(self, correction: bool = True) -> float:
        """Entropy estimation for data
        Estimates the entropy given the fitted empirical
        distribution.
        
        Parameters
        ----------
        correction : bool, default=True
            does the Miller-Maddow correction
        
        Returns
        -------
        h_entropy : float
            entropy of the empirical distribution
        """
        # calculate entropy
        H = self.estimator_.entropy()

        # MLE Estimator with Miller-Maddow Correction
        # print(self.hist_counts_)
        if correction is True:
            H += 0.5 * (np.sum(self.hist_counts_ > 0) - 1) / self.hist_counts_.sum()

        return H


class QuantileHistogram(PDFEstimator):
    def __init__(
        self,
        bins: Union[int, str] = "auto",
        alpha: float = 1e-5,
        n_quantiles: int = 1_000,
        subsample: Optional[int] = 10_000,
        random_state: int = 123,
        support_extension: Union[float, int] = 10,
        kwargs: Dict = {},
    ) -> None:
        self.bins = bins
        self.alpha = alpha
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.random_state = random_state
        self.support_extension = support_extension
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        X = check_array(X.reshape(-1, 1), ensure_2d=False, copy=True)

        # get support bounds
        support_bounds = get_domain_extension(X, self.support_extension)

        # calculate histogram
        hist = np.histogram(
            X.squeeze(), bins=self.bins, range=support_bounds, **self.kwargs
        )
        self.hist_counts = hist[0]

        # needed for the entropy calculation
        self.hpdf_ = np.asarray(hist[0], dtype=np.float64)
        self.hpdf_ += self.alpha
        self.hbins_ = np.asarray(hist[1], dtype=np.float64)
        self.hbin_widths_ = self.hbins_[1:] - self.hbins_[:-1]
        self.hpdf_ = self.hpdf_ / float(np.sum(self.hpdf_ * self.hbin_widths_))
        self.hpdf_ = np.hstack([self.alpha, self.hpdf_])

        # Get CDF
        self.n_quantiles_ = max(1, min(self.n_quantiles, X.shape[0]))

        rng = check_random_state(self.random_state)

        self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)

        references = self.references_ * 100

        if self.subsample < X.shape[0]:
            subsample_idx = rng.choice(X.shape[0], size=self.subsample, replace=False)

            X = X.take(subsample_idx, axis=0, mode="clip")

        # X = np.hstack([support_bounds[0], X.squeeze(), support_bounds[1]])
        self.quantiles_ = np.nanpercentile(X, references)

        self.quantiles_ = np.maximum.accumulate(self.quantiles_)
        self.support_bounds_ = support_bounds

        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        return np.interp(X, self.hbins_, self.hpdf_)

    def logpdf(self, X: np.ndarray) -> np.ndarray:

        return np.log(self.pdf(X.squeeze()))

    def cdf(self, X: np.ndarray) -> np.ndarray:

        return np.interp(X.squeeze(), self.quantiles_, self.references_)

    def ppf(self, X: np.ndarray) -> np.ndarray:
        return np.interp(X.squeeze(), self.references_, self.quantiles_,)

    def entropy(self, correction: bool = True) -> float:
        raise NotImplementedError
