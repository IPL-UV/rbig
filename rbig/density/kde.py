from typing import Union, Optional, Dict, Tuple
import numpy as np
from scipy import stats, integrate
from sklearn.utils import check_array
from rbig.utils import get_support_reference, get_domain_extension
import logging

import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.utils import check_array
from sklearn.neighbors import KernelDensity
from rbig.density.base import PDFEstimator
from typing import Optional, Dict, Union
from rbig.utils import get_support_reference, get_domain_extension
from scipy.interpolate import interp1d
from rbig.density.utils import kde_cdf
from rbig.density.empirical import estimate_empirical_cdf
from KDEpy import FFTKDE

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def exact_kde_est_pdf(
    X: np.ndarray,
    nbins: int = 100,
    bw_method: str = "scott",
    fft: bool = False,
    support_extension: Union[float, int] = 10,
    support_bounds: Optional[Tuple[float, float]] = None,
    kwargs: Dict = {},
) -> np.ndarray:
    X = check_array(X, ensure_2d=False, copy=True)

    # get grid
    if support_bounds is None:
        lb, ub = get_domain_extension(X, support_extension)
    else:
        lb, ub = support_bounds

    hbins = np.concatenate([lb, X, ub])

    # fit KDE Model
    if fft is False:
        estimator = stats.gaussian_kde(X, bw_method=bw_method,)
        estimator.fit(kernel="gau", bw_method=bw_method, fft=True, gridsize=nbins)

        hpdf = estimator.pdf(hbins)
    else:
        estimator = sm.nonparametric.KDEUnivariate(X.squeeze())

        hpdf = estimator.pdf(hbins)
    return hpdf, hbins


class KDEScipy(PDFEstimator):
    """KDE pdf estimator using the exact scipy implementation 
    This implementation using the Gaussian kernel and we compute
    the PDF estimation exactly using integration. To get PDF estimations
    for the future, we can either compute it exactly or use interpolation
    (much faster). The empirical function is found using the ECDF estimator.
    
    Parameters
    ----------
    bw_method : str, default='scott'
        bandwidth for the kernel estimation. almost all methods have
        scott so that's the default. 
    n_quantiles : Optional[int], default=None
        the number of quantiles (support) to use for the PDF/CDF/PPF functions.
    support_extension : int, default=10
        the amount to extend the support of the fitted data X. Affects
        the PDF,CDF, and PPF functions. Extending the support will allow
        more data to be interpolated.
    interp : bool, default=True
        decides whether to interpolate the pdf estimation or compute it
        exactly
    
    Attributes
    ----------
    hbins_ : np.ndarray, (n_quantiles)
        the support for the PDF, CDF and PPF
    hpdf_ : np.ndarray, (n_quantiles)
        the pdf estimates for the PDF function
    hcdf_ : np.ndarray, (n_quantiles)
        the quantiles for the CDF/PPF function
    """

    def __init__(
        self,
        bw_method: str = "scott",
        n_quantiles: int = 1_000,
        support_extension: Union[int, float] = 10,
        interp: bool = True,
    ) -> None:
        self.bw_method = bw_method
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension
        self.interp = interp

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fits the PDF estimation to the data
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            the data to be fitted
        y : np.ndarray, 
            not used. For compatibility only
        """
        # fit model
        estimator = stats.gaussian_kde(X.squeeze(), bw_method=self.bw_method,)

        self.hbins_ = get_support_reference(
            support=X.squeeze(),
            extension=self.support_extension,
            n_quantiles=self.n_quantiles,
        )

        self.hcdf_ = np.vectorize(lambda x: estimator.integrate_box_1d(-np.inf, x))(
            self.hbins_.squeeze()
        )
        if self.interp:
            self.hpdf_ = estimator.evaluate(self.hbins_.squeeze())
        else:
            self.estimator_ = estimator

        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """Estimates the PDF from new data
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            input data
        
        Returns
        -------
        X_pdf : np.ndarray, (n_samples)
            pdf of the inputs X
        """
        if self.interp:
            return np.interp(X.squeeze(), self.hbins_, self.hpdf_)
        else:
            return self.estimator_.evaluate(X.squeeze())

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        """Estimates the log PDF from new data
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            input data
        
        Returns
        -------
        X_lpdf : np.ndarray, (n_samples)
            log pdf of the inputs X
        """
        return np.log(self.pdf(X))

    def cdf(self, X: np.ndarray) -> np.ndarray:
        """Estimates the CDF from new data
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            input data
        
        Returns
        -------
        X_cdf : np.ndarray, (n_samples)
            cdf of the inputs X
        """
        return np.interp(X, self.hbins_, self.hcdf_)

    def ppf(self, X: np.ndarray) -> np.ndarray:
        """Estimates the PPF from new data
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            input data
        
        Returns
        -------
        X_ppf : np.ndarray, (n_samples)
            pdf of the inputs X
        """
        return np.interp(X, self.hcdf_, self.hbins_)

    def entropy(self, base: int = 2) -> float:

        # get log pdf
        raise NotImplementedError


class KDEFFT(KDEScipy, PDFEstimator):
    """KDE pdf estimator using the FFT from the statsmodels package
    This implementation using the FFT for the Gaussian kernel from the
    statsmodels package. The PDF estimation exactly using integration.
    To get PDF estimations for the future, we can either compute it exactly
    or use interpolation (much faster). The empirical function is found
    using the ECDF estimator.
    
    Parameters
    ----------
    bw_method : str, default='scott'
        bandwidth for the kernel estimation. almost all methods have
        scott so that's the default. 
    n_quantiles : Optional[int], default=None
        the number of quantiles (support) to use for the PDF/CDF/PPF functions.
    support_extension : int, default=10
        the amount to extend the support of the fitted data X. Affects
        the PDF,CDF, and PPF functions. Extending the support will allow
        more data to be interpolated.
    interp : bool, default=True
        decides whether to interpolate the pdf estimation or compute it
        exactly
    
    Attributes
    ----------
    hbins_ : np.ndarray, (n_quantiles)
        the support for the PDF, CDF and PPF
    hpdf_ : np.ndarray, (n_quantiles)
        the pdf estimates for the PDF function
    hcdf_ : np.ndarray, (n_quantiles)
        the quantiles for the CDF/PPF function
    """

    def __init__(
        self,
        bw_method: Union[float, str] = "normal_reference",
        n_quantiles: int = 1_000,
        support_extension: Union[int, float] = 10,
        interp: bool = True,
    ) -> None:
        self.bw_method = bw_method
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension
        self.interp = interp

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fits the PDF estimation to the data
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            the data to be fitted
        y : np.ndarray,
            not used. For compatibility only
        """
        # fit model
        self.hbins_ = get_support_reference(
            support=X.squeeze(),
            extension=self.support_extension,
            n_quantiles=self.n_quantiles,
        )

        estimator = sm.nonparametric.KDEUnivariate(X.squeeze())

        estimator.fit(
            kernel="gau", bw=self.bw_method, fft=True, gridsize=self.n_quantiles,
        )

        # evaluate cdf from KDE estimator
        if self.interp:
            self.hpdf_ = estimator.evaluate(self.hbins_.squeeze())
        else:
            self.estimator_ = estimator
        #

        # estimate the empirical CDF function from data
        self.hcdf_ = estimate_empirical_cdf(X.squeeze(), self.hbins_)

        # self.hcdf_ = kde_cdf(X, self.hbins_, bw=estimator.bw)

        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """Estimates the PDF from new data
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            input data
        
        Returns
        -------
        X_pdf : np.ndarray, (n_samples)
            pdf of the inputs X
        """
        if self.interp:
            return np.interp(X.squeeze(), self.hbins_, self.hpdf_)
        else:
            return self.estimator_.evaluate(X.squeeze())


class KDEEpanechnikov(KDEScipy, PDFEstimator):

    """
    Parameters
    ----------
    kernel : str, default='epa'
        Many kernels available. For this toolbox, I recommend the
        epa kernel for fast PDF evaluation.

    bw_method : str, default='ISJ',
    """

    def __init__(
        self,
        kernel: "epa",
        bw_method: Union[float, str] = "ISJ",
        n_quantiles: int = 1_000,
        support_extension: Union[int, float] = 10,
        norm: int = 2,
    ) -> None:
        self.kernel = kernel
        self.bw_method = bw_method
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension
        self.norm = norm
        self.interp = True

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fits the PDF estimation to the data
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            the data to be fitted
        y : np.ndarray, 
            not used. For compatibility only
        """
        # estimate the empirical CDF function from data
        self.hbins_ = get_support_reference(
            support=X.squeeze(),
            extension=self.support_extension,
            n_quantiles=self.n_quantiles,
        )

        estimator = FFTKDE(kernel=self.kernel, bw=self.bw_method, norm=self.norm)

        estimator.fit(X.squeeze())

        self.hpdf_ = estimator.evaluate(self.hbins_.squeeze())

        self.hcdf_ = estimate_empirical_cdf(X.squeeze(), self.hbins_)

        return self


class KDESklearn(KDEScipy, PDFEstimator):
    def __init__(
        self,
        algorithm: str = "kd_tree",
        kernel: str = "gaussian",
        metric: str = "euclidean",
        n_quantiles: int = 1_000,
        support_extension: Union[int, float] = 10,
        kwargs: Optional[Dict] = {},
    ) -> None:
        self.algorithm = algorithm
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension
        self.kernel = kernel
        self.metric = metric
        self.kwargs = kwargs
        self.interp = True

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fits the PDF estimation to the data
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            the data to be fitted
        y : np.ndarray, 
            not used. For compatibility only
        """
        X = check_array(X.reshape(-1, 1), ensure_2d=True, copy=True)

        # fit model
        estimator = KernelDensity(
            algorithm=self.algorithm,
            metric=self.metric,
            kernel=self.kernel,
            **self.kwargs,
        ).fit(X)

        # get reference support
        self.hbins_ = get_support_reference(
            support=X, extension=self.support_extension, n_quantiles=self.n_quantiles,
        )
        self.hpdf_ = np.exp(estimator.score_samples(self.hbins_[:, None]))

        # get bin widths
        self.hcdf_ = estimate_empirical_cdf(X.squeeze(), self.hbins_)

        return self
