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
    def __init__(
        self,
        bw_method: str = "scott",
        n_quantiles: int = 1_000,
        support_extension: Union[int, float] = 10,
    ) -> None:
        self.bw_method = bw_method
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        # fit model
        estimator = stats.gaussian_kde(X.squeeze(), bw_method=self.bw_method,)

        self.hbins_ = get_support_reference(
            support=X, extension=self.support_extension, n_quantiles=self.n_quantiles
        )
        self.hpdf_ = np.exp(estimator.logpdf(self.hbins_.squeeze()))

        self.hcdf_ = np.vectorize(lambda x: estimator.integrate_box_1d(-np.inf, x))(
            self.hbins_.squeeze()
        )

        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        return np.interp(X, self.hbins_, self.hpdf_)

    def logpdf(self, X: np.ndarray) -> np.ndarray:

        return np.log(self.pdf(X))

    def cdf(self, X: np.ndarray) -> np.ndarray:
        return np.interp(X, self.hbins_, self.hcdf_)

    def ppf(self, X: np.ndarray) -> np.ndarray:
        return np.interp(X, self.hcdf_, self.hbins_)

    def entropy(self, base: int = 2) -> float:

        # get log pdf
        raise NotImplementedError


class KDEFFT(KDEScipy, PDFEstimator):
    def __init__(
        self,
        bw_method: Union[float, str] = "normal_reference",
        n_quantiles: int = 1_000,
        support_extension: Union[int, float] = 10,
    ) -> None:
        self.bw_method = bw_method
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        # fit model
        self.hbins_ = get_support_reference(
            support=X, extension=self.support_extension, n_quantiles=self.n_quantiles
        )

        estimator = sm.nonparametric.KDEUnivariate(X.squeeze())

        estimator.fit(
            kernel="gau", bw=self.bw_method, fft=True, gridsize=self.n_quantiles,
        )

        self.hpdf_ = estimator.evaluate(self.hbins_.squeeze())

        self.hcdf_ = estimate_empirical_cdf(X, self.hbins_)
        # self.hcdf_ = kde_cdf(X, self.hbins_, bw=estimator.bw)

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

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

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
            support=X, extension=self.support_extension, n_quantiles=self.n_quantiles
        )
        self.hpdf_ = np.exp(estimator.score_samples(self.hbins_[:, None]))

        # get bin widths
        hbin_widths = self.hbins_[1:] - self.hbins_[:-1]

        # get cdf
        hcdf_ = np.cumsum(self.hpdf_[1:] * hbin_widths)

        self.hcdf_ = np.hstack([0.0, hcdf_])

        return self
