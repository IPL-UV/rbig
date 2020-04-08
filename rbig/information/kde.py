import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.utils import check_array
from sklearn.neighbors import KernelDensity
from rbig.information.base import PDFEstimator
from typing import Optional, Dict, Union
from rbig.utils import get_support_reference


def kde_entropy_uni(X: np.ndarray, **kwargs) -> float:

    # check input array
    X = check_array(X, ensure_2d=True)

    # initialize KDE
    kde_density = sm.nonparametric.KDEUnivariate(X)

    kde_density.fit(**kwargs)

    return kde_density.entropy


class KDESklearn(PDFEstimator):
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

        X = check_array(X, ensure_2d=True, copy=True)

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
        estimator = stats.gaussian_kde(X, bw_method=self.bw_method,)

        # get reference support
        self.hbins_ = get_support_reference(
            support=X, extension=self.support_extension, n_quantiles=self.n_quantiles
        )
        self.hpdf_ = estimator.pdf(self.hbins_)

        # get bin widths
        hbin_widths = self.hbins_[1:] - self.hbins_[:-1]

        # get cdf
        hcdf_ = np.cumsum(self.hpdf_[1:] * hbin_widths)

        self.hcdf_ = np.hstack([0.0, hcdf_])

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
