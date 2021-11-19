from typing import Union, NamedTuple
from scipy.stats import rv_histogram
import numpy as np
import statsmodels.api as sm
from functools import partial
from statsmodels.distributions.empirical_distribution import ECDF
from astropy.stats import histogram as astro_hist
from scipy.special import ndtr


class MarginalHistogramUniformization:
    name: str = "marghistuni"

    def __init__(
        self,
        X: np.ndarray,
        bins: Union[int, str] = "auto",
        alpha: float = 1e-10,
        bound_ext: float = 0.1,
    ):

        estimators = []

        for iX in X.T:
            diff = iX.max() - iX.min()
            lower_bound = iX.min() - bound_ext * diff
            upper_bound = iX.max() + bound_ext * diff

            # create histogram
            if bins in ["blocks", "knuth"]:
                hist = astro_hist(iX, bins=bins, range=(lower_bound, upper_bound))
            else:
                hist = np.histogram(iX, bins=bins, range=(lower_bound, upper_bound))

            # create histogram object
            i_estimator = rv_histogram(hist)

            # add some regularization
            i_estimator._hpdf += alpha

            estimators.append(i_estimator)

        self.estimators = estimators

    def forward(self, X):

        Z = np.zeros_like(X)
        for idim, iX in enumerate(X.T):
            Z[:, idim] = self.estimators[idim].cdf(iX)

        return Z

    def inverse(self, Z):

        X = np.zeros_like(Z)

        for idim, iZ in enumerate(Z.T):

            X[:, idim] = self.estimators[idim].ppf(iZ)

        return X

    def gradient(self, X):

        X_grad = np.zeros_like(X)

        for idim, iX in enumerate(X.T):
            X_grad[:, idim] = self.estimators[idim].logpdf(iX)
        X_grad = X_grad.sum(axis=-1)
        return X_grad


class KDEParams(NamedTuple):
    support: np.ndarray
    pdf_est: np.ndarray
    cdf_est: np.ndarray


class MarginalKDEUniformization:
    name: str = "kdefft"

    def __init__(
        self,
        X: np.ndarray,
        grid_size: int = 50,
        n_quantiles: int = 1_000,
        bound_ext: float = 0.1,
        fft: bool = True,
    ):

        estimators = []

        # estimate bandwidth
        bw = np.power(X.shape[0], -1 / (X.shape[1] + 4.0))

        for iX in X.T:

            # create histogram
            estimator = sm.nonparametric.KDEUnivariate(iX.squeeze())

            estimator.fit(
                kernel="gau",
                bw=bw,
                fft=fft,
                gridsize=grid_size,
            )

            # estimate support
            diff = iX.max() - iX.min()
            lower_bound = iX.min() - bound_ext * diff
            upper_bound = iX.max() + bound_ext * diff
            support = np.linspace(lower_bound, upper_bound, n_quantiles)

            # estimate empirical pdf from data
            hpdf = estimator.evaluate(support)

            # estimate empirical cdf from data
            hcdf = ECDF(iX)(support)

            kde_params = KDEParams(support=support, pdf_est=np.log(hpdf), cdf_est=hcdf)
            estimators.append(kde_params)

        self.estimators = estimators

    def forward(self, X):

        Z = np.zeros_like(X)
        for idim, iX in enumerate(X.T):
            iparams = self.estimators[idim]
            Z[:, idim] = np.interp(iX, xp=iparams.support, fp=iparams.cdf_est)

        return Z

    def inverse(self, Z):

        X = np.zeros_like(Z)

        for idim, iZ in enumerate(Z.T):

            iparams = self.estimators[idim]
            X[:, idim] = np.interp(iZ, xp=iparams.cdf_est, fp=iparams.support)

        return X

    def gradient(self, X):

        X_grad = np.zeros_like(X)

        for idim, iX in enumerate(X.T):

            iparams = self.estimators[idim]
            X_grad[:, idim] = np.interp(iX, xp=iparams.support, fp=iparams.pdf_est)

        X_grad = X_grad.sum(axis=-1)
        return X_grad


# class MarginalKDEUniformization:
#     name: str = "kdefft"

#     def __init__(
#         self,
#         X: np.ndarray,
#         grid_size: int = 50,
#         n_quantiles: int = 50,
#         bound_ext: float = 0.1,
#         fft: bool = True,
#     ):

#         estimators = []

#         # estimate bandwidth
#         bw = np.power(X.shape[0], -1 / (X.shape[1] + 4.0))

#         for iX in X.T:

#             # estimate support
#             diff = iX.max() - iX.min()
#             lower_bound = iX.min() - bound_ext * diff
#             upper_bound = iX.max() + bound_ext * diff
#             support = np.linspace(lower_bound, upper_bound, n_quantiles)

#             bw = scotts_method(X.shape[0], 1) * 0.5

#             # calculate the pdf for gaussian pdf
#             pdf_support = broadcast_kde_pdf(support, iX, bw)

#             # calculate the cdf for support points
#             factor = normalization_factor(iX, bw)

#             quantiles = broadcast_kde_cdf(support, iX, factor)

#             kde_params = KDEParams(
#                 support=support, pdf_est=np.log(pdf_support), cdf_est=quantiles
#             )
#             estimators.append(kde_params)

#         self.estimators = estimators

#     def forward(self, X):

#         Z = np.zeros_like(X)
#         for idim, iX in enumerate(X.T):
#             iparams = self.estimators[idim]
#             Z[:, idim] = np.interp(iX, xp=iparams.support, fp=iparams.cdf_est)

#         return Z

#     def inverse(self, Z):

#         X = np.zeros_like(Z)

#         for idim, iZ in enumerate(Z.T):

#             iparams = self.estimators[idim]
#             X[:, idim] = np.interp(iZ, xp=iparams.cdf_est, fp=iparams.support)

#         return X

#     def gradient(self, X):

#         X_grad = np.zeros_like(X)

#         for idim, iX in enumerate(X.T):

#             iparams = self.estimators[idim]
#             X_grad[:, idim] = np.interp(iX, xp=iparams.support, fp=iparams.pdf_est)

#         X_grad = X_grad.sum(axis=-1)
#         return X_grad


def broadcast_kde_pdf(eval_points, samples, bandwidth):

    n_samples = samples.shape[0]

    # distances (use broadcasting)
    # print(eval_points.shape, samples.shape)
    rescaled_x = (
        eval_points[:, np.newaxis] - samples[np.newaxis, :]
    ) / bandwidth  # (2 * bandwidth ** 2)

    # compute the gaussian kernel
    gaussian_kernel = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * rescaled_x ** 2)

    # rescale
    K = np.sum(gaussian_kernel, axis=1) / n_samples / bandwidth

    return K


def broadcast_kde_cdf(x_evals, samples, factor):
    return ndtr((x_evals[:, np.newaxis] - samples[np.newaxis, :]) / factor).mean(axis=1)


def normalization_factor(data, bw):

    data_covariance = np.cov(data[:, np.newaxis], rowvar=0, bias=False)

    covariance = data_covariance * bw ** 2

    stdev = np.sqrt(covariance)

    return stdev


def scotts_method(
    n_samples: int,
    n_features: int,
):
    return np.power(n_samples, -1.0 / (n_features + 4))
