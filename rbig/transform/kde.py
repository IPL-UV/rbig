import functools
from typing import Dict, Tuple, Union

import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d

from rbig.utils import (estimate_empirical_cdf, get_domain_extension,
                        get_support_reference)


def kde_fft_uniformization(
    X: np.ndarray,
    bw_method: str = "scott",
    n_quantiles: int = 50,
    support_extension: Union[int, float] = 10,
    interp: bool = True,
    kwargs: Dict = {},
) -> Dict[str, np.ndarray]:
    """Univariate histogram density estimator.
    A light wrapper around the scipy `stats.rv_histogram` function
    which calculates the empirical distribution. After this has
    been fitted, this method will have the standard density
    functions like pdf, logpdf, cdf, ppf. All of these are
    necessary for the transformations.
    Parameters
    ----------
    X : np.ndarray, (n_samples)
        the data to be transformed with returned parameters
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
    Returns
    -------
    X : np.ndarray, (n_samples)
        the transformed data that has been uniformized
    params : Dict
        a dictionary containing the functions to estimate future params

        * uniform_logpdf - function to estimate the log pdf
        * uniform_cdf - function to estimate the cdf
        * uniform_ppf - function to estimate the ppf
    """
    hbins_ = get_support_reference(
        support=X.squeeze(), extension=support_extension, n_quantiles=n_quantiles,
    )

    estimator = sm.nonparametric.KDEUnivariate(X.squeeze())
    bw = scotts_factor(X.reshape(-1, 1))
    estimator.fit(
        kernel="gau", bw=bw, fft=True, gridsize=n_quantiles,
    )

    # evaluate cdf from KDE estimator
    # if interp:
    hpdf_ = estimator.evaluate(hbins_.squeeze())
    # else:
    #     estimator_ = estimator
    #

    # estimate the empirical CDF function from data
    hcdf_ = estimate_empirical_cdf(X.squeeze(), hbins_)

    pdf_estimator = functools.partial(np.interp, xp=hbins_, fp=hpdf_)
    cdf_estimator = functools.partial(np.interp, xp=hbins_, fp=hcdf_)
    ppf_estimator = functools.partial(np.interp, xp=hcdf_, fp=hbins_)
    # return transformation and functions
    def logpdf(x):
        return np.log(pdf_estimator(x))

    return {
        "x_bounds": [hbins_.min(), hbins_.max()],
        "uniform_logpdf": logpdf,
        "uniform_cdf": cdf_estimator,
        "uniform_ppf": ppf_estimator,
    }


def scotts_factor(X: np.ndarray) -> float:
    """Scotts Method to estimate the length scale of the 
    rbf kernel.

        factor = n**(-1./(d+4))

    Parameters
    ----------
    X : np.ndarry
        Input array

    Returns
    -------
    factor : float
        the length scale estimated

    """
    n_samples, n_features = X.shape

    return np.power(n_samples, -1 / (n_features + 4.0))
