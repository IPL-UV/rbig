import functools
from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

from rbig.transform.kde import kde_fft_uniformization
from rbig.utils import bin_estimation, get_domain_extension

BOUNDS_THRESHOLD = 1e-7


def uniform_transform_params(
    X: np.ndarray, method: str = "custom", params: Dict = {}
) -> Dict[str, np.ndarray]:

    if method == "custom":
        return custom_uniform(
            X,
            support_extension=params.get("support_extension", 10),
            precision=params.get("precision", 1_000),
        )
    elif method == "histogram":
        return histogram_uniformization(
            X,
            bins=params.get("bins", "auto"),
            alpha=params.get("alpha", 1e-5),
            support_extension=params.get("support_extension", 10),
            kwargs=params.get("kwargs", {}),
        )
    elif method == "kdefft":
        return kde_fft_uniformization(
            X,
            bw_method=params.get("bw_method", "scott"),
            n_quantiles=params.get("n_quantiles", 50),
            support_extension=params.get("support_extension", 10),
            kwargs=params.get("kwargs", {}),
        )
    else:
        raise ValueError("Unrecognized uniform transformation")


def uniform_inverse_transform(X: np.ndarray, params: Dict[str, Callable]) -> np.ndarray:
    """Inverts the marginal uniformization
    params are used from the uniform transform function.
    Parameters
    ----------
    X : np.ndarray, (n_samples)
        inputs to be transformed
    params : Dict[str, Callable]
        params to be used to do the transformation. Must have a parameter
        'uniform_cdf' to do the inverse transformation
    Returns
    -------
    Xtrans : np.ndarray, (n_samples)
        data in the original domain.
    """
    return params["uniform_ppf"](X)


def histogram_uniformization(
    X: np.ndarray,
    bins: Union[str, float] = "auto",
    alpha: float = 1e-5,
    support_extension: int = 10,
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
    # get extended support bounds
    support_bounds = get_domain_extension(X, support_extension)
    # create histogram
    hist = np.histogram(X, bins=bins, range=support_bounds, **kwargs)

    # create histogram object
    estimator = stats.rv_histogram(hist)

    # add some noise to
    estimator._hpdf += alpha

    # return transformation and functions
    def logpdf(x):
        return np.log(estimator.pdf(x))

    return {
        "x_bounds": [estimator._hbins.min(), estimator._hbins.max()],
        "uniform_logpdf": logpdf,
        "uniform_cdf": estimator.cdf,
        "uniform_ppf": estimator.ppf,
        # raw values
        "support_pdf": np.concatenate([np.array([0.0]), estimator._hbins]),
        "pdf": estimator._hpdf,
        "support_cdf": estimator._hbins,
        "quantiles": estimator._hcdf,
    }


def custom_uniform(
    X: np.ndarray, support_extension: float = 0.1, precision: int = 1_000
) -> Dict[str, np.ndarray]:
    """
    Takes univariate data and transforms it to have approximately uniform dist
    Parameters
    ----------
    X : ndarray
        The univariate data [1xS] where S is the number of samples in the dataset
    support_extension : int, default=10
        Extend the marginal PDF support by % of this amount as a fraction of the
        total x domain.
    precision : int
        The number of points in the marginal PDF
    Returns
    -------
    uni_uniform_data : ndarray
    univariate uniform data
    transform_params : dictionary
    parameters of the transform. We save these so we can invert them later
    """
    n_samples = len(X)
    support_extension = (support_extension / 100) * abs(np.max(X) - np.min(X))

    # not sure exactly what we're doing here, but at a high level we're
    nbins = bin_estimation(X, rule="sqrt")
    # constructing bins for the histogram
    bin_edges = np.linspace(np.min(X), np.max(X), int(nbins + 1),)
    bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)

    counts, _ = np.histogram(X, bin_edges)
    # counts = np.asarray(counts, dtype=np.float64)
    # counts += 1e-7

    bin_size = bin_edges[2] - bin_edges[1]
    pdf_support = np.hstack(
        (bin_centers[0] - bin_size, bin_centers, bin_centers[-1] + bin_size)
    )
    empirical_pdf = np.hstack((0.0, counts / (np.sum(counts) * bin_size), 0.0))
    # ^ this is unnormalized
    c_sum = np.cumsum(counts)
    cdf = (1 - 1 / n_samples) * c_sum / n_samples

    incr_bin = bin_size / 2

    new_bin_edges = np.hstack(
        (
            np.min(X) - support_extension,
            np.min(X),
            bin_centers + incr_bin,
            np.max(X) + support_extension + incr_bin,
        )
    )

    extended_cdf = np.hstack((0.0, 1.0 / n_samples, cdf, 1.0))
    new_support = np.linspace(new_bin_edges[0], new_bin_edges[-1], int(precision))
    learned_cdf = interp1d(new_bin_edges, extended_cdf)
    uniform_cdf = np.maximum.accumulate(learned_cdf(new_support))
    # ^ linear interpolation
    uniform_cdf /= np.max(uniform_cdf)
    uniform_cdf_func = interp1d(new_support, uniform_cdf, fill_value="extrapolate")
    # uniform_cdf_func = functools.partial(np.interp, xp=new_support, fp=uniform_cdf)
    uniform_ppf_func = interp1d(uniform_cdf, new_support, fill_value="extrapolate")
    # uniform_ppf_func = functools.partial(np.interp, xp=uniform_cdf, fp=new_support)
    uniform_pdf_func = interp1d(pdf_support, empirical_pdf, fill_value="extrapolate")

    def uniform_logpdf_func(x):
        return np.log(uniform_pdf_func(x))

    return {
        "x_bounds": [pdf_support.min(), pdf_support.max()],
        "uniform_logpdf": uniform_logpdf_func,
        "support": new_support,
        "quantiles": uniform_cdf,
        "uniform_cdf": uniform_cdf_func,
        "uniform_ppf": uniform_ppf_func,
    }
