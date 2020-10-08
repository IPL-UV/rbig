from typing import Optional, Tuple, Dict, Union

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from rbig.utils import get_domain_extension

# from rbig.utils import make_cdf_monotonic


def univariate_make_normal(
    uni_data: np.ndarray, extension: float, precision: int
) -> Tuple[np.ndarray, Dict]:
    """
    Takes univariate data and transforms it to have approximately normal dist
    We do this through the simple composition of a histogram equalization
    producing an approximately uniform distribution and then the inverse of the
    normal CDF. This will produce approximately gaussian samples.
    Parameters
    ----------
    uni_data : ndarray
    The univariate data [Sx1] where S is the number of samples in the dataset
    extension : float
    Extend the marginal PDF support by this amount.
    precision : int
    The number of points in the marginal PDF

    Returns
    -------
    uni_gaussian_data : ndarray
    univariate gaussian data
    params : dictionary
    parameters of the transform. We save these so we can invert them later
    """
    data_uniform, params = univariate_make_uniform(uni_data.T, extension, precision)
    return stats.norm.ppf(data_uniform).T, params


def univariate_make_uniform(
    uni_data: np.ndarray, extension: float, precision: int
) -> Tuple[np.ndarray, Dict]:
    """
    Takes univariate data and transforms it to have approximately uniform dist
    Parameters
    ----------
    uni_data : ndarray
    The univariate data [1xS] where S is the number of samples in the dataset
    extension : float
    Extend the marginal PDF support by this amount. Default 0.1
    precision : int
    The number of points in the marginal PDF
    Returns
    -------
    uni_uniform_data : ndarray
    univariate uniform data
    transform_params : dictionary
    parameters of the transform. We save these so we can invert them later
    """
    n_samps = uni_data.shape[0]
    support_extension = (extension / 100) * abs(np.max(uni_data) - np.min(uni_data))

    # not sure exactly what we're doing here, but at a high level we're
    # constructing bins for the histogram
    bin_edges = np.linspace(
        np.min(uni_data), np.max(uni_data), int(np.sqrt(n_samps)) + 1
    )
    bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)

    counts, _ = np.histogram(uni_data, bin_edges)

    bin_size = bin_edges[2] - bin_edges[1]
    pdf_support = np.hstack(
        (bin_centers[0] - bin_size, bin_centers, bin_centers[-1] + bin_size)
    )
    empirical_pdf = np.hstack((0.0, counts / (np.sum(counts) * bin_size), 0.0))
    # ^ this is unnormalized
    c_sum = np.cumsum(counts)
    cdf = (1 - 1 / n_samps) * c_sum / n_samps

    incr_bin = bin_size / 2

    new_bin_edges = np.hstack(
        (
            np.min(uni_data) - support_extension,
            np.min(uni_data),
            bin_centers + incr_bin,
            np.max(uni_data) + support_extension + incr_bin,
        )
    )

    extended_cdf = np.hstack((0.0, 1.0 / n_samps, cdf, 1.0))
    new_support = np.linspace(new_bin_edges[0], new_bin_edges[-1], int(precision))
    learned_cdf = interp1d(new_bin_edges, extended_cdf)
    uniform_cdf = make_cdf_monotonic(learned_cdf(new_support))
    # ^ linear interpolation
    uniform_cdf /= np.max(uniform_cdf)
    uni_uniform_data = interp1d(new_support, uniform_cdf)(uni_data)

    return (
        uni_uniform_data,
        {
            "empirical_pdf_support": pdf_support,
            "empirical_pdf": empirical_pdf,
            "uniform_cdf_support": new_support,
            "uniform_cdf": uniform_cdf,
        },
    )


def univariate_invert_normalization(
    uni_gaussian_data: np.ndarray, trans_params: Dict
) -> np.ndarray:
    """
    Inverts the marginal normalization
    See the companion, univariate_make_normal.py, for more details
    """
    uni_uniform_data = stats.norm.cdf(uni_gaussian_data)

    uni_data = univariate_invert_uniformization(uni_uniform_data, trans_params)

    return uni_data


def univariate_invert_uniformization(
    uni_uniform_data: np.ndarray, trans_params: Dict
) -> np.ndarray:
    """
    Inverts the marginal uniformization transform specified by trans_params
    See the companion, univariate_make_normal.py, for more details
    """
    # simple, we just interpolate based on the saved CDF
    return interp1d(trans_params["uniform_cdf"], trans_params["uniform_cdf_support"])(
        uni_uniform_data
    )


def estimate_empirical_cdf(
    X: np.ndarray, X_new: Optional[np.ndarray] = None
) -> np.ndarray:

    # initialize ecdf
    ecdf_f = ECDF(X)
    if X_new is None:
        return ecdf_f(X)
    else:
        return ecdf_f(X_new)


def bin_estimation(n_samples: int, rule="standard") -> float:
    """Bin estimation for the histogram"""

    if rule == "sturge":
        n_bins = int(np.ceil(1 + 3.322 * np.log10(n_samples)))

    elif rule == "standard":
        n_bins = int(np.ceil(np.sqrt(n_samples)))

    else:
        raise ValueError(f"Unrecognized bin estimation rule: {rule}")

    return n_bins


def make_cdf_monotonic(cdf):
    """
    Take a cdf and just sequentially readjust values to force monotonicity
    There's probably a better way to do this but this was in the original
    implementation. We just readjust values that are less than their predecessors
    Parameters
    ----------
    cdf : ndarray
      The values of the cdf in order (1d)
    """
    # laparra's version
    corrected_cdf = cdf.copy()
    for i in range(1, len(corrected_cdf)):
        if corrected_cdf[i] <= corrected_cdf[i - 1]:
            if abs(corrected_cdf[i - 1]) > 1e-14:
                corrected_cdf[i] = corrected_cdf[i - 1] + 1e-14
            elif corrected_cdf[i - 1] == 0:
                corrected_cdf[i] = 1e-80
            else:
                corrected_cdf[i] = corrected_cdf[i - 1] + 10 ** (
                    np.log10(abs(corrected_cdf[i - 1]))
                )
    return corrected_cdf
