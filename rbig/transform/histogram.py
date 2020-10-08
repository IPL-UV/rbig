from collections import namedtuple
from typing import Tuple, Dict
from rbig.density import make_cdf_monotonic
import numpy as np
from scipy.interpolate import interp1d
from rbig.transform.univariate import UniParams


def univariate_make_uniform(
    data: np.ndarray, extension: float = 0.1, precision: int = 1_000
) -> Tuple[np.ndarray, Dict]:
    """
    Takes univariate data and transforms it to have approximately uniform dist

    Parameters
    ----------
    data : np.ndarray
        The data to be estimated by the PDF and CDF (n_samples, )
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
    n_samps = data.shape[0]
    support_extension = (extension / 100) * abs(np.max(data) - np.min(data))

    # not sure exactly what we're doing here, but at a high level we're
    # constructing bins for the histogram
    bin_edges = np.linspace(np.min(data), np.max(data), int(np.sqrt(n_samps)) + 1)
    bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)

    counts, _ = np.histogram(data, bin_edges)

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
            np.min(data) - support_extension,
            np.min(data),
            bin_centers + incr_bin,
            np.max(data) + support_extension + incr_bin,
        )
    )

    extended_cdf = np.hstack((0.0, 1.0 / n_samps, cdf, 1.0))
    new_support = np.linspace(new_bin_edges[0], new_bin_edges[-1], int(precision))
    learned_cdf = interp1d(new_bin_edges, extended_cdf)
    uniform_cdf = make_cdf_monotonic(learned_cdf(new_support))
    # ^ linear interpolation
    uniform_cdf /= np.max(uniform_cdf)
    uni_uniform_data = interp1d(new_support, uniform_cdf)(data)

    return (
        uni_uniform_data,
        {
            "empirical_pdf_support": pdf_support,
            "empirical_pdf": empirical_pdf,
            "uniform_cdf_support": new_support,
            "uniform_cdf": uniform_cdf,
        },
    )

