from typing import Union
import numpy as np
from scipy.stats import norm, gaussian_kde


def negative_log_likelihood(X: np.ndarray, X_ldj: np.ndarray) -> np.ndarray:
    pz = norm.logpdf(X).sum(axis=-1)
    log_prob = pz + X_ldj
    return -np.mean(log_prob)


def neg_entropy_normal(data, bins: Union[str, int] = "auto") -> np.ndarray:
    """Function to calculate the marginal negative entropy
    (negative entropy per dimensions). It uses a histogram
    scheme to initialize the bins and then uses a KDE
    scheme to approximate a smooth solution.

    Parameters
    ----------
    data : array, (samples x dimensions)

    Returns
    -------
    neg : array, (dimensions)

    """

    n_samples, d_dimensions = data.shape

    neg = np.zeros(d_dimensions)

    # Loop through dimensions
    for idim in range(d_dimensions):

        # =====================
        # Histogram Estimation
        # =====================

        # Get Histogram
        [hist_counts, bin_edges] = np.histogram(
            a=data[:, idim],
            bins=bins,
            range=(data[:, idim].min(), data[:, idim].max()),
        )

        # calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # get delta between bin centers
        delta = bin_centers[3] - bin_centers[2]

        # Calculate probabilities of normal distribution
        pg = norm.pdf(bin_centers, 0, 1)

        # ==================
        # KDE Function Est.
        # ==================

        # Initialize KDE function with data
        kde_model = gaussian_kde(data[:, idim])

        # Calculate probabilities for each bin
        hx = kde_model.pdf(bin_centers)

        # Calculate probabilities
        px = hx / (hx.sum() * delta)

        # ====================
        # Compare
        # ====================

        # Find the indices greater than zero
        idx = np.where((px > 0) & (pg > 0))

        # calculate the negative entropy
        neg[idim] = delta * (px[idx] * np.log2(px[idx] / pg[idx])).sum()

    return neg
