from scipy import stats
import numpy as np


def neg_entropy_normal(data: np.ndarray) -> np.ndarray:
    """Function to calculate the marginal negative entropy
    (negative entropy per dimensions). It uses a histogram
    scheme to initialize the bins and then uses a KDE
    scheme to approximate a smooth solution.
    Parameters
    ----------
    data : array, (n_samples, n_features)
        input data to be transformed
    Returns
    -------
    neg_ent : np.ndarray, (n_features)
        marginal neg entropy per features
    """

    n_samples, d_dimensions = data.shape

    # bin estimation
    # TODO: Use function
    n_bins = int(np.ceil(np.sqrt(n_samples)))

    neg = np.zeros(d_dimensions)

    # Loop through dimensions
    for idim in range(d_dimensions):

        # =====================
        # Histogram Estimation
        # =====================

        # Get Histogram
        [hist_counts, bin_edges] = np.histogram(
            a=data[:, idim],
            bins=n_bins,
            range=(data[:, idim].min(), data[:, idim].max()),
        )

        # calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # get delta between bin centers
        delta = bin_centers[3] - bin_centers[2]

        # Calculate probabilities of normal distribution
        pg = stats.norm.pdf(bin_centers, 0, 1)

        # ==================
        # KDE Function Est.
        # ==================

        # Initialize KDE function with data
        kde_model = stats.gaussian_kde(data[:, idim])

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
