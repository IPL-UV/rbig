import numpy as np
from scipy.stats import entropy as sci_entropy


def entropy(hist_counts, correction=None):

    # MLE Estimator with Miller-Maddow Correction
    if not (correction is None):
        correction = 0.5 * (np.sum(hist_counts > 0) - 1) / hist_counts.sum()
    else:
        correction = 0.0

    # Plut in estimator of entropy with correction
    return sci_entropy(hist_counts, base=2) + correction


def entropy_marginal(data, bin_est="standard", correction=True):
    """Calculates the marginal entropy (the entropy per dimension) of a
    multidimensional dataset. Uses histogram bin counnts. Also features
    and option to add the Shannon-Miller correction.
    
    Parameters
    ----------
    data : array, (n_samples x d_dimensions)
    
    bin_est : str, (default='standard')
        The bin estimation method.
        {'standard', 'sturge'}
    
    correction : bool, default=True
    
    Returns
    -------
    H : array (d_dimensions)
    
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    """
    n_samples, d_dimensions = data.shape

    n_bins = bin_estimation(n_samples, rule=bin_est)

    H = np.zeros(d_dimensions)

    for idim in range(d_dimensions):
        # Get histogram (use default bin estimation)
        [hist_counts, bin_edges] = np.histogram(
            a=data[:, idim],
            bins=n_bins,
            range=(data[:, idim].min(), data[:, idim].max()),
        )

        # Calculate bin_centers from bin_edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # get difference between the bins
        delta = bin_centers[3] - bin_centers[2]

        # Calculate the marginal entropy
        H[idim] = entropy(hist_counts, correction=correction) + np.log2(delta)

    return H


def bin_estimation(n_samples, rule="standard"):

    if rule == "sturge":
        n_bins = int(np.ceil(1 + 3.322 * np.log10(n_samples)))

    elif rule == "standard":
        n_bins = int(np.ceil(np.sqrt(n_samples)))

    else:
        raise ValueError(f"Unrecognized bin estimation rule: {rule}")

    return n_bins
