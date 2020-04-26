from scipy import special
import numpy as np


def kde_cdf(data: np.ndarray, X: np.ndarray, bw: float) -> np.ndarray:
    """
    Computes the integral of a 1D pdf between two bounds.
    Parameters
    ----------
    data : np.ndarray, (n_samples, 1)
        original dataset

    X : np.ndarray, (m_samples, 1)
        Data to calculate integration

    bw : float
        the bandwidth of the kernel matrix

    Returns
    -------
    x_cdf : np.narray, (m_samples, 1)
        The result of the integral.
    """
    # ======================
    # estimate covariance factor
    # ======================
    stdev = np.sqrt(np.cov(data.squeeze(), rowvar=1) * bw ** 2)

    # ======================
    # Estimate the weights
    # ======================
    weights = 1 / data.shape[0]

    def _kde_cdf(x: float) -> float:
        normalized_low = np.ravel((-np.inf - data) / stdev)
        normalized_high = np.ravel((x - data) / stdev)
        return np.sum(
            weights * (special.ndtr(normalized_high) - special.ndtr(normalized_low))
        )

    return np.vectorize(_kde_cdf)(X)


def calculate_cdf(data: np.ndarray, X: np.ndarray, bw: float) -> np.ndarray:
    """
    Computes the integral of a 1D pdf between two bounds.
    Parameters
    ----------
    data : np.ndarray, (n_samples, 1)
        original dataset
    X : np.ndarray, (m_samples, 1)
        Data to calculate integration

    Returns
    -------
    x_cdf : np.narray, (m_samples, 1)
        The result of the integral.
    Raises
    ------
    ValueError
        If the KDE is over more than one dimension.
    """
    if X.shape[1] > 1:
        raise ValueError("integrate_box_1d() only handles 1D pdfs")

    # calculate covariance
    covariance = np.cov(X.T, rowvar=1) * bw ** 2

    stdev = np.ravel(np.sqrt(covariance))

    normalized_low = np.ravel((-np.inf - data) / stdev)
    normalized_high = np.ravel((X - data) / stdev)

    x_cdf = np.sum(special.ndtr(normalized_high) - special.ndtr(normalized_low))
    return x_cdf
