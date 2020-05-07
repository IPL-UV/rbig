from scipy import special
import numpy as np
from typing import Optional

import numpy as np
from scipy import stats


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


def negative_log_likelihood(
    Z: np.ndarray, X: np.ndarray, X_slogdet: Optional[np.ndarray] = None
) -> float:
    """Calculates the negative log-likelihood of an invertible transformation.
    
    Parameters
    ----------
    Z : np.ndarray, (n_samples, n_features)
        the transformed data
    X : np.ndarray, (n_samples, n_features)
        the original data
        Not used, for compatibility only.
    X_slogdet : np.ndarray, (n_features, n_features)
        the log det jacobian of the transformed variable
    
    Returns
    -------
    nll : float
        the negative log likelihood of Z given the transformation
        X_slogdet
    """

    # calculate log probability in the latent space
    Z_logprob = stats.norm().logpdf(Z)

    # calculate the probability of the transform
    X_logprob = Z_logprob.sum(axis=1) + X_slogdet.sum(axis=1)

    # return the nll
    return np.mean(X_logprob)
