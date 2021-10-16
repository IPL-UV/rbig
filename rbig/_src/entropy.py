from typing import Union
import numpy as np
from scipy.stats import rv_histogram


def univariate_entropy(
    X: np.ndarray, bins: Union[int, str] = "auto", correction: bool = True
) -> np.ndarray:

    # Get histogram (use default bin estimation)
    hist = np.histogram(
        a=X,
        bins=bins,
        range=(X.min(), X.max()),
    )

    # Calculate differential entropy
    H = rv_histogram(hist).entropy()

    if correction:
        H += 0.5 * (np.sum(hist[0] > 0) - 1) / np.sum(hist[0])

    return H


def entropy_marginal(
    data: np.ndarray, bins="auto", correction: bool = True
) -> np.ndarray:
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
    d_dimensions = data.shape[1]

    H = np.zeros(d_dimensions)

    for idim, i_data in enumerate(data.T):

        H[idim] = univariate_entropy(i_data, bins=bins, correction=correction)

    return H


import numpy as np
from rbig._src.uniform import MarginalHistogramUniformization
from rbig._src.invcdf import InverseGaussCDF
from rbig._src.rotation import PCARotation, RandomRotation
from tqdm import trange
from rbig._src.total_corr import information_reduction


def rbig_entropy(
    X: np.ndarray,
    bins: str = "auto",
    alpha: float = 1e-10,
    bound_ext: float = 0.3,
    eps: float = 1e-10,
    rotation: str = "PCA",
    zero_tolerance: int = 60,
    max_layers: int = 1_000,
):

    Z = X.copy()
    info_losses = []

    # initialize loss
    with trange(max_layers) as pbar:
        for ilayer in pbar:
            X_before = Z.copy()
            # Marginal Uniformization
            ibijector = MarginalHistogramUniformization(
                X=Z, bound_ext=bound_ext, bins=bins, alpha=alpha
            )

            Z = ibijector.forward(Z)

            # Inverse Gauss CDF
            ibijector = InverseGaussCDF(eps=eps)
            Z = ibijector.forward(Z)

            # Rotation
            if rotation.lower() == "pca":
                ibijector = PCARotation(X=Z)
            elif rotation.lower() == "random":
                ibijector = RandomRotation(X=Z)
            else:
                raise ValueError(f"Unrecognized rotation method: {rotation}")

            Z = ibijector.forward(Z)

            info_red = information_reduction(x_data=X_before, y_data=Z, bins=bins)

            info_losses.append(info_red)

            if ilayer > zero_tolerance:
                if np.sum(np.abs(info_losses[-zero_tolerance:])) == 0:
                    info_losses = info_losses[:-zero_tolerance]
                    pbar.set_description(
                        f"Completed! (Total Info Red: {np.sum(info_losses):.4f})"
                    )
                    break

            pbar.set_description(f"Info Red: {info_red:.2e}")
    Hx = entropy_marginal(X)
    return Hx.sum() - np.array(info_losses).sum()
