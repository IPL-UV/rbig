import numpy as np
from rbig._src.entropy import entropy_marginal
from rbig._src.uniform import MarginalHistogramUniformization
from rbig._src.invcdf import InverseGaussCDF
from rbig._src.rotation import PCARotation, RandomRotation
from tqdm import trange


def information_reduction(
    x_data, y_data, bins="auto", tol_dimensions=None, correction=True
):
    """Computes the multi-information (total correlation) reduction after a linear
    transformation

            Y = X * W
            II = I(X) - I(Y)

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data before the transformation, where n_samples is the number of samples
        and n_features is the number of features

    Y : array-like, shape (n_samples, n_features)
        Data after the transformation, where n_samples is the number of samples
        and n_features is the number of features

    tol_dimensions : float, optional
        Tolerance on the minimum multi-information difference

    Returns
    -------
    II : float
        The multi-information

    Information
    -----------
    Author: Valero Laparra
            Juan Emmanuel Johnson
    """
    # check that number of samples and dimensions are equal
    err_msg = "Number of samples for x and y should be equal."
    np.testing.assert_equal(x_data.shape, y_data.shape, err_msg=err_msg)

    n_samples, n_dimensions = x_data.shape

    # minimum multi-information heuristic
    if tol_dimensions is None or 0:
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        tol_dimensions = np.interp(n_samples, xxx, yyy)

    # preallocate data
    hx = np.zeros(n_dimensions)
    hy = np.zeros(n_dimensions)

    # calculate the marginal entropy
    hx = entropy_marginal(x_data, bins=bins, correction=correction)
    hy = entropy_marginal(y_data, bins=bins, correction=correction)

    # Information content
    I = np.sum(hy) - np.sum(hx)
    II = np.sqrt(np.sum((hy - hx) ** 2))

    p = 0.25
    if II < np.sqrt(n_dimensions * p * tol_dimensions ** 2) or I < 0:
        I = 0

    return I


def rbig_total_corr(
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

    return np.array(info_losses).sum()
