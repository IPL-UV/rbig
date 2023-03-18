from typing import Optional
import numpy as np
from scipy.stats import multivariate_normal
from rbig._src.total_corr import information_reduction
from rbig._src.uniform import MarginalHistogramUniformization, MarginalKDEUniformization
from rbig._src.invcdf import InverseGaussCDF
from rbig._src.rotation import PCARotation, RandomRotation, ICARotation
from rbig._src.base import FlowModel
from tqdm.autonotebook import trange


def train_rbig_info_loss(
    X: np.ndarray,
    uniformizer: str = "hist",
    bins: str = "auto",
    alpha: float = 1e-10,
    bound_ext: float = 0.3,
    eps: float = 1e-10,
    rotation: str = "PCA",
    zero_tolerance: int = 60,
    max_layers: int = 1_000,
    random_state: Optional[int] = 123,
    max_iter: int = 10,
):

    Z = X.copy()
    transformations = []
    info_losses = []

    # initialize loss
    with trange(max_layers) as pbar:
        for ilayer in pbar:
            X_before = Z.copy()
            # Marginal Uniformization
            if uniformizer == "hist":
                ibijector = MarginalHistogramUniformization(
                    X=Z, bound_ext=bound_ext, bins=bins, alpha=alpha
                )
            elif uniformizer == "kde":
                ibijector = MarginalKDEUniformization(
                    X=Z, bound_ext=bound_ext, fft=True
                )

            transformations.append(ibijector)
            Z = ibijector.forward(Z)

            # Inverse Gauss CDF
            ibijector = InverseGaussCDF(eps=eps)
            transformations.append(ibijector)
            Z = ibijector.forward(Z)

            # Rotation
            if rotation.lower() == "pca":
                ibijector = PCARotation(X=Z, random_state=random_state)
            elif rotation.lower() == "random":
                ibijector = RandomRotation(X=Z)
            elif rotation.lower() == "ica":
                ibijector = ICARotation(
                    X=Z, random_state=random_state, max_iter=max_iter
                )
            else:
                raise ValueError(f"Unrecognized rotation method: {rotation}")

            transformations.append(ibijector)
            Z = ibijector.forward(Z)

            info_red = information_reduction(
                x_data=X_before,
                y_data=Z,
                bins=bins,
            )

            info_losses.append(info_red)

            if ilayer > zero_tolerance:
                if np.sum(np.abs(info_losses[-zero_tolerance:])) == 0:
                    info_losses = info_losses[:-zero_tolerance]
                    transformations = transformations[: -3 * zero_tolerance]
                    pbar.set_description(
                        f"Completed! (Total Info Red: {np.sum(info_losses):.4f})"
                    )
                    break

            pbar.set_description(f"Info Red: {info_red:.2e}")

    base_dist = multivariate_normal(mean=np.zeros(X.shape[1]), cov=np.ones(X.shape[1]))

    # init flow model
    gf_model = FlowModel(transformations, base_dist)

    gf_model.info_loss = np.array(info_losses)

    return gf_model
