from typing import Optional

import numpy as np
from scipy import stats


def negative_log_likelihood(
    Z: np.ndarray, X: np.ndarray, X_slogdet: Optional[np.ndarray] = None
) -> float:

    # calculate log probability in the latent space
    Z_logprob = stats.norm().logpdf(Z)

    # calculate the probability of the transform
    X_logprob = Z_logprob.sum(axis=1) + X_slogdet.sum(axis=1)

    # return the nll
    return np.mean(X_logprob)
