from statsmodels.distributions.empirical_distribution import ECDF
from typing import Optional
import numpy as np


def estimate_empirical_cdf(X: np.ndarray, X_new: Optional[np.ndarray] = None):

    # initialize ecdf
    ecdf_f = ECDF(X)
    if X_new is None:
        return ecdf_f(X)
    else:
        return ecdf_f(X_new)
