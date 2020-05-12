import numpy as np
from scipy import stats


def gauss_entropy_uni(X: np.ndarray) -> float:

    loc = X.mean(axis=0)
    scale = np.cov(X.T)

    # assume it's a Gaussian
    norm_dist = stats.norm(loc=loc, scale=scale)

    return norm_dist.entropy()[0]


def gauss_entropy_multi(X: np.ndarray) -> float:

    mean = X.mean(axis=0)
    cov = np.cov(X.T)

    # assume it's a Gaussian
    norm_dist = stats.multivariate_normal(mean=mean, cov=cov)

    return norm_dist.entropy()
