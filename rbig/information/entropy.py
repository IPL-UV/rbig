import numpy as np
from scipy import stats
from typing import Union, Optional, Dict
from rbig.information.histogram import hist_entropy
from rbig.information.knn import knn_entropy
from rbig.information.kde import kde_entropy_uni
from rbig.information.gaussian import gauss_entropy_uni, gauss_entropy_multi
from sklearn.utils import check_array


def univariate_entropy(X: np.ndarray, method: str = "histogram", **kwargs) -> float:
    """Calculates the entropy given the method initiali"""
    # check input array
    X = check_array(X, ensure_2d=True)

    n_samples, n_features = X.shape

    msg = "n_features is greater than 1. Please use Multivariate instead."
    assert 1 == n_features, msg

    if method == "histogram":
        return hist_entropy(X, **kwargs)

    elif method == "knn":
        return knn_entropy(X, **kwargs)

    elif method == "kde":
        return kde_entropy_uni(X, **kwargs)

    elif method in ["gauss", "gaussian"]:
        return gauss_entropy_uni(X)
    else:
        raise ValueError(f"Unrecognized method: {method}")


def marginal_entropy(X: np.ndarray, method: str = "histogram", **kwargs) -> np.ndarray:
    # check input array
    X = check_array(X, ensure_2d=True)

    n_samples, n_features = X.shape

    # msg = "n_features is less than or equal to 2. Please use Univariate instead."
    # assert 1 < n_features, msg

    H_entropy = list()
    for ifeature in X.T:
        H_entropy.append(univariate_entropy(ifeature[:, None], method, **kwargs))

    return np.array(H_entropy)


def multivariate_entropy(X: np.ndarray, method: str = "knn", **kwargs) -> float:
    if method == "knn":
        return knn_entropy(X, **kwargs)

    elif method in ["gauss", "gaussian"]:
        return gauss_entropy_multi(X)
    else:
        raise ValueError(f"Unrecognized method: {method}")
