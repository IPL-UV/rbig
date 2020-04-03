from typing import Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma, psi
from sklearn.utils import check_array


def knn_entropy(
    X: np.ndarray, k: int = 5, algorithm: str = "brute", n_jobs: int = 1, **kwargs: Dict
) -> float:
    """Calculates the Entropy using the knn method.

    Parameters
    ----------
    X : np.ndarray, (n_samples x d_dimensions)
        The data to find the nearest neighbors for.

    k : int, default=10
        The number of nearest neighbors to find.

    algorithm : str, default='brute',
        The knn algorithm to use.
        ('brute', 'ball_tree', 'kd_tree', 'auto')

    n_jobs : int, default=-1
        The number of cores to use to find the nearest neighbors

    kwargs: dict, Optional
        any extra kwargs to use for the KNN estimator

    Returns
    -------
    H : float
        Entropy calculated from kNN algorithm
    """
    X = check_array(X, ensure_2d=True)

    # initialize KNN estimator
    n_samples, d_dimensions = X.shape

    # volume of unit ball in d^n
    vol = (np.pi ** (0.5 * d_dimensions)) / gamma(0.5 * d_dimensions + 1)

    # 1. Calculate the K-nearest neighbors

    clf_knn = NearestNeighbors(**kwargs)

    clf_knn.fit(X)

    distances, _ = clf_knn.kneighbors(X)

    # return distance to kth nearest neighbor
    distances = distances[:, -1]

    # add error margin to avoid zeros
    distances += np.finfo(X.dtype).eps

    # estimation
    return (
        d_dimensions * np.mean(np.log(distances))
        + np.log(vol)
        + psi(n_samples)
        - psi(k)
    )


# volume of unit ball
def volume_unit_ball(n_features: int, norm: int = 2) -> float:
    """Volume of the d-dimensional unit ball
    
    Parameters
    ----------
    n_features : int
        Number of dimensions to estimate the volume
    
    norm : int, default=2
        The type of ball to get the volume.
        * 2 : euclidean distance
        * 1 : manhattan distance
        * 0 : chebyshev distance
    
    Returns
    -------
    vol : float
        The volume of the d-dimensional unit ball
    """

    # get ball
    if norm == 0:
        b = float("inf")
    elif norm == 1:
        b = 1.0
    elif norm == 2:
        b = 2.0
    else:
        raise ValueError(f"Unrecognized norm: {norm}")

    return (np.pi ** (0.5 * n_features)) ** n_features / gamma(b / n_features + 1)
