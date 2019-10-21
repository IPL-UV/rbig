import numpy as np
from scipy.special import gamma, psi
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.utils import gen_batches
from .ensemble import Ensemble
from sklearn.utils import check_random_state, check_array
from typing import Optional, Union
import statsmodels.api as sm


class KNNEstimator(BaseEstimator, Ensemble):
    def __init__(
        self,
        n_neighbors: int = 20,
        algorithm: str = "brute",
        n_jobs: int = -1,
        ensemble=False,
        batch_size=100,
        kwargs: Optional[dict] = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.kwargs = kwargs
        self.batch_size = batch_size

    def fit(self, X: np.ndarray) -> BaseEstimator:

        self.vol = volume_unit_ball(X.shape[1])

        if self.ensemble:
            self.H_x = self._fit_ensemble(X, self.batch_size)
        else:
            self.H_x = self._fit(X)

        return self

    def _fit(self, X: np.ndarray) -> float:

        # 1. Calculate the K-nearest neighbors

        dist = knn_distance(
            X,
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
            kwargs=self.kwargs,
        )

        return (
            np.log(X.shape[0] - 1)
            - psi(self.n_neighbors)
            + np.log(self.vol)
            + (X.shape[1] / X.shape[0]) * np.log(dist[:, self.n_neighbors - 1]).sum()
        )

    def score(self, X):

        return self.H_x


class Univariate:
    def __init__(self):
        pass

    @staticmethod
    def histogram_entropy(
        X: np.ndarray, bins: Union[str, int] = "auto", correction: bool = True
    ) -> float:
        """Calculates the entropy using the histogram. Option to do a Miller Maddow
        correction.
        
        Parameters
        ----------
        """
        # get histogram
        hist_counts = np.histogram(X, bins=bins, range=(X.min(), X.max()))

        # create random variable
        hist_dist = stats.rv_histogram(hist_counts)

        # calculate entropy
        H = hist_dist.entropy()

        # MLE Estimator with Miller-Maddow Correction
        if correction == True:
            H += 0.5 * (np.sum(hist_counts[0] > 0) - 1) / hist_counts[0].sum()

        return H

    @staticmethod
    def knn_entropy(X: np.ndarray, k: int = 5, algorithm="brute", n_jobs=1):
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
        
            
        Returns
        -------
        H : float
            Entropy calculated from kNN algorithm
        """
        # initialize estimator
        knn_clf = KNNEstimator(n_neighbors=k, algorithm=algorithm, n_jobs=n_jobs)

        knn_clf.fit(X)
        return knn_clf.score(X)

    @staticmethod
    def kde_entropy(
        X: np.ndarray,
        kernel="gau",
        bw="normal_reference",
        gridsize=50,
        adjust=1,
        cut=3,
        clip=(-np.inf, np.inf),
    ):

        # initialize KDE
        kde_density = sm.nonparametric.KDEUnivariate(X)

        kde_density.fit(bw=bw, gridsize=gridsize, adjust=adjust, cut=cut, clip=clip)
        return kde_density.entropy


class MarginalEntropy(Univariate):
    def __init__(
        self,
        univariate_method: str = "knn",
        bins: str = "auto",
        correction: bool = True,
        k: int = 10,
        kernel: str = "gau",
        bw: str = "normal_reference",
        kwargs: Optional[dict] = None,
    ) -> None:
        self.univariate_method = univariate_method
        self.bins = bins
        self.correction = correction
        self.k = k
        self.kernel = kernel
        self.bw = bw
        self.kwargs = kwargs

    def entropy(self, X):
        if self.kwargs is None:
            kwargs = dict()
        else:
            kwargs = self.kwargs

        H = list()

        # Loop through and calculate the entropy for the marginals
        for ifeature in X.T:

            if self.univariate_method == "knn":
                H.append(self.knn_entropy(ifeature[:, None], k=self.k, **kwargs))
            elif self.univariate_method == "kde":
                H.append(
                    self.kde_entropy(
                        ifeature[:, None], kernel=self.kernel, bw=self.bw, **kwargs
                    )
                )
            elif self.univariate_method == "histogram":
                H.append(
                    self.histogram_entropy(
                        ifeature[:, None],
                        bins=self.bins,
                        correction=self.correction,
                        **kwargs,
                    )
                )
            else:
                raise ValueError(
                    f"Unrecognized entropy method: {self.univariate_method}"
                )
        H = np.transpose(H)
        return H


# volume of unit ball
def volume_unit_ball(d_dimensions: int) -> float:
    """Volume of the d-dimensional unit ball
    
    Parameters
    ----------
    d_dimensions : int
        Number of dimensions to estimate the volume
        
    Returns
    -------
    vol : float
        The volume of the d-dimensional unit ball
    """
    return (np.pi ** (0.5 * d_dimensions)) / gamma(0.5 * d_dimensions + 1)


# KNN Distances
def knn_distance(
    X: np.ndarray,
    n_neighbors: int = 20,
    algorithm: str = "brute",
    n_jobs: int = -1,
    kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Light wrapper around sklearn library.
    
    Parameters
    ----------
    X : np.ndarray, (n_samples x d_dimensions)
        The data to find the nearest neighbors for.
    
    n_neighbors : int, default=20
        The number of nearest neighbors to find.
    
    algorithm : str, default='brute', 
        The knn algorithm to use.
        ('brute', 'ball_tree', 'kd_tree', 'auto')
    
    n_jobs : int, default=-1
        The number of cores to use to find the nearest neighbors
    
    kwargs : dict, Optional
        Any extra keyword arguments.
        
    Returns
    -------
    distances : np.ndarray, (n_samples x d_dimensions)
    """
    if kwargs:
        clf_knn = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm=algorithm, n_jobs=n_jobs, **kwargs
        )
    else:

        clf_knn = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm=algorithm, n_jobs=n_jobs
        )

    clf_knn.fit(X)

    dists, _ = clf_knn.kneighbors(X)

    return dists

