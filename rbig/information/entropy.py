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

    @staticmethod
    def gaussian(X: np.ndarray) -> None:

        loc = X.mean(axis=0)
        scale = np.cov(X.T)

        # assume it's a Gaussian
        norm_dist = stats.norm(loc=loc, scale=scale)

        return norm_dist.entropy()


class Multivariate:
    def __init__(self, seed=123):
        self.seed = seed

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
    def expF_entropy(X: np.ndarray) -> None:

        n_dims = X.shape[1]

        # source params, theta
        theta_1 = X.mean(axis=0)
        theta_2 = np.cov(X.T)

        # natural params, eta
        eta_1 = np.linalg.inv(theta_2) @ theta_1[:, None]
        eta_2 = 0.5 * np.linalg.inv(theta_2)

        # log-normalizer, F(eta)
        eta_1_inv = np.linalg.inv(eta_2)
        f_eta = (
            0.25 * np.trace(eta_1.T @ eta_1_inv @ eta_1)
            - 0.5 * np.linalg.slogdet(eta_2)[1]
            + (n_dims / 2.0) * np.log(np.pi)
        )

        # gradient log normalizer, dF(eta)
        df_eta_1 = 0.5 * eta_1_inv @ eta_1
        df_eta_2 = -0.5 * eta_1_inv - 0.25 * (eta_1_inv @ eta_1) @ (eta_1_inv @ eta_1).T

        # outer product
        H = f_eta - ((eta_1 * df_eta_1).sum() + (eta_2 * df_eta_2).sum())

        return H

    @staticmethod
    def gaussian(X: np.ndarray) -> None:

        mean = X.mean(axis=0)
        cov = np.cov(X.T)

        # assume it's a Gaussian
        norm_dist = stats.multivariate_normal(mean=mean, cov=cov)

        return norm_dist.entropy()


class KNNEstimator(BaseEstimator, Ensemble):
    """Performs the KNN search to
    
    Parameters
    ----------
    n_neighbors : int, default = 10
        The kth neigbour to use for distance

    algorithm : str, default='auto' 
        The algorithm to use for the knn search. 
        ['auto', 'brute', 'kd_tree', 'ball_tree']
        * Auto - automatically found
        * brute - brute-force search
        * kd_tree - KDTree, fast for generalized N-point problems
        * ball_tree - BallTree, fast for generalized N-point problems
        KDTree has a faster query time but longer build time.
        BallTree has a faster build time but longer query time.
        
    n_jobs : int, default=-1
        Number of cores to use for nn search
    
    ensemble : bool, default=False
        Whether to use an ensemble of estimators via batches
    
    batch_size : int, default=100
        If ensemble=True, this determines the number of batches
        of data to use to estimate the entropy

    kwargs : any extra kwargs to use. Please see 
        sklearn.neighbors.NearestNeighbors function.
    
    min_dist : float, default=0.0
        Ensures that all distances are at least 0.0.

    Attributes
    ----------
    H_x : float,
        The estimated entropy of the data.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        algorithm: str = "auto",
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

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> BaseEstimator:
        """
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Data to be estimated.
        """

        if self.ensemble:
            self.H_x = self._fit_ensemble(X, self.batch_size)
        else:
            self.H_x = self._fit(X)

        return self

    def _fit(self, X: np.ndarray) -> float:

        n_samples, d_dimensions = X.shape

        # volume of unit ball in d^n
        vol = (np.pi ** (0.5 * d_dimensions)) / gamma(0.5 * d_dimensions + 1)

        # 1. Calculate the K-nearest neighbors
        distances = knn_distance(
            X,
            n_neighbors=self.n_neighbors + 1,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
            kwargs=self.kwargs,
        )

        # return distance to kth nearest neighbor
        distances = distances[:, -1]

        # add error margin to avoid zeros
        distances += np.finfo(X.dtype).eps

        # estimation
        return (
            d_dimensions * np.mean(np.log(distances))
            + np.log(vol)
            + psi(n_samples)
            - psi(self.n_neighbors)
        )

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        return self.H_x


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

            elif self.univariate_method == "gauss":
                H.append(self.gaussian(ifeature[:, None]))
            elif self.univariate_method in ["expF"]:
                raise NotImplementedError()
            else:
                raise ValueError(
                    f"Unrecognized entropy method: {self.univariate_method}"
                )
        H = np.transpose(H)
        return H


# volume of unit ball
def volume_unit_ball(d_dimensions: int, radii: int, norm=2) -> float:
    """Volume of the d-dimensional unit ball
    
    Parameters
    ----------
    d_dimensions : int
        Number of dimensions to estimate the volume
    
    radii : int,
    
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

    return (np.pi ** (0.5 * d_dimensions)) ** d_dimensions / gamma(b / d_dimensions + 1)


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

