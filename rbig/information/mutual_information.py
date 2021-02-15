from .entropy import KNNEstimator
from sklearn.base import BaseEstimator
from .ensemble import Ensemble
from typing import Optional
import numpy as np
from sklearn.utils import check_array


class MutualInformation(BaseEstimator):
    def __init__(self, estimator: str = "knn", kwargs: Optional[dict] = None) -> None:
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> BaseEstimator:
        """
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Data to be estimated.
        """
        X = check_array(X)
        if self.estimator == "knn":
            self.model = (
                KNNEstimator(**self.kwargs)
                if self.kwargs is not None
                else KNNEstimator()
            )
        elif self.estimator in ["rbig", "kde", "histogram"]:
            raise NotImplementedError(f"{self.estimator} is not implemented yet.")

        else:
            raise ValueError(f"Unrecognized estimator: {self.estimator}")
        if Y is not None:
            Y = check_array(Y)
            self._fit_mutual_info(X, Y)
        else:
            raise ValueError(f"X dims are less than 2. ")

        return self

    def _fit_multi_info(self, X: np.ndarray) -> float:

        # fit full
        model_full = self.model.fit(X)
        H_x = model_full.score(X)

        # fit marginals
        H_x_marg = 0
        for ifeature in X.T:

            model_marginal = self.model.fit(ifeature)
            H_x_marg += model_marginal.score(ifeature)

        # calcualte the multiinformation
        self.MI = H_x_marg - H_x

        return H_x_marg - H_x

    def _fit_mutual_info(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:

        # MI for X
        model_x = self.model.fit(X)
        H_x = model_x.score(X)
        print("Marginal:", H_x)

        # MI for Y
        model_y = self.model.fit(Y)
        H_y = model_y.score(Y)
        print("Marginal:", H_y)

        # MI for XY
        model_xy = self.model.fit(np.hstack([X, Y]))
        H_xy = model_xy.score(X)
        print("Full:", H_xy)

        # save the MI
        self.MI = H_x + H_y - H_xy

        return self.MI

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        return self.MI


class TotalCorrelation(BaseEstimator):
    def __init__(self, estimator: str = "knn", kwargs: Optional[dict] = None) -> None:
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> BaseEstimator:
        """
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Data to be estimated.
        """
        X = check_array(X)

        if self.estimator == "knn":
            self.model = (
                KNNEstimator(**self.kwargs)
                if self.kwargs is not None
                else KNNEstimator()
            )
        elif self.estimator in ["rbig", "kde", "histogram"]:
            raise NotImplementedError(f"{self.estimator} is not implemented yet.")

        else:
            raise ValueError(f"Unrecognized estimator: {self.estimator}")

        if y is None and X.shape[1] > 1:

            self._fit_multi_info(X)
        else:
            raise ValueError(f"X dims are less than 2. ")

        return self

    def _fit_multi_info(self, X: np.ndarray) -> float:

        # fit full
        model_full = self.model.fit(X)
        H_x = model_full.score(X)
        print("Full:", H_x)
        # fit marginals
        H_x_marg = 0
        for ifeature in X.T:

            model_marginal = self.model.fit(ifeature[:, None])

            H_xi = model_marginal.score(ifeature[:, None])
            print("Marginal:", H_xi)
            H_x_marg += H_xi

        # calcualte the multiinformation
        self.MI = H_x_marg - H_x

        return self

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        return self.MI

