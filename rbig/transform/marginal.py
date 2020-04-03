from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from rbig.transform import HistogramGaussianization


class MarginalGaussianization(BaseEstimator, TransformerMixin):
    def __init__(self, kwargs=None):
        self.transformer = HistogramGaussianization()
        if kwargs is None:
            kwargs = dict()
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        # check input array
        X = check_array(X, ensure_2d=True, copy=True)

        marginal_transforms = list()

        # fit per dimension
        for ifeature in X.T:
            marginal_transforms.append(
                HistogramGaussianization(**self.kwargs).fit(ifeature)
            )

        self.marginal_transforms = marginal_transforms

        # check
        assert len(self.marginal_transforms) == X.shape[1]

        return self

    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        return_jacobian: bool = False,
    ) -> np.ndarray:

        # check input array
        X = check_array(X, ensure_2d=True, copy=True).T

        assert len(self.marginal_transforms) == X.shape[0]

        # fit per dimension
        if return_jacobian:
            jacobians = np.zeros(X.shape)
            for itrans, ifeature in enumerate(X):
                X[itrans, :], jacobians[itrans, :] = self.marginal_transforms[
                    itrans
                ].transform(ifeature, return_jacobian=return_jacobian)
            return X.T, jacobians.T
        else:
            for itrans, ifeature in enumerate(X):
                X[itrans, :] = self.marginal_transforms[itrans].transform(
                    ifeature, return_jacobian=return_jacobian
                )
            return X.T

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        # check input array
        X = check_array(X, ensure_2d=True, copy=True).T

        assert len(self.marginal_transforms) == X.shape[0]

        # fit per dimension
        for itrans, ifeature in enumerate(X):
            X[itrans, :] = self.marginal_transforms[itrans].inverse_transform(ifeature)
        return X.T

    def log_abs_det_jacobian(self, X: np.ndarray, log: bool = True) -> np.ndarray:

        # check input array
        X = check_array(X, ensure_2d=True, copy=True).T

        assert len(self.marginal_transforms) == X.shape[0]

        jacobians = np.zeros(X.shape)
        for itrans, ifeature in enumerate(X):
            jacobians[itrans, :] = self.marginal_transforms[itrans].abs_det_jacobian(
                ifeature, log=log
            )
        return jacobians.T
