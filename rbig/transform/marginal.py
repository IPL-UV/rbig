from typing import Callable, Optional, Tuple, Union, Dict
import numpy as np
from sklearn.utils import check_array, check_random_state
from sklearn.base import clone
from rbig.transform.base import DensityMixin, BaseTransform


class MarginalTransformation(BaseTransform, DensityMixin):
    def __init__(self, transformer) -> None:
        self.transformer = transformer

    def fit(self, X: np.ndarray) -> None:
        X = check_array(X, ensure_2d=True, copy=True)

        transforms = []

        for feature_idx in range(X.shape[1]):

            transformer = clone(self.transformer)
            transforms.append(transformer.fit(X[:, feature_idx]))

        self.transforms_ = transforms

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, ensure_2d=True, copy=True)

        for feature_idx in range(X.shape[1]):

            X[:, feature_idx] = (
                self.transforms_[feature_idx].transform(X[:, feature_idx]).squeeze()
            )

        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, ensure_2d=True, copy=True)
        for feature_idx in range(X.shape[1]):

            X[:, feature_idx] = (
                self.transforms_[feature_idx]
                .inverse_transform(X[:, feature_idx].squeeze())
                .squeeze()
            )

        return X

    def log_abs_det_jacobian(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> float:
        X = check_array(X, ensure_2d=True, copy=True)
        # print(X.shape)
        for feature_idx in range(X.shape[1]):

            t = (
                self.transforms_[feature_idx]
                .log_abs_det_jacobian(X[:, feature_idx].squeeze())
                .squeeze()
            )
            # print(t.shape)
            X[:, feature_idx] = t
        return X
