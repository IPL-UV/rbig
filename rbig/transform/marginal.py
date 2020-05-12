from typing import Callable, Optional, Tuple, Union, Dict
import numpy as np
from sklearn.utils import check_array, check_random_state
from sklearn.base import clone
from rbig.transform.base import DensityMixin, BaseTransform


class MarginalTransformation(BaseTransform, DensityMixin):
    """Marginal transformation for any univariate transformer.
    This class wraps any univariate transformer to do a marginal
    transformation. Includes a transform, inverse_transform and a
    log_det_jacobian method. It performs a feature-wise
    transformation on all fitted data.
    Parameters
    ----------
    Transformer : BaseTransform
        any base transform method
    Attributes
    ----------
    transforms_ : List[BaseTransform]
        a list of base transformers

    n_features_ : int
        number of features in the marginal transformation
    """

    def __init__(self, transformer: BaseTransform) -> None:
        self.transformer = transformer

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fits the data feature-wise.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            2D input data to be transformed
        
        y : np.ndarray,
            not used, for compatibility only
        
        Returns
        -------
        self : MarginalTransformation
            an instance of self
        """
        X = check_array(X, ensure_2d=True, copy=True)
        self.n_features_ = X.shape[1]

        transforms = []

        for feature_idx in range(X.shape[1]):

            transformer = clone(self.transformer)
            transforms.append(transformer.fit(X[:, feature_idx]))

        self.transforms_ = transforms

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Performs feature-wise transformation on fitted data.
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            2D input data to be transformed.
        Returns
        -------
        Xtrans : np.ndarray, (n_samples, n_features)
            2D transformed data
        """
        X = check_array(X, ensure_2d=True, copy=True)

        for feature_idx in range(X.shape[1]):

            X[:, feature_idx] = (
                self.transforms_[feature_idx].transform(X[:, feature_idx]).squeeze()
            )

        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Performs feature-wise  inverse transformation on fitted data.
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            2D input data to be transformed.
        Returns
        -------
        Xtrans : np.ndarray, (n_samples, n_features)
            2D transformed data
        """
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
    ) -> np.ndarray:
        """Calculates feature-wise log determinant jacobian of input data
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            2D input data to be transformed.
        Returns
        -------
        X_logdetjacobian : np.ndarray, (n_samples, n_features)
            Feature-wise log-determinant jacobian of data
        """
        X = check_array(X, ensure_2d=True, copy=True)
        # print(X.shape)
        for feature_idx in range(X.shape[1]):

            X[:, feature_idx] = (
                self.transforms_[feature_idx]
                .log_abs_det_jacobian(X[:, feature_idx].squeeze())
                .squeeze()
            )
            # print(t.shape)
        return X
