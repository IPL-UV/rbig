from typing import Optional

import numpy as np
from numpy.linalg import inv, slogdet
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array

from rbig.transform.base import BaseTransform

# TODO - Whitening transform - https://github.com/lucastheis/isa/blob/master/code/transforms/whiteningtransform.py
# TODO - Orthogonality Checker - https://github.com/davidinouye/destructive-deep-learning/blob/master/ddl/linear.py#L327


class OrthogonalTransform(BaseTransform):
    """This transformation performs an orthogonal (orthonormal) rotation of
    your input X
    
    Parameters
    ----------
    rotation : str, ['pca', 'random_o', 'random_so']
        the orthogonal rotation type
        pca       - prinicpal components analysis
        random_o  - random orthogonal matrix from Haar O() group
        random_so - random orthogonal matrix from Haar SO() group

    random_state : int,
        To control the seed for the random initializations

    kwargs : dict,
        The keyword arguments for the transformation

    Attributes
    ----------
    R_ : np.ndarray, (n_features, n_features)
        The rotation matrix used to transform and inverse transform
        fitted to data X
    
    n_features_ : int, (n_features)
        the number of dimensions
    """

    def __init__(
        self, rotation: str = "pca", random_state: int = 123, kwargs: dict = None
    ):
        self.rotation = rotation
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fits the Linear transform to the data
        
        This method finds the Orthogonal transformation using PCA
        or some random orthonormal transformation.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            The data to be fitted to find the transformation

        y : not used, only for compatibility

        Returns
        -------
        obj : instance of self
        """
        X = check_array(X, ensure_2d=True, copy=True)

        # get data dimensions
        self.n_features_ = X.shape[1]

        if self.rotation == "pca":

            if self.kwargs is not None:
                model = PCA(random_state=self.random_state, **self.kwargs)

            else:

                model = PCA(random_state=self.random_state)

            model.fit(X)

            self.R_ = model.components_.T

        elif self.rotation == "random_o":

            self.R_ = stats.special_ortho_group.rvs(
                dim=self.n_features_, random_state=self.random_state
            )

        elif self.rotation == "random_so":

            self.R_ = stats.ortho_group.rvs(
                dim=self.n_features_, random_state=self.random_state
            )

        else:
            raise ValueError(f"Unrecognized rotation: {self.rotation}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:

        X = check_array(X, ensure_2d=True, copy=True)

        return np.dot(X, self.R_)

    def inverse_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Apply inverse destructive transformation to X.

        Parameters
        ----------
        X : np.ndarray,  (n_samples x n_features)
            New data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : np.ndarray, default=None
            Not used in the transformation but kept for compatibility.

        Returns
        -------
        X_new : np.ndarray, shape (n_samples, n_features)
            Transformed data.
        """
        X = check_array(X, ensure_2d=True, copy=True)

        return X @ self.R_.T

    def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Calculates the log abs det jacobian for some input
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            the input data
        
        Returns
        -------
        dX : np.ndarray, (n_samples, n_features)
            the log abs det jacobian for some input
        """

        X = check_array(X, ensure_2d=True, copy=True)

        return np.zeros(X.shape)
