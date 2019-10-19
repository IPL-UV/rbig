import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import slogdet, inv
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from rbig.base import ScoreMixin
from typing import Optional
from rbig.ica import OrthogonalICA

# TODO - Whitening transform - https://github.com/lucastheis/isa/blob/master/code/transforms/whiteningtransform.py


class OrthogonalTransform(BaseEstimator, TransformerMixin, ScoreMixin):
    """This transformation performs an orthogonal (orthonormal) rotation of
    your input X
    
    Parameters
    ----------
    rotation : str, ['pca', 'random_o', 'random_so']
        the orthogonal rotation type
        pca       - prinicpal components analysis
        random_o  - random orthogonal matrix from Haar O() group
        random_so - random orthogonal matrix from Haar SO() group
        ica       - independent components analysis with orthogonal
                    constraints

    random_state : int,
        To control the seed for the random initializations

    kwargs : dict,
        The keyword arguments for the transformation

    Attributes
    ----------
    R : np.ndarray
        The rotation matrix used to transform and inverse transform
        fitted to data X  
    """

    def __init__(
        self, rotation: str = "pca", random_state: int = 123, kwargs: dict = None
    ):
        self.rotation = rotation
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Finds the Orthogonal transformation using PCA
        or some random orthogonal rotation matrix.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples x n_features)
            The data to be fitted to find the transformation

        Returns
        -------
        obj : object
        """
        X = check_array(X)

        # get data dimensions
        d_dimensions = X.shape[1]

        if self.rotation == "pca":

            if self.kwargs is not None:
                model = PCA(random_state=self.random_state, **self.kwargs)

            else:

                model = PCA(random_state=self.random_state)

            model.fit(X)

            self.R = model.components_.T

        elif self.rotation == "random_o":

            self.R = stats.special_ortho_group.rvs(
                dim=d_dimensions, random_state=self.random_state
            )

        elif self.rotation == "random_so":

            self.R = stats.ortho_group.rvs(
                dim=d_dimensions, random_state=self.random_state
            )

        elif self.rotation == "ica":

            if self.kwargs is not None:
                model = OrthogonalICA(random_state=self.random_state, **self.kwargs)

            else:

                model = OrthogonalICA(random_state=self.random_state)

            model.fit(X)

            self.R = model.components_.T
        else:
            raise ValueError(f"Unrecognized rotation: {self.rotation}")

        return self

    def transform(self, X):

        X = check_array(X)

        return X @ self.R

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
        X = check_array(X)

        return X @ self.R.T

    def score_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Calculates the log determinant Jacobian for each sample.
        We have constrained our rotation matrix to be orthogonal.
        So the logdetj will be zero. Therefore the score will simply be
        a vector of zeros of length n_features.
        
        Parameters
        ----------
        X : np.ndarray, 
            The data matrix. We just need the number of features.
        
        y : np.ndarray, default=None 
            Not used in the transformation but kept for compatibility.
        """
        X = check_array(X)

        return np.zeros((X.shape[0],))

