from typing import Optional, Union, Callable, Tuple
import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn.utils import check_random_state, check_array
from rbig.base import DensityMixin, ScoreMixin, DensityTransformerMixin
from rbig.utils import check_input_output_dims
from sklearn.base import BaseEstimator, TransformerMixin

BOUNDS_THRESHOLD = 1e-7
CLIP_MIN = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
CLIP_MAX = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))


class InverseGaussCDF(BaseEstimator, DensityTransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        # check inputs
        X = check_array(X, ensure_2d=True, copy=True)

        self.n_samples_, self.n_features_ = X.shape

        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # check inputs
        X = check_array(X, ensure_2d=True, copy=True)

        # Loop through features
        for feature_idx in range(X.shape[1]):

            X[:, feature_idx] = self._transform(
                X[:, feature_idx], stats.norm.ppf, inverse=False
            )

        # check_input_output_dims(
        #     X, (X.shape[0], self.n_features_), "ICDF Gauss", "Foward"
        # )
        return X

    def inverse_transform(self, X: np.ndarray):

        # check inputs
        X = check_array(X, ensure_2d=True, copy=True)

        # Loop through features
        for feature_idx in range(X.shape[1]):

            X[:, feature_idx] = self._transform(
                X[:, feature_idx], stats.norm.cdf, inverse=False
            )

        # check_input_output_dims(
        #     X, (X.shape[0], self.n_features_), "ICDF Gauss", "Inverse"
        # )
        return X

    def _transform(
        self,
        X: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        inverse: bool = False,
    ) -> np.ndarray:
        """Internal function that handles the boundary issues that can occur
        when doing the transformations. Wraps the transform and inverse
        transform functions.
        """

        # check array
        X = check_array(X, ensure_2d=False, copy=True)

        # perform transformation
        X = f(X)

        if inverse == False:
            # get boundaries
            X = np.clip(X, CLIP_MIN, CLIP_MAX)

        return X

    def log_abs_det_jacobian(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:

        X = check_array(X, ensure_2d=False, copy=True)

        # X = np.clip(X, CLIP_MIN, CLIP_MAX)

        X = -stats.norm.logpdf(self.transform(X))

        # check_input_output_dims(
        #     X, (X.shape[0], self.n_features_), "ICDF Gauss", "Jacobian"
        # )

        return X

    def sample(
        self, n_samples: int = 1, random_state: Optional[Union[RandomState, int]] = None
    ) -> np.ndarray:
        """Generate random samples from this.
        
        Parameters
        ----------
        n_samples : int, default=1
            The number of samples to generate. 
        
        random_state : int, RandomState,None, Optional, default=None
            The int to be used as a seed to generate the random 
            uniform samples.
        
        Returns
        -------
        X : np.ndarray, (n_samples, )
        """
        #
        rng = check_random_state(random_state)

        U = rng.randn(n_samples, self.n_features_)

        X = self.inverse_transform(U)
        return X
