from typing import Optional, Union, Callable, Tuple
import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn.utils import check_random_state, check_array
from rbig.base import UniformMixin, DensityMixin, ScoreMixin, GaussMixin
from sklearn.base import BaseEstimator, TransformerMixin

BOUNDS_THRESHOLD = 1e-7
CLIP_MIN = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
CLIP_MAX = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))


class InverseGaussCDF(GaussMixin, DensityMixin, ScoreMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        return self

    def transform(
        self, X: np.ndarray, return_jacobian: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Calculate the transformation
        X_trans = self._transform(X, stats.norm.ppf, inverse=False)

        # Return Jacobian
        if not return_jacobian:
            return X_trans
        else:
            return X_trans, self.log_abs_det_jacobian(X_trans)

    def inverse_transform(self, X: np.ndarray):

        return self._transform(X, stats.norm.cdf, inverse=False)

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

    def abs_det_jacobian(self, X: np.ndarray):
        X = check_array(X, ensure_2d=False, copy=True)
        return 1 / stats.norm.pdf(self.transform(X))

    def log_abs_det_jacobian(self, X: np.ndarray):
        X = check_array(X, ensure_2d=False, copy=True)
        return -stats.norm.logpdf(self.transform(X))

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

        U = rng.randn(n_samples)

        X = self.inverse_transform(U)
        return X
