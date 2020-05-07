from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn.utils import check_array, check_random_state

from rbig.transform.base import DensityMixin, BaseTransform
from rbig.utils import make_interior

BOUNDS_THRESHOLD = 1e-7
CLIP_MIN = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
CLIP_MAX = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))


class InverseGaussCDF(BaseTransform, DensityMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        # check inputs
        X = check_array(X, ensure_2d=True, copy=True)

        self.n_samples_, self.n_features_ = X.shape

        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # check inputs
        X = check_array(X.reshape(-1, 1), ensure_2d=True, copy=True)

        # make interior probability
        X = make_interior(X, bounds=(0.0, 1.0),)

        return self._transform(X, stats.norm.ppf, inverse=False)

    def inverse_transform(self, X: np.ndarray):

        # check inputs
        X = check_array(X.reshape(-1, 1), ensure_2d=True, copy=True)

        return self._transform(X, stats.norm.cdf, inverse=True)

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

    def log_abs_det_jacobian(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> np.ndarray:

        X = check_array(X.reshape(-1, 1), ensure_2d=False, copy=True)

        X = make_interior(X, bounds=(0.0, 1.0))

        X = -stats.norm.logpdf(self.transform(X))

        return X

    def sample(
        self,
        n_samples: int = 1,
        random_state: Optional[Union[RandomState, int]] = None,
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

        G = rng.randn(n_samples, self.n_features_)

        X = self.inverse_transform(G)
        return X
