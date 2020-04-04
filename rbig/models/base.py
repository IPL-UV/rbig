from typing import Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator

from rbig.transform.base import DensityMixin, BaseTransform
from rbig.layers import RBIGParams
from rbig.losses import RBIGLoss


class BaseModel(BaseTransform, DensityMixin):
    """A sequence of Gaussianization transforms.
    
    Parameters
    ----------
    """

    def __init__(self, flow: RBIGParams, loss: RBIGLoss,) -> None:
        self.flow = flow
        self.loss = loss

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        raise NotImplementedError

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, return_jacobian=False
    ) -> np.ndarray:

        raise NotImplementedError

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        raise NotImplementedError

    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        raise NotImplementedError

    def sample(self, n_samples: int = 1) -> np.ndarray:

        raise NotImplementedError
