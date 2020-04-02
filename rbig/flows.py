from typing import Iterable, Optional
import numpy as np
from sklearn.base import BaseEstimator
from rbig.base import DensityTransformerMixin, ScoreMixin


class GaussianizationFlowModel(BaseEstimator, DensityTransformerMixin, ScoreMixin):
    """A sequence of Gaussianization transforms.
    
    Parameters
    ----------
    """

    def __init__(self, flows: Iterable) -> None:
        self.flows = flows

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        pass

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        pass

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pass

    def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        pass

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        pass

    def sample(self, n_samples: int) -> np.ndarray:
        pass
