from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BaseLayer(BaseEstimator, TransformerMixin):
    def __init__(
        self, mg_params: Optional[Dict] = {}, rot_params: Optional[Dict] = {}
    ) -> None:
        self.mg_params_ = mg_params
        self.rot_params_ = rot_params

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        raise NotImplementedError

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, return_jacobian=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        raise NotImplementedError

    def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
