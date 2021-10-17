from sklearn.utils.validation import check_array
from rbig._src.model import RBIG
from typing import Union
import numpy as np


class MutualInfoRBIG:
    def __init__(
        self,
        bins: Union[int, str] = "auto",
        alpha: float = 1e-10,
        bound_ext: float = 0.3,
        eps: float = 1e-10,
        rotation: str = "PCA",
        zero_tolerance: int = 60,
        max_layers: int = 1_000,
    ):
        self.bins = bins
        self.alpha = alpha
        self.bound_ext = bound_ext
        self.eps = eps
        self.rotation = rotation
        self.zero_tolerance = zero_tolerance
        self.max_layers = max_layers

    def fit(self, X, Y):
        X = check_array(X, ensure_2d=True, copy=True)
        Y = check_array(Y, ensure_2d=True, copy=True)

        # TC for Model I
        rbig_model_X = RBIG(
            bins=self.bins,
            alpha=self.alpha,
            bound_ext=self.bound_ext,
            eps=self.eps,
            rotation=self.rotation,
            zero_tolerance=self.zero_tolerance,
            max_layers=self.max_layers,
        )
        X = rbig_model_X.fit_transform(X)
        self.rbig_model_X = rbig_model_X

        # TC for RBIG Model II
        rbig_model_Y = RBIG(
            bins=self.bins,
            alpha=self.alpha,
            bound_ext=self.bound_ext,
            eps=self.eps,
            rotation=self.rotation,
            zero_tolerance=self.zero_tolerance,
            max_layers=self.max_layers,
        )
        Y = rbig_model_Y.fit_transform(Y)
        self.rbig_model_Y = rbig_model_Y

        # TC for RBIG Model X, Y
        XY = np.hstack([X, Y])

        rbig_model_XY = RBIG(
            bins=self.bins,
            alpha=self.alpha,
            bound_ext=self.bound_ext,
            eps=self.eps,
            rotation=self.rotation,
            zero_tolerance=self.zero_tolerance,
            max_layers=self.max_layers,
        )
        rbig_model_XY.fit(XY)

        self.rbig_model_XY = rbig_model_XY

        return self

    def mutual_info(self):
        return self.rbig_model_XY.total_correlation()
