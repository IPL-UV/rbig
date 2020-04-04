from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

from rbig.layers.base import BaseLayer
from rbig.transform import HistogramGaussianization, OrthogonalTransform


class RBIGBlock(BaseLayer):
    def __init__(
        self, mg_params: Optional[Dict] = {}, rot_params: Optional[Dict] = {}
    ) -> None:
        self.mg_params_ = mg_params
        self.rot_params_ = rot_params

    def __repr__(self):

        # loop through and get mg
        rep_str = ["MG Params:"]
        rep_str += [f"{ikey}={iparam}" for ikey, iparam in self.mg_params_.items()]
        rep_str += ["\nRotation Params:"]
        rep_str += [f"{ikey}={iparam}" for ikey, iparam in self.rot_params_.items()]
        return " ".join(rep_str)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        # marginal transformation
        self.marginal_gaussian_ = HistogramGaussianization(**self.mg_params_)

        X = self.marginal_gaussian_.fit_transform(X)

        # rotation
        self.rotation_ = OrthogonalTransform(**self.rot_params_).fit(X)

        return self

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, return_jacobian=False
    ) -> Tuple[np.ndarray, np.ndarray]:

        # marginal transformation
        Xmg = self.marginal_gaussian_.transform(X)

        # rotation
        Xtrans = self.rotation_.transform(Xmg)

        if not return_jacobian:
            return Xtrans
        else:
            dX_mg = self.marginal_gaussian_.log_abs_det_jacobian(X)
            dX_rot = self.rotation_.log_abs_det_jacobian(Xmg)
            return Xtrans, dX_mg + dX_rot

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        # rotation transpose
        X = self.rotation_.inverse_transform(X)

        # inverse marginal gaussianization
        X = self.marginal_gaussian_.inverse_transform(X)
        return X

    def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:

        # marginal transformation
        Xmg = self.marginal_gaussian_.transform(X)
        dX_mg = self.marginal_gaussian_.log_abs_det_jacobian(X)

        # rotation
        dX_rot = self.rotation_.log_abs_det_jacobian(Xmg)

        return dX_mg + dX_rot


class RBIGParams(NamedTuple):
    # marginal transform parameters
    nbins: int = 100
    alpha: float = 1e-6
    # rotation parameters
    rotation: str = "pca"
    random_state: int = 123
    rot_kwargs: Dict = {}

    def fit_data(self, X: np.ndarray) -> RBIGBlock:

        # initialize RBIG Block
        gauss_block = RBIGBlock(
            mg_params={"nbins": self.nbins, "alpha": self.alpha,},
            rot_params={
                "rotation": self.rotation,
                "random_state": self.random_state,
                "kwargs": self.rot_kwargs,
            },
        )

        return gauss_block.fit(X)
