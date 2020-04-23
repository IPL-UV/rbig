from rbig.transform.gaussianization import QuantileGaussianization, PowerGaussianization
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

from rbig.layers.base import BaseLayer
from rbig.transform import (
    HistogramGaussianization,
    OrthogonalTransform,
    KDEGaussianization,
)


class RBIGBlock(BaseLayer):
    def __init__(
        self,
        mg_method: str = "histogram",
        mg_params: Optional[Dict] = {},
        rot_params: Optional[Dict] = {},
    ) -> None:
        self.mg_method = mg_method
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
        if self.mg_method == "histogram":
            self.marginal_gaussian_ = HistogramGaussianization(**self.mg_params_)
        elif self.mg_method == "kde":
            self.marginal_gaussian_ = KDEGaussianization(**self.mg_params_)
        elif self.mg_method == "quantile":
            self.marginal_gaussian_ = QuantileGaussianization(**self.mg_params_)
        elif self.mg_method == "power":
            self.marginal_gaussian_ = PowerGaussianization(**self.mg_params_)
        else:
            raise ValueError(
                f"Unrecognized Marginal Gaussianization method: {self.mg_method}"
            )

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


class RBIGKDEParams(NamedTuple):
    # marginal transform parameters
    kde_method: str = "exact"
    bw_estimator: str = "scott"
    support_extension: Union[int, float] = 10
    n_quantiles: int = 1_000
    # rotation parameters
    rotation: str = "pca"
    random_state: int = 123
    rot_kwargs: Dict = {}

    def fit_data(self, X: np.ndarray) -> RBIGBlock:

        # initialize RBIG Block
        gauss_block = RBIGBlock(
            mg_method="kde",
            mg_params={
                "method": self.kde_method,
                "bw_estimator": self.bw_estimator,
                "support_extension": self.support_extension,
                "n_quantiles": self.n_quantiles,
            },
            rot_params={
                "rotation": self.rotation,
                "random_state": self.random_state,
                "kwargs": self.rot_kwargs,
            },
        )

        return gauss_block.fit(X)


class RBIGHistParams(NamedTuple):
    # marginal transform parameters
    nbins: int = 100
    alpha: float = 1e-6
    support_extension: Union[int, float] = 10
    # rotation parameters
    rotation: str = "pca"
    random_state: int = 123
    rot_kwargs: Dict = {}

    def fit_data(self, X: np.ndarray) -> RBIGBlock:

        # initialize RBIG Block
        gauss_block = RBIGBlock(
            mg_method="histogram",
            mg_params={
                "nbins": self.nbins,
                "alpha": self.alpha,
                "support_extension": self.support_extension,
            },
            rot_params={
                "rotation": self.rotation,
                "random_state": self.random_state,
                "kwargs": self.rot_kwargs,
            },
        )

        return gauss_block.fit(X)


class RBIGQuantileParams(NamedTuple):
    # marginal transform parameters
    n_quantiles: int = 1_000
    support_extension: Union[int, float] = 10
    subsample: int = 1e5
    random_state: int = 123
    # rotation parameters
    rotation: str = "pca"
    random_state: int = 123
    rot_kwargs: Dict = {}

    def fit_data(self, X: np.ndarray) -> RBIGBlock:

        # initialize RBIG Block
        gauss_block = RBIGBlock(
            mg_method="quantile",
            mg_params={
                "n_quantiles": self.n_quantiles,
                "subsample": self.subsample,
                "random_state": self.random_state,
                "support_extension": self.support_extension,
            },
            rot_params={
                "rotation": self.rotation,
                "random_state": self.random_state,
                "kwargs": self.rot_kwargs,
            },
        )

        return gauss_block.fit(X)


class RBIGPowerParams(NamedTuple):
    # marginal transform parameters
    standardize: bool = True
    support_extension: Union[int, float] = 10
    copy: bool = True
    # rotation parameters
    rotation: str = "pca"
    random_state: int = 123
    rot_kwargs: Dict = {}

    def fit_data(self, X: np.ndarray) -> RBIGBlock:

        # initialize RBIG Block
        gauss_block = RBIGBlock(
            mg_method="power",
            mg_params={
                "standardize": self.standardize,
                "copy": self.copy,
                "support_extension": self.support_extension,
            },
            rot_params={
                "rotation": self.rotation,
                "random_state": self.random_state,
                "kwargs": self.rot_kwargs,
            },
        )

        return gauss_block.fit(X)


RBIGParams = Union[RBIGKDEParams, RBIGHistParams, RBIGQuantileParams, RBIGPowerParams]
