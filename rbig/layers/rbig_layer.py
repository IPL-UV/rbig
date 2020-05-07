from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

from rbig.layers.base import BaseLayer
from rbig.transform.base import BaseTransform


class RBIGLayer(BaseLayer):
    """RBIG Layer which holds the transformations defined by RBIG.
    
    This layer holds the marginal gaussianization and rotation transformations.
    This is a convenience function to fit and transform the data and return the
    log det jacobian
    
    Parameters
    ----------
    mg_transform : BaseTransform
        the base class for the marginal gaussianization
    
    rot_transform : BaseTransform
        the base class for the rotation
    """

    def __init__(
        self, mg_transform: BaseTransform, rot_transform: BaseTransform,
    ) -> None:
        self.mg_transform = mg_transform
        self.rot_transform = rot_transform

    # def __repr__(self):

    #     # loop through and get mg
    #     rep_str = ["MG Params:"]
    #     rep_str += [f"{ikey}={iparam}" for ikey, iparam in self.mg_params_.items()]
    #     rep_str += ["\nRotation Params:"]
    #     rep_str += [f"{ikey}={iparam}" for ikey, iparam in self.rot_params_.items()]
    #     return " ".join(rep_str)

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, return_jacobian=True,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # marginal transformation
        try:
            Xmg = self.mg_transform.transform(X)
        except AttributeError:
            Xmg = self.mg_transform.fit_transform(X)
        # rotation
        try:
            Xtrans = self.rot_transform.transform(Xmg)
        except AttributeError:
            Xtrans = self.rot_transform.fit_transform(Xmg)

        if not return_jacobian:
            return Xtrans
        else:
            dX_mg = self.mg_transform.log_abs_det_jacobian(X)
            dX_rot = self.rot_transform.log_abs_det_jacobian(Xmg)
            return Xtrans, dX_mg + dX_rot

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        # rotation transpose
        X = self.rot_transform.inverse_transform(X)

        # inverse marginal gaussianization
        X = self.mg_transform.inverse_transform(X)
        return X

    def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:

        # marginal transformation
        Xmg = self.mg_transform.transform(X)
        dX_mg = self.mg_transform.log_abs_det_jacobian(X)

        # rotation
        dX_rot = self.rot_transform.log_abs_det_jacobian(Xmg)

        return dX_mg + dX_rot
