from typing import Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator

from rbig.transform.base import DensityMixin, BaseTransform
from rbig.layers import RBIGLayer
from rbig.stopping import StoppingCriteria


class BaseModel(BaseTransform, DensityMixin):
    """A base model defining the transformations
    
    Parameters
    ----------
    flow : RBIGLayer
        an rbig layer (marginal gaussianization + rotation)
        see rbig.density.layers for details
    
    loss : RBIGLoss
        an rbig loss function (info, maxlayers, etc)
        see rbig.density.loss for details
    """

    def __init__(self, flow: RBIGLayer, stopping_criteria: StoppingCriteria,) -> None:
        self.flow = flow
        self.stopping_criteria = stopping_criteria

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, return_jacobian=True,
    ) -> np.ndarray:
        """Forward transformation
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            data to be transformed from original domain to a "more 
            Gaussian" domain.
        
        y: not used, only here for compatability

        return_jacobian : bool, default=True
            option to return the jacobian transformation with the forward 
            transformation
        
        Returns
        -------
        NotImplementedError
        """
        raise NotImplementedError

    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        raise NotImplementedError

    def sample(self, n_samples: int = 1) -> np.ndarray:

        raise NotImplementedError
