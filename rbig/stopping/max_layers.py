from typing import Optional, Callable

import numpy as np

from rbig.stopping import StoppingCriteria
from rbig.density.utils import negative_log_likelihood


class MaxLayers(StoppingCriteria):
    """Uses maximum number of layers as stopping criteria
    
    Parameters
    ----------
    n_layers : int, default=50
        maximum number of layers to use
    loss_func : Callable, default=negative_log_likelihood
        a callable function to calculate the loss between layers.
        The default is negative log likelihood
    
    Attributes
    ----------
    losses_ = List[float]
        the loss values calculated at every iteration
    """

    def __init__(
        self, n_layers: int = 50, loss_func: Optional[Callable] = None,
    ) -> None:
        super().__init__(n_layers=n_layers, tol_layers=None)
        self.loss_func = loss_func

    def calculate_loss(
        self, Xtrans: np.ndarray, X: np.ndarray, X_slogdet: Optional[np.ndarray] = None,
    ) -> float:
        """Calculates the loss using the callable function
        
        Parameters
        ----------
        Xtrans : np.ndarray, (n_samples, n_features)
            transformed variable
        X : np.ndarray, (n_samples, n_features)
            original variable
        dX : np.ndarray, (n_features, n_features), Optional
            log det jacobian of the transformation
            Not needed, just there for compatibility
        
        Returns
        -------
        loss : float
            the loss based on the callable function
            if class is initialized with None, the default is
            the negative log likelihood.
        """
        # compute loss (default nll)
        if self.loss_func is None:
            loss_val = negative_log_likelihood(Xtrans, X, X_slogdet)
        else:
            loss_val = self.loss_func(Xtrans, X, X_slogdet)

        # add loss values
        self.losses_.append(loss_val)
        # print(self.losses_[-1])

        return loss_val

    def check_tolerance(self, current_layer: int) -> bool:
        """Checks if we have reached the max number of layers. 
        
        Returns a bool"""
        if current_layer >= self.n_layers:
            return False
        else:
            return True
