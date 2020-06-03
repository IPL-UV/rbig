from typing import Dict, Optional

import numpy as np

from rbig.information import information_reduction
from rbig.stopping import StoppingCriteria


class InfoLoss(StoppingCriteria):
    """Stopping criteria based on the amount of information
    loss between invertible transformations

    This method will check the change in total correlation (multi-information)
    between an invertible transformation. After some number of tolerance layers
    this method will output a bool to say stop.

    see rbig.information.totalcorrinformation_reduction for more information
    about how the information loss is calculated.
    
    Parameters
    ----------
    method : str, default='histogram'
        method used to calculate the marginal entropy
        * histogram - fastest
        * kde - slowest
        * kdefft - medium
        * knn - fast
    p_value : float,
        value for cut-off for tolerance of information reduction.
        a less p-value is more strict and will result in the method
        taking longer to converge.
    kwargs: dict,
        any kwargs to be used for the entropy estimator
        see rbig.information.entropy for more details
    tol_layers : int, default=50
        minimum number of layers that need to have no changes before
        going to zero.
    
    Attributes
    ----------
    losses_ : List[float]
        list of loss values that accumulated per layer
    """

    def __init__(
        self,
        method: str = "histogram",
        p_value: float = 0.25,
        kwargs: Dict = {},
        tol_layers: int = 50,
    ) -> None:

        super().__init__(n_layers=None, tol_layers=tol_layers)
        self.method = method
        self.p_value = p_value
        self.kwargs = kwargs
        self.name_ = "info_loss"
        self.losses_ = list()

    def calculate_loss(
        self, Xtrans: np.ndarray, X: np.ndarray, dX: Optional[np.ndarray] = None,
    ) -> float:
        """Calculates the loss based on the difference in information between
        transformations
        
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
        info_loss : float
            the amount of information lossed from transforamtion
        """
        delta_tc = information_reduction(
            X, Xtrans, method=self.method, p_value=self.p_value, kwargs=self.kwargs,
        )
        # add loss values
        self.losses_.append(delta_tc)

        # return change in marginal entropy
        return delta_tc

    def check_tolerance(self, current_layer: int) -> bool:
        """Helper function to check if method converged or not."""
        if current_layer <= self.tol_layers:
            return True
        else:
            # get the abs sum of the last n layers
            tol = np.sum(self.losses_[-self.tol_layers :])
            # print("Toleranrce:", tol)

            # calculate the changes
            if tol < 0.001:
                # no changes, don't use the last n layers
                # n_layers = len(self.losses_)
                n = self.tol_layers
                # print(self.tol_layers, len(self.losses_))
                self.losses_ = self.losses_[:-n]
                # print(len(self.losses_))
                return False
            else:
                # continue
                return True
