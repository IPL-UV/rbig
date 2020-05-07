from typing import Dict, Optional

import numpy as np
from scipy.interpolate import interp1d
from sklearn.utils import check_array

from rbig.information.entropy import marginal_entropy
from rbig.stopping import StoppingCriteria


class NegEntropyLoss(StoppingCriteria):
    """Uses the negative entropy as a stopping criteria.
    
    Parameters
    ----------
    tol_layers : int, default=50
        the number of layers before we stop the iterations.
    
    Attributes
    ----------
    losses_ : List[float]
        a list of all loss values
    
    """

    def __init__(self, tol_layers: int = 50,) -> None:

        super().__init__(n_layers=None, tol_layers=tol_layers)
        self.losses_ = list()

    def calculate_loss(
        self, Xtrans: np.ndarray, X: np.ndarray, X_slogdet: Optional[np.ndarray] = None,
    ) -> float:
        """Calculates the loss based on the difference in negentropy"""

        delta_neg = diff_negentropy(Xtrans, X, X_slogdet)
        # add loss values
        self.losses_.append(delta_neg)

        # return change in marginal entropy
        return delta_neg

    def check_tolerance(self, current_layer: int) -> bool:
        """Check if the tolerance layers have been reached."""
        if current_layer <= self.tol_layers:
            return True
        else:
            # get the abs sum of the last n layers
            tol = np.abs(self.losses_[-self.tol_layers :]).sum()

            # calculate the changes
            if tol == 0:
                # no changes, don't use the last n layers
                self.losses_ = self.losses_[: self.tol_layers]
                return False
            else:
                # continue
                return True


def diff_negentropy(Xtrans, X, X_slogdet) -> float:
    """Difference in negentropy between invertible transformations
    Function to calculate the difference in negentropy between a variable
    X and an invertible transformation
    
    Parameters
    ----------
    Xtrans : np.ndarray, (n_samples, n_features)
        the transformed variable
    X : np.ndarray, (n_samples, n_features)
        the variable
    X_slogdet: np.ndarray, (n_features, n_features)
        the log det jacobian of the transformation
    Returns
    -------
    x_negent : float
        the difference in negentropy between transformations
    """
    # calculate the difference in negentropy
    delta_neg = np.mean(
        0.5 * np.linalg.norm(Xtrans, 2) ** 2
        - X_slogdet
        - 0.5 * np.linalg.norm(X, 2) ** 2
    )

    # return change in marginal entropy
    return delta_neg
