from typing import Optional, Callable

import numpy as np

from rbig.losses import RBIGLoss
from rbig.losses.likelihood import negative_log_likelihood


class MaxLayersLoss(RBIGLoss):
    def __init__(
        self, n_layers: int = 50, loss_func: Optional[Callable] = None,
    ) -> None:
        super().__init__(n_layers=n_layers, tol_layers=None)
        self.loss_func = loss_func

    def calculate_loss(
        self, Xtrans: np.ndarray, X: np.ndarray, X_slogdet: Optional[np.ndarray] = None
    ) -> float:

        # compute loss (default nll)
        if self.loss_func is None:
            loss_val = negative_log_likelihood(Xtrans, X, X_slogdet)
        else:
            loss_val = self.loss_func(Xtrans, X, X_slogdet)

        # add loss values
        self.loss_vals.append(loss_val)

        return loss_val

    def check_tolerance(self, current_layer: int) -> bool:
        if current_layer >= self.n_layers:
            return False
        else:
            return True
