from typing import Optional

import numpy as np


class RBIGLoss:
    def __init__(
        self, n_layers: Optional[int] = None, tol_layers: Optional[int] = None
    ) -> None:
        self.n_layers = n_layers
        self.tol_layers = tol_layers
        self.loss_vals = list()

    def calculate_loss(
        self, Xtrans: np.ndarray, X: np.ndarray, dX: Optional[np.ndarray] = None
    ) -> float:
        """Abstract method to calculate the loss"""
        raise NotImplementedError

    def check_tolerance(self, current_layer: int) -> bool:
        """A helper function to check the number of iterations
        
        This outputs a bool to see if thi"""
        raise NotImplementedError
