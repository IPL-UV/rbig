from typing import Optional, NamedTuple, Dict

import numpy as np
from scipy import stats
from rbig.information.reduction import information_reduction


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


class MaxLayers(RBIGLoss):
    def __init__(self, n_layers: int = 50) -> None:
        super().__init__(n_layers=n_layers, tol_layers=None)

    def calculate_loss(
        self, Xtrans: np.ndarray, X: np.ndarray, dX: Optional[np.ndarray] = None
    ) -> float:

        # calculate probability in latent space
        Z_logprob = stats.norm().logpdf(Xtrans)

        # calculate probability transform
        X_logprob = Z_logprob.sum(axis=1) + dX.sum(axis=1)

        # calculate negative look likelihood
        nll = np.mean(X_logprob)

        # save loss values
        self.loss_vals.append(nll)

        # return instance of
        return nll

    def check_tolerance(self, current_layer: int) -> bool:
        if current_layer >= self.n_layers:
            return False
        else:
            return True


class InformationLoss(RBIGLoss):
    def __init__(
        self,
        method: str = "histogram",
        p_value: float = 0.25,
        kwargs: Dict = {},
        tol_layers: Optional[int] = None,
    ) -> None:

        super().__init__(n_layers=None, tol_layers=tol_layers)
        self.method = method
        self.p_value = p_value
        self.kwargs = kwargs
        self.info_losses_ = list()

    def calculate_loss(
        self, Xtrans: np.ndarray, X: np.ndarray, dX: Optional[np.ndarray] = None
    ) -> float:

        delta_tc = information_reduction(
            X, Xtrans, method=self.method, p_value=self.p_value, kwargs=self.kwargs
        )
        # add loss values
        self.loss_vals.append(delta_tc)

        # return change in marginal entropy
        return delta_tc

    def check_tolerance(self, current_layer: int) -> bool:
        if current_layer <= self.tol_layers:
            return True
        else:
            # get the abs sum of the last n layers
            tol = np.abs(self.loss_vals[-self.tol_layers :]).sum()

            # calculate the changes
            if tol == 0:
                # no changes, don't use the last n layers
                self.loss_vals = self.loss_vals[: self.tol_layers]
                return False
            else:
                # continue
                return True
