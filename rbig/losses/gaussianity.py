from typing import Dict, Optional

import numpy as np
from scipy.interpolate import interp1d
from sklearn.utils import check_array

from rbig.information.entropy import marginal_entropy
from rbig.losses import RBIGLoss


class NegEntropyLoss(RBIGLoss):
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
        self, Xtrans: np.ndarray, X: np.ndarray, X_slogdet: Optional[np.ndarray] = None
    ) -> float:

        # delta_tc = information_reduction(
        #     X, Xtrans, method=self.method, p_value=self.p_value, kwargs=self.kwargs
        # )
        delta_neg = diff_negentropy(Xtrans, X, X_slogdet)
        # add loss values
        self.loss_vals.append(delta_neg)

        # return change in marginal entropy
        return delta_neg

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


def diff_negentropy(Xtrans, X, X_slogdet) -> float:

    # calculate the difference in negentropy
    delta_neg = np.mean(
        0.5 * np.linalg.norm(Xtrans, 2) ** 2
        - X_slogdet
        - 0.5 * np.linalg.norm(X, 2) ** 2
    )

    # return change in marginal entropy
    return delta_neg
