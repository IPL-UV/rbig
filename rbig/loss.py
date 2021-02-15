from .information import InformationReduction
from typing import Optional
import numpy as np


class TCLoss(InformationReduction):
    def __init__(
        self,
        loss_tolerance: int = 60,
        reduction_tol: Optional[float] = None,
        p_value: float = 0.25,
        univariate_method="histogram",
        bins: str = "auto",
        correction: bool = True,
        k: int = 10,
        kernel: str = "gau",
        bw: str = "normal_reference",
        kwargs: Optional[dict] = None,
    ):
        super().__init__(
            reduction_tol=reduction_tol,
            p_value=p_value,
            univariate_method=univariate_method,
            bins=bins,
            correction=correction,
            k=k,
            kernel=kernel,
            bw=bw,
            kwargs=kwargs,
        )
        self.loss_tolerance = loss_tolerance
        self.losses = list()
        self.layers = 0

    def add_loss(self, X, Y):

        # increase iteration
        self.layers += 1

        # add loss
        loss = self.calculate_difference(X, Y)
        # print("Info Red.:", loss)

        if self.layers >= self.loss_tolerance:
            stop = self.check_iterations()
        else:
            stop = False
        # send exit if passed
        if stop == True:
            # print("Layers:", self.layers)
            # print("Tol Layers:", tol_layers)
            self.losses[:-50]
            return stop
        else:
            self.losses.append(loss)
            tol_layers = self.layers
            return stop

    def check_iterations(self):
        # print("Checking layers...")
        stop = False
        # print(self.losses[-1])
        # if the last x layers have a rough sum of zero
        last_layers = self.losses[-self.loss_tolerance :]
        # print("# last layers:", len(last_layers))
        tol = np.abs(last_layers).sum()
        # print("Info Red.:", tol)
        if tol == 0:
            stop = True
        return stop


class NegLogLikelihood:
    def __init__(self, loss_tolerance=10):
        self.loss_tolerance = loss_tolerance

    def check_loss(self):
        pass
