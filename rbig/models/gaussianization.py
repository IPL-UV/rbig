from typing import Dict, Optional

import numpy as np
from scipy import stats

from rbig.layers import RBIGParams
from rbig.losses import RBIGLoss
from rbig.models.base import BaseModel


class GaussianizationModel(BaseModel):
    """A sequence of Gaussianization transforms.
    
    Parameters
    ----------
    """

    def __init__(self, flow: RBIGParams, loss: RBIGLoss,) -> None:
        self.flow = flow
        self.loss = loss

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        # initialize transforms
        self.flows = list()
        self.n_features_ = X.shape[1]

        X_logdetjacobian = np.zeros(shape=X.shape)
        n_layers = 0
        add_layers = True

        while add_layers:
            # increase the the layer iterator
            n_layers += 1

            # initialize rbig block
            iflow = self.flow.fit_data(X)

            # transform data
            Xtrans, dX = iflow.transform(X, y=None, return_jacobian=True)

            # regularize the jacobian
            dX[dX > 0.0] = 0.0

            X_logdetjacobian += dX

            # calculate loss (Xt, X, dX)
            _ = self.loss.calculate_loss(Xtrans, X, X_logdetjacobian)

            # check stopping criteria
            add_layers = self.loss.check_tolerance(n_layers)

            # save losses to class
            self.losses_ = self.loss.loss_vals

            # append flows to flow list
            self.flows.append(iflow)

            X = np.copy(Xtrans)

        # save number of layers
        self.n_layers_ = len(self.losses_)

        # reduce the number of layers based on # loss values
        if self.n_layers_ < len(self.flows):
            self.flows = self.flows[: self.n_layers_]

        return self

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, return_jacobian=False
    ) -> np.ndarray:

        X_slogdet = np.zeros(shape=X.shape)
        for iflow in self.flows:

            X, dX = iflow.transform(X, None, return_jacobian=True)

            X_slogdet += dX

        return X, X_slogdet

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        for iflow in self.flows[::-1]:

            X = iflow.inverse_transform(X)
        return X

    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        # transform

        Z, dX = self.transform(X)
        prior_logprob = stats.norm().logpdf(Z).sum(axis=1)

        # get rid of extreme values
        dX[dX > 0.0] = 0.0
        return prior_logprob + dX.sum(axis=1)

    def sample(self, n_samples: int = 1) -> np.ndarray:

        Z = stats.norm().rvs((n_samples, self.n_features_))
        return self.inverse_transform(Z)
