from typing import Dict, Optional

import numpy as np
from scipy import stats

from rbig.layers import RBIGLayer
from rbig.stopping import StoppingCriteria
from rbig.models.base import BaseModel
from sklearn.base import clone
from copy import deepcopy


class GaussianizationModel(BaseModel):
    """A Gaussianization model.

    Takes an RBIGLayer and an RBIGLoss (really )
    
    Parameters
    ----------
    flow : RBIGLayer
        an rbig layer (marginal gaussianization + rotation)
        see rbig.density.layers for details
    
    loss : StoppingCriteria
        an rbig stopping criteria which records the loss values
        at each iteration.
        see rbig.density.loss for details
    
    Attributes
    ----------

    flows_ : List[RBIGLayer]
        a list of all of the layers needed to Gaussianize the data.
    
    n_features_ : int
        dimensionality of the input data to be transformed
    
    losses_ : List[float]
        lost values for each layer

    Examples
    --------

    >>> # Step 1 - Pick a Uniformization Transformer
    >>> uniform_clf = HistogramUniformization(
        bins=100, support_extension=10, alpha=1e-4, n_quantiles=None
        )
    >>> # Step 2 - Initialize Marginal Gaussianization Transformer
    >>> mg_gaussianizer = MarginalGaussianization(uniform_clf)
    >>> # Step 3 - Pick Rotation transformer
    >>> orth_transform = OrthogonalTransform('pca')
    >>> # Step 4 - Initialize RBIG Block
    >>> rbig_block = RBIGLayer(mg_gaussianizer, orth_transform)
    >>> # Step 5 - Initialize loss function
    >>> rbig_loss = MaxLayers(n_layers=50)
    >>> # Step 6 - Intialize Gaussianization Model
    >>> rbig_model = GaussianizationModel(rbig_block, rbig_loss)
    >>> # fit model to data
    >>> Z, X_slogdzet = rbig_model.fit_transform(data, return_jacobian=True)
    """

    def __init__(self, flow: RBIGLayer, stopping_criteria: StoppingCriteria,) -> None:
        self.flow = flow
        self.stopping_criteria = stopping_criteria

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit Gaussianization model to data adhering to the stopping criteria
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            the data to be transformed to the Gaussian domain
        
        y : not used, for compatability only
        """
        # initialize transforms
        self.flows_ = list()
        self.n_features_ = X.shape[1]

        X_logdetjacobian = np.zeros(shape=X.shape)
        n_layers = 0
        add_layers = True

        while add_layers:
            # increase the the layer iterator
            n_layers += 1

            # initialize rbig block
            iflow = deepcopy(self.flow)

            # transform data
            Xtrans, dX = iflow.transform(X, y=None, return_jacobian=True)

            # regularize the jacobian
            dX[dX > 0.0] = 0.0

            X_logdetjacobian += dX

            # calculate loss (Xt, X, dX)
            _ = self.stopping_criteria.calculate_loss(Xtrans, X, X_logdetjacobian)

            # check stopping criteria
            add_layers = self.stopping_criteria.check_tolerance(n_layers)

            # save losses to class
            self.losses_ = self.stopping_criteria.losses_

            # append flows to flow list
            self.flows_.append(iflow)

            X = np.copy(Xtrans)

        # save number of layers
        self.n_layers_ = len(self.losses_)

        # reduce the number of layers based on # loss values
        if self.n_layers_ < len(self.flows_):
            self.flows_ = self.flows_[: self.n_layers_]

        return self

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, return_jacobian=False,
    ) -> np.ndarray:
        """Transforms data from original domain to Gaussian domain
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            the data to be transformed
        
        y : not used, for compatability only
        
        return_jacobian : bool, default=False
            option to return the jacobian of the whole transformation
        
        Returns
        -------
        Z : np.ndarray, (n_samples, n_features)
            the data that has been transformed to the gaussian domain
            
        X_slogdet : np.ndarray, (n_samples, n_features)
            the log det-jacobian of transformation of X
        """
        X_slogdet = np.zeros(shape=X.shape)
        for iflow in self.flows_:

            X, dX = iflow.transform(X, None, return_jacobian=True)

            X_slogdet += dX

        return X, X_slogdet

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        for iflow in self.flows_[::-1]:

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
