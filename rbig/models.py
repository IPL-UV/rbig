from collections import OrderedDict
from typing import Optional, Dict

# from .loss import TCLoss
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

from rbig.base import DensityTransformerMixin, ScoreMixin
from rbig.layers import RBIGParams
from rbig.losses import RBIGLoss, MaxLayers


class RBIGModel(BaseEstimator, DensityTransformerMixin, ScoreMixin):
    """A sequence of Gaussianization transforms.
    
    Parameters
    ----------
    """

    def __init__(
        self, flow: RBIGParams = RBIGParams(), loss: RBIGLoss = MaxLayers(),
    ) -> None:
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


# # TODO: Get ARGS functions
# class RBIGFlow(BaseEstimator, TransformerMixin, DensityMixin, ScoreMixin):
#     def __init__(
#         self,
#         n_layers=10,
#         rotation="ica",
#         n_quantiles=1_000,
#         subsample=2_000,
#         random_state=123,
#         support_ext=0.0,
#         interp="linear",
#     ):
#         self.n_layers = n_layers
#         self.rotation = rotation
#         self.n_quantiles = n_quantiles
#         self.subsample = subsample
#         self.support_ext = support_ext
#         self.random_state = random_state
#         self.interp = interp

#     def fit(self, X, y=None):

#         transforms = dict()

#         # fit layers sequentially
#         for ilayer in range(self.n_layers):

#             transforms[ilayer] = RBIGBlock(
#                 rotation=self.rotation,
#                 n_quantiles=self.n_quantiles,
#                 subsample=self.subsample,
#                 random_state=self.random_state,
#                 support_ext=self.support_ext,
#                 interp=self.interp,
#             )

#             transforms[ilayer].fit(X)

#             X = transforms[ilayer].transform(X)

#         # save the tranforms
#         self.transforms = transforms

#         return self

#     def transform(self, X, y=None):

#         # apply tranform sequentially
#         for ilayer in range(self.n_layers):

#             X = self.transforms[ilayer].transform(X)

#         return X

#     def inverse_transform(self, X, y=None):

#         # apply transform sequentially backwards
#         for ilayer in reversed(range(self.n_layers)):

#             X = self.transforms[ilayer].inverse_transform(X)

#         return X

#     def score_samples(self, X, y=None):

#         X_log_prob = np.ones(X.shape[0])
#         for ilayer in range(self.n_layers):

#             X_log_prob += self.transforms[ilayer].score_samples(X)

#         return X_log_prob


# class RBIG(RBIGFlow):
#     def __init__(
#         self,
#         # RBIG Params
#         max_layers=10,
#         rotation="ica",
#         n_quantiles=1_000,
#         subsample=2_000,
#         random_state=123,
#         support_ext=0.0,
#         interp="linear",
#         # Info Reduction Params
#         p_value=0.25,
#         univariate_method="histogram",
#         bins="auto",
#         correction=True,
#         k=10,
#         kernel="gau",
#         bw="normal_reference",
#         # Loss Function Params
#         loss_tolerance=50,
#     ):
#         super().__init__(
#             n_layers=max_layers,
#             rotation=rotation,
#             n_quantiles=n_quantiles,
#             subsample=subsample,
#             random_state=random_state,
#             support_ext=support_ext,
#             interp=interp,
#         )

#         # set stopping criterian
#         self.loss_tolerance = loss_tolerance

#         # initialize loss function
#         self.loss_function = TCLoss(
#             loss_tolerance=loss_tolerance,
#             p_value=p_value,
#             univariate_method=univariate_method,
#             bins=bins,
#             correction=correction,
#             k=k,
#             kernel=kernel,
#             bw=bw,
#         )

#     def fit(self, X, y=None):

#         transforms = list()
#         data = X.copy()

#         # fit layers sequentially
#         for ilayer in range(self.n_layers):
#             # original data

#             transforms.append(
#                 RBIGBlock(
#                     rotation=self.rotation,
#                     n_quantiles=self.n_quantiles,
#                     subsample=self.subsample,
#                     random_state=self.random_state,
#                     support_ext=self.support_ext,
#                     interp=self.interp,
#                 )
#             )
#             data_aug = transforms[-1].fit_transform(data)

#             # transforms[ilayer].transform(data)

#             stop = self._check_loss(data_aug, data)

#             if stop is True:

#                 transforms = transforms[:-50]
#                 self.n_layers = ilayer - 50 + 1
#                 assert self.n_layers == len(transforms)
#                 break
#             else:
#                 data = data_aug.copy()

#         # save the tranforms
#         self.transforms = transforms
#         self.info_reduction = self.loss_function.losses
#         return self

#     # Check loss, give stop criteria
#     def _check_loss(self, X, Y):

#         # update loss function
#         stop = self.loss_function.add_loss(X, Y)

#         return stop
