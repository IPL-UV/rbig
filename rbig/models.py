from sklearn.base import BaseEstimator, TransformerMixin
from .base import ScoreMixin, DensityMixin
from .layers import RBIGBlock
import numpy as np


class RBIGFlow(BaseEstimator, TransformerMixin, DensityMixin, ScoreMixin):
    def __init__(self, n_layers=10, rotation="ica"):
        self.n_layers = n_layers
        self.rotation = rotation

    def fit(self, X, y=None):

        transforms = dict()

        # fit layers sequentially
        for ilayer in range(self.n_layers):

            transforms[ilayer] = RBIGBlock(rotation=self.rotation)

            transforms[ilayer].fit(X)

            X = transforms[ilayer].transform(X)

        # save the tranforms
        self.transforms = transforms

        return self

    def transform(self, X, y=None):

        # apply tranform sequentially
        for ilayer in range(self.n_layers):

            X = self.transforms[ilayer].transform(X)

        return X

    def inverse_transform(self, X, y=None):

        # apply transform sequentially backwards
        for ilayer in reversed(range(self.n_layers)):

            X = self.transforms[ilayer].inverse_transform(X)

        return X

    def score_samples(self, X, y=None):

        X_log_prob = np.ones(X.shape[0])
        for ilayer in range(self.n_layers):

            X_log_prob += self.transforms[ilayer].score_samples(X)

        return X_log_prob

