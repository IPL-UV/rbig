from typing import List
import numpy as np
from scipy.stats import norm


class Bijector:
    def forward(self, X):
        raise NotImplemented

    def inverse(self, X):
        raise NotImplemented

    def gradient(self, X):
        raise NotImplemented


class CompositeBijector:
    def __init__(self, bijectors: List[Bijector]):
        self.bijectors = bijectors

    def forward(self, X):

        Z = X.copy()
        for ibijector in self.bijectors:
            Z = ibijector.forward(Z)

        return Z

    def inverse(self, Z):

        X = Z.copy()
        for ibijector in reversed(self.bijectors):
            X = ibijector.inverse(X)

        return X

    def gradient(self, X):

        Z = X.copy()
        X_grad = np.zeros_like(X).sum(axis=-1)
        for ibijector in self.bijectors:
            X_grad += ibijector.gradient(Z)
            Z = ibijector.forward(Z)

        return X_grad


class FlowModel(CompositeBijector):
    def __init__(self, bijectors: List[Bijector], base_dist):
        self.bijectors = bijectors
        self.base_dist = base_dist

    def sample(self, n_samples: 10):
        pz_samples = self.base_dist.rvs(size=n_samples)

        X = self.inverse(pz_samples)
        return X

    def predict_proba(self, X):

        # forward tranformation
        Z = self.forward(X)

        pz = norm.logpdf(Z).sum(axis=-1)

        # gradient transformation
        X_ldj = self.gradient(X)

        return np.exp(pz + X_ldj)

    def score_samples(self, X):
        prob = self.predict_proba(X)
        return -np.mean(np.log(prob))
