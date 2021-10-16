import numpy as np
from scipy.stats import norm


class InverseGaussCDF:
    name: str = "invgausscdf"

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

        # create histogram object
        self.estimator = norm(loc=0, scale=1)

    def forward(self, X):

        Z = np.clip(X, self.eps, 1 - self.eps)

        Z = self.estimator.ppf(Z)

        return Z

    def inverse(self, Z):
        X = self.estimator.cdf(Z)

        return X

    def gradient(self, X):
        Z = self.forward(X)

        X_log_grad = -self.estimator.logpdf(Z)

        X_log_grad = X_log_grad.sum(axis=-1)

        return X_log_grad
