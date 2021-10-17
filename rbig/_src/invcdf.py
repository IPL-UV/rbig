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


class GaussCDF:
    name: str = "gausscdf"

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

        # create histogram object
        self.estimator = norm(loc=0, scale=1)

    def inverse(self, X):

        Z = np.clip(X, self.eps, 1 - self.eps)

        Z = self.estimator.ppf(Z)

        return Z

    def forward(self, Z):
        X = self.estimator.cdf(Z)

        return X

    def gradient(self, X):

        X_log_grad = self.estimator.logpdf(X)

        X_log_grad = X_log_grad.sum(axis=-1)

        return X_log_grad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def stable_sigmoid(x):
    return np.where(x < -9, np.exp(x), sigmoid(x))


def softplus(x):
    return np.log(np.exp(x) + 1)


def stable_softplus(x):
    return np.where(x < -9, np.log1p(np.exp(x)), softplus(x))


class Sigmoid:
    name: str = "sigmoid"

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def inverse(self, X):

        Z = np.clip(X, self.eps, 1 - self.eps)

        Z = np.log(X) - np.log1p(-X)

        return Z

    def forward(self, Z):
        X = stable_sigmoid(Z)

        return X

    def gradient(self, X):

        X_log_grad = -stable_softplus(-X) - stable_softplus(X)

        X_log_grad = X_log_grad.sum(axis=-1)

        return X_log_grad


class Logit:
    name: str = "logit"

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def forward(self, X):

        Z = np.clip(X, self.eps, 1 - self.eps)

        Z = np.log(X) - np.log1p(-X)

        return Z

    def inverse(self, Z):
        X = stable_sigmoid(Z)

        return X

    def gradient(self, X):

        X_log_grad = -stable_softplus(-X) - stable_softplus(X)

        X_log_grad = -X_log_grad.sum(axis=-1)

        return X_log_grad
