from typing import Union
import numpy as np
from scipy.stats import multivariate_normal
from rbig._src.total_corr import information_reduction
from rbig._src.training import train_rbig_info_loss
from rbig._src.uniform import MarginalHistogramUniformization
from rbig._src.invcdf import InverseGaussCDF
from rbig._src.rotation import PCARotation, RandomRotation
from rbig._src.base import FlowModel
from tqdm import trange
from sklearn.base import BaseEstimator, TransformerMixin


class RBIG(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        uniformizer: str = "hist",
        bins: Union[int, str] = "auto",
        alpha: float = 1e-10,
        bound_ext: float = 0.3,
        eps: float = 1e-10,
        rotation: str = "PCA",
        zero_tolerance: int = 60,
        max_layers: int = 1_000,
        max_iter: int = 10,
    ):
        self.uniformizer = uniformizer
        self.bins = bins
        self.alpha = alpha
        self.bound_ext = bound_ext
        self.eps = eps
        self.rotation = rotation
        self.zero_tolerance = zero_tolerance
        self.max_layers = max_layers
        self.max_iter = max_iter

    def fit(self, X, y=None):

        gf_model = train_rbig_info_loss(
            X=X,
            uniformizer=self.uniformizer,
            bins=self.bins,
            alpha=self.alpha,
            bound_ext=self.bound_ext,
            eps=self.eps,
            rotation=self.rotation,
            zero_tolerance=self.zero_tolerance,
            max_layers=self.max_layers,
            max_iter=self.max_iter,
        )
        self.gf_model = gf_model
        self.info_loss = gf_model.info_loss
        return self

    def transform(self, X, y=None):
        return self.gf_model.forward(X)

    def inverse_transform(self, X, y=None):
        return self.gf_model.inverse(X)

    def log_det_jacobian(self, X, y=None):
        return self.gf_model.gradient(X)

    def predict_proba(self, X, y=None):
        return self.gf_model.predict_proba(X)

    def sample(self, n_samples: int = 10):
        return self.gf_model.sample(n_samples)

    def total_correlation(self):
        return self.info_loss.sum()
