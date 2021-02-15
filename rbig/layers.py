from sklearn.base import BaseEstimator, TransformerMixin
from .base import ScoreMixin, DensityMixin
from .quantile import QuantileGaussian
from .linear import OrthogonalTransform
import numpy as np
from sklearn.utils import check_array


class RBIGBlock(BaseEstimator, TransformerMixin, DensityMixin, ScoreMixin):
    def __init__(
        self,
        rotation="ica",
        n_quantiles=1_000,
        subsample=2_000,
        random_state=123,
        support_ext=0.0,
        interp="linear",
    ):
        self.rotation = rotation
        self.n_quantiles = n_quantiles
        self.bin_est = None
        self.subsample = subsample
        self.random_state = random_state
        self.support_ext = support_ext
        self.interp = interp

    def fit(self, X, y=None):

        X = check_array(X)

        self.d_dimensions = X.shape[1]

        # ========
        # Rotation
        # ========
        transform_R = OrthogonalTransform(rotation=self.rotation, random_state=123)

        data_R = transform_R.fit_transform(X)

        # ========================
        # Marginal Gaussianization
        # ========================

        transform_MG = QuantileGaussian(
            n_quantiles=self.n_quantiles,
            bin_est=self.bin_est,
            support_ext=self.support_ext,
            subsample=self.subsample,
            random_state=self.random_state,
            interp=self.interp,
        )

        transform_MG.fit(data_R)

        # save transforms
        self.transform_R = transform_R
        self.transform_MG = transform_MG

        return self

    def transform(self, X, y=None):

        # Rotation
        data_R = self.transform_R.transform(X)

        # Marginal Gaussianization
        data_MG = self.transform_MG.transform(data_R)

        return data_MG

    def inverse_transform(self, X, y=None):

        # Inverse Marginal Gaussianization
        data_iMG = self.transform_MG.inverse_transform(X)

        # Rotation Transpose
        data_Rt = self.transform_R.inverse_transform(data_iMG)

        return data_Rt

    def score_samples(self, X, y=None):

        # Rotation score (ignore because it's zero...)

        # MG score
        X_log_prob = self.transform_MG.score_samples(X)

        return X_log_prob
