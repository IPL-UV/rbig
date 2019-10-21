import numpy as np
from .base import ScoreMixin, DensityMixin
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from typing import Optional
from sklearn.utils import check_array, check_random_state
import warnings
from scipy.interpolate import interp1d
from sklearn.preprocessing import minmax_scale


BOUNDS_THRESHHOLD = 1e-7


class QuantileGaussian(BaseEstimator, TransformerMixin, ScoreMixin, DensityMixin):
    def __init__(
        self,
        n_quantiles: int = 1_000,
        bin_est: Optional[str] = None,
        subsample: int = 1_000,
        random_state: int = 123,
        support_ext: float = 10,
        interp: str = "linear",
    ) -> None:
        self.n_quantiles = n_quantiles
        self.bin_est = bin_est
        self.subsample = subsample
        self.random_state = random_state
        self.support_ext = support_ext
        self.interp = interp

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        # check n_quantiles is valid
        if self.n_quantiles <= 0:
            raise ValueError(
                f"Invalid n_quantiles: {self.n_quantiles}"
                "The number of quantiles must be at least 1."
            )

        # check n_subsamples is valid
        if self.subsample <= 0:
            raise ValueError(
                f"Invalid number of subsamples: {self.subsample}"
                "The number of quantiles must be at least 1."
            )

        # check that n_quantiles > subsamples
        if self.n_quantiles > self.subsample:
            raise ValueError(
                f"The n_quanties '{self.n_quantiles}' must not be greater than "
                f"subsamples: '{self.subsample}'."
            )

        # check inputs X
        X = check_array(X, copy=False)

        n_samples, self.d_dimensions = X.shape

        # check n_quantiles > n_samples
        if self.n_quantiles > n_samples:
            warnings.warn(
                f"n_quantiles '{self.n_quantiles}' is greater than total "
                f"number of samples '{n_samples}'. Setting to n_samples."
            )

        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        # initialize random state
        rng = check_random_state(self.random_state)

        self._fit(X, rng)

        return self

    def _fit(self, X, random_state):

        # dimensions of data
        n_samples, n_features = X.shape

        self.quantiles_ = []
        self.references_ = []
        for ifeature in X.T:
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(
                    n_samples, size=self.subsample, replace=False
                )

                ifeature = ifeature.take(subsample_idx, mode="clip")

            # create the quantiles reference for each feature
            if self.bin_est is None:
                references = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
            else:
                references = np.histogram_bin_edges(
                    ifeature, bins=self.bin_est, range=(0, 1)
                )
            # save quantiles
            quantiles = np.nanpercentile(ifeature, references * 100)
            self.quantiles_.append(quantiles)

            # save references
            self.references_.append(references)

        # extend support
        if self.support_ext != 0.0:

            for idx, ifeature in enumerate(X.T):

                # Extend the support
                new_reference, new_quantiles = self.extend_support(
                    self.references_[idx], self.quantiles_[idx], self.support_ext
                )

                self.quantiles_[idx] = new_quantiles
                self.references_[idx] = new_reference

        return self

    def extend_support(self, references, quantiles, support_extension):
        # Extend Support
        new_reference = np.hstack(
            [-support_extension / 100, references, 1 + support_extension / 100]
        )

        # extrapolate
        new_quantiles = interp1d(
            references, quantiles, kind=self.interp, fill_value="extrapolate", axis=0
        )(new_reference)

        # scale new
        reference_scale = minmax_scale(new_reference, axis=0)

        # Put back in original domain
        new_quantiles = interp1d(
            new_reference,
            new_quantiles,
            kind=self.interp,
            fill_value="extrapolate",
            axis=0,
        )(reference_scale)

        return reference_scale, new_quantiles

    def transform(self, X, y=None):
        X = check_array(X, copy=True)
        X = self._transform(X, inverse=False)
        return X

    def inverse_transform(self, X, y=None):
        X = check_array(X, copy=True)
        X = self._transform(X, inverse=True)
        return X

    def _transform(self, X, inverse=False):
        for feature_idx in range(X.shape[1]):
            X[:, feature_idx] = self.transform_col(
                X[:, feature_idx],
                self.quantiles_[feature_idx],
                self.references_[feature_idx],
                inverse=inverse,
            )
        return X

    def transform_col(self, X_col, quantiles, references, inverse):

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0  # CHECK: references[0]
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[1]

            # use CDF function
            # with np.errstate(invalid="ignore"):  # hide NAN comparison warnings
            X_col = stats.norm.cdf(X_col)

        # Find index for lower and higher bounds
        with np.errstate(invalid="ignore"):
            lower_bounds_idx = X_col - BOUNDS_THRESHHOLD < lower_bound_x
            upper_bounds_idx = X_col + BOUNDS_THRESHHOLD > upper_bound_x

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]

        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col[isfinite_mask] = 0.5 * (
                interp1d(
                    quantiles, references, kind=self.interp, fill_value="extrapolate"
                )(X_col_finite)
                - interp1d(
                    -quantiles[::-1],
                    -references[::-1],
                    kind=self.interp,
                    fill_value="extrapolate",
                    copy=True,
                )(-X_col_finite)
            )
            # X_col[isfinite_mask] = 0.5 * (
            #     np.interp(X_col_finite, quantiles, references)
            #     - np.interp(-X_col_finite, -quantiles[::-1], -references[::-1])
            # )
        else:
            X_col[isfinite_mask] = interp1d(
                references,
                quantiles,
                kind=self.interp,
                fill_value="extrapolate",
                copy=True,
            )(X_col_finite)
            # X_col[isfinite_mask] = np.interp(X_col_finite, references, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # Forward - Match Output distribution
        if not inverse:
            # with np.errstate(invalid="ignore"):
            X_col = stats.norm.ppf(X_col)
            # find the value to clip the data to avoid mapping to
            # infinity. Clip such that the inverse transform will be
            # consistent
            clip_min = stats.norm.ppf(BOUNDS_THRESHHOLD - np.spacing(1))
            clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHHOLD - np.spacing(1)))
            X_col = np.clip(X_col, clip_min, clip_max)
            # else output distribution is uniform and the ppf is the
            # identity function so we let X_col unchanged

        return X_col

    def score_samples(self, X, y=None):

        # transform data, invCDF(X)
        x_ = self.inverse_transform(X)

        # get - log probability, - log PDF( invCDF (x) )
        independent_log_prob = -stats.norm.logpdf(x_)

        # sum of log-likelihood is product of indepenent likelihoods
        return independent_log_prob.sum(axis=1)
