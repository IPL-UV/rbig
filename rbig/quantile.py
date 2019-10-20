import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from typing import Optional
from sklearn.utils import check_array, check_random_state
import warnings
from scipy.interpolate import interp1d


BOUNDS_THRESHHOLD = 1e-7


class QuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_quantiles: int = 1_000,
        subsample: int = 1_000,
        random_state: int = 123,
        domain_ext: float = 0.0,
    ) -> None:
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.random_state = random_state
        self.domain_ext = domain_ext

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
        X = check_array(X, copy=True)

        n_samples = X.shape[0]

        # check n_quantiles > n_samples
        if self.n_quantiles > n_samples:
            warnings.warn(
                f"n_quantiles '{self.n_quantiles}' is greater than total "
                f"number of samples '{n_samples}'. Setting to n_samples."
            )

        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        # create the quantiles reference
        self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)

        # initialize random state
        rng = check_random_state(self.random_state)
        self = self._fit(X, rng)
        return self

    def _fit(self, X, random_state):

        # dimensions of data
        n_samples, n_features = X.shape

        # reference quantiles
        references = self.references_ * 100

        self.quantiles_ = []
        for ifeature in X.T:
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(
                    n_samples, size=self.subsample, replace=False
                )

                ifeature = ifeature.take(subsample_idx, mode="clip")

            self.quantiles_.append(np.nanpercentile(ifeature, references))

        quantiles_ = np.transpose(self.quantiles_)

        # extend support
        if self.domain_ext != 0.0:

            references_, quantiles_ = self.extend_support(
                self.references_, quantiles_, self.domain_ext
            )
            self.references_ = references_
        self.quantiles_ = quantiles_
        return self

    def extend_support(self, references, quantiles, extension):

        # Extend support
        new_reference = np.hstack([extension, references, extension + 1])

        # Find new quantiles
        new_quantiles = interp1d(
            references, quantiles, kind="linear", fill_value="extrapolate", axis=0
        )(new_reference)

        return new_reference, new_quantiles

    def transform(self, X, y=None):
        X = self._transform(X, inverse=False)
        return X

    def inverse_transform(self, X, y=None):
        X = self._transform(X, inverse=True)
        return X

    def _transform(self, X, inverse=False):
        for feature_idx in range(X.shape[1]):
            X[:, feature_idx] = self.transform_col(
                X[:, feature_idx],
                self.quantiles_[:, feature_idx],
                self.references_,
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
            with np.errstate(invalid="ignore"):  # hide NAN comparison warnings
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
                    quantiles, references, kind="linear", fill_value="extrapolate"
                )(X_col_finite)
                - interp1d(
                    -quantiles[::-1],
                    -references[::-1],
                    kind="linear",
                    fill_value="extrapolate",
                )(-X_col_finite)
            )
        else:
            X_col[isfinite_mask] = interp1d(
                references, quantiles, kind="linear", fill_value="extrapolate"
            )(X_col_finite)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # Forward - Match Output distribution
        if not inverse:
            with np.errstate(invalid="ignore"):
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
