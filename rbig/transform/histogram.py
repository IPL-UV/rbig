from typing import Callable, Optional, Tuple, Union, Dict

import numpy as np
from numpy.random import RandomState
from scipy import stats
from rbig.information.histogram import ScipyHistogram

# Base classes
from sklearn.utils import check_array, check_random_state

from rbig.transform.base import DensityMixin, BaseTransform
from rbig.utils import (
    get_domain_extension,
    bin_estimation,
    make_interior,
    check_input_output_dims,
)

BOUNDS_THRESHOLD = 1e-7


class ScipyHistogramUniformization(BaseTransform, DensityMixin):
    def __init__(
        self,
        nbins: Optional[Union[int, str]] = "auto",
        alpha: float = 1e-5,
        support_extension: Union[float, int] = 10,
        kwargs: Dict = {},
    ) -> None:
        self.nbins = nbins
        self.alpha = alpha
        self.support_extension = support_extension
        self.kwargs = kwargs

    def fit(self, X: np.ndarray) -> None:
        """Finds an empirical distribution based on the
        histogram approximation.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
        
        Returns
        -------
        self : instance of self
        """
        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        self.n_samples_, self.n_features_ = X.shape

        # initialize histogram
        marginal_transforms = list()

        #
        # Loop through features
        for feature in X.T:

            support_bounds = get_domain_extension(
                feature, extension=self.support_extension
            )

            # calculate histogram
            hist, edges = np.histogram(
                feature, bins=self.nbins, range=support_bounds, **self.kwargs
            )

            # print("Hist:", np.min(hist), np.max(hist))

            # add some regularization
            hist = hist.astype(np.float64)
            hist += self.alpha
            # print("Hist:", hist.min(), hist.max())
            # normalize bins by bin_edges
            # edges = np.array(edges)
            # hist = hist / (edges[1:] - edges[:-1])
            # print("Hist:", hist.min(), hist.max())
            # print("Edges:", edges.min(), edges.max())

            # save marginal transform
            marginal_transforms.append(stats.rv_histogram((hist, edges)))

        # calculate the rv-based on the histogram
        self.marginal_transforms_ = marginal_transforms

        return self

    def _transform(self, X: np.ndarray, inverse: bool = False) -> np.ndarray:

        for feature_idx in range(X.shape[1]):

            # transform each column
            if inverse:
                X[:, feature_idx] = self._transform_feature(
                    X[:, feature_idx],
                    self.marginal_transforms_[feature_idx].ppf,
                    self.marginal_transforms_[feature_idx]._hcdf,
                    inverse=inverse,
                )
            else:
                X[:, feature_idx] = self._transform_feature(
                    X[:, feature_idx],
                    self.marginal_transforms_[feature_idx].cdf,
                    self.marginal_transforms_[feature_idx]._hcdf,
                    inverse=inverse,
                )
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Forward transform which is the CDF function of
        the samples

            z  = P(x)
            P() - CDF
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            incomining samples

        Returns
        -------
        X : np.ndarray
            transformed data
        """
        X = check_array(X, ensure_2d=True, copy=True)

        n_samples = X.shape[0]

        X = self._transform(X, inverse=False)

        check_input_output_dims(X, (n_samples, self.n_features_), "Histogram", "Foward")
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform which is the Inverse CDF function
        applied to the samples

            x = P^-1(z)
            P^-1() - Inverse CDF (PPF)

        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
        
        Returns
        -------
        X : np.ndarray, (n_samples, 1)
            Transformed data
        """
        X = check_array(X, ensure_2d=True, copy=True)

        n_samples = X.shape[0]

        X = self._transform(X, inverse=True)

        check_input_output_dims(
            X, (n_samples, self.n_features_), "Histogram", "Inverse"
        )

        return X

    def _transform_feature(
        self,
        X: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        quantiles: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """Internal function that handles the boundary issues that can occur
        when doing the transformations. Wraps the transform and inverse
        transform functions.
        """
        if inverse == True:
            # get boundaries
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
        else:
            # get boundaries
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1

        # find indices for upper and lower bounds
        lower_bounds_idx = X == lower_bound_x
        upper_bounds_idx = X == upper_bound_x

        # mask out infinite values
        isfinite_mask = ~np.isnan(X)
        X_col_finite = X[isfinite_mask]

        X[isfinite_mask] = f(X_col_finite)

        # set bounds
        X[upper_bounds_idx] = upper_bound_y
        X[lower_bounds_idx] = lower_bound_y

        return X

    def score_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Returns the log likelihood. It
        calculates the mean of the log probability.
        """
        log_prob = self.log_abs_det_jacobian(X, y).sum(axis=1).reshape(-1, 1)

        check_input_output_dims(log_prob, (X.shape[0], 1), "Histogram", "Log Prob")

        return log_prob

    def log_abs_det_jacobian(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> float:
        """Returns the log likelihood. It
        calculates the mean of the log probability.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
         incoming samples
        
        Returns
        -------
        X_jacobian : (n_samples, n_features),
            the mean of the log probability
        """
        X = check_array(X, ensure_2d=True, copy=True)

        n_samples = X.shape[0]

        log_scores = list()

        for feature_idx in range(X.shape[1]):

            # clip extreme values that are input
            X[:, feature_idx] = make_interior(
                X[:, feature_idx],
                bounds=(
                    self.marginal_transforms_[feature_idx].a,
                    self.marginal_transforms_[feature_idx].b,
                ),
            )

            # transform each column
            iscore = self.marginal_transforms_[feature_idx].logpdf(X[:, feature_idx])
            log_scores.append(iscore)

        # print(f"hstack: {np.hstack(log_scores).shape}")
        # print(f"vstack: {np.vstack(log_scores).shape}")
        log_scores = np.vstack(log_scores).T

        assert log_scores.shape == (
            n_samples,
            self.n_features_,
        ), f"Histogram: Jacobian lost dims, {X.shape}"

        check_input_output_dims(
            log_scores, (X.shape[0], self.n_features_), "Histogram", "Jacobian"
        )

        return log_scores

    def _clip_infs(self, f: Callable) -> Tuple[float, float]:
        clip_min = f(BOUNDS_THRESHOLD - np.spacing(1))
        clip_max = f(1 - (BOUNDS_THRESHOLD - np.spacing(1)))

        return clip_min, clip_max

    def sample(
        self, n_samples: int = 1, random_state: Optional[Union[RandomState, int]] = None
    ) -> np.ndarray:
        """Generate random samples from this.
        
        Parameters
        ----------
        n_samples : int, default=1
            The number of samples to generate. 
        
        random_state : int, RandomState,None, Optional, default=None
            The int to be used as a seed to generate the random 
            uniform samples.
        
        Returns
        -------
        X : np.ndarray, (n_samples, )
        """
        #
        rng = check_random_state(random_state)

        U = rng.rand(n_samples, self.n_features_)

        X = self.inverse_transform(U)
        return X


class HistogramUniformization(BaseTransform, DensityMixin):
    def __init__(
        self, bins_est="sqrt", pdf_extension=0.1, cdf_precision=1000, alpha=1e-6
    ):
        self.bins_est = bins_est
        self.pdf_extension = pdf_extension
        self.cdf_precision = cdf_precision
        self.alpha = alpha

    def fit(self, X):
        nbins = bin_estimation(X, rule=self.bins_est)

        # Get Histogram (Histogram PDF, Histogtam bins)
        hpdf, hbins = np.histogram(X, bins=nbins)
        hpdf = np.array(hpdf, dtype=float)
        hpdf += self.alpha
        assert len(hpdf) == nbins

        # CDF
        hcdf = np.cumsum(hpdf)
        hcdf = (1 - 1 / X.shape[0]) * hcdf / X.shape[0]

        # Get Bin Widths
        hbin_widths = hbins[1:] - hbins[:-1]
        hbin_centers = 0.5 * (hbins[:-1] + hbins[1:])
        assert len(hbin_widths) == nbins

        # Get Bin StepSizde
        bin_step_size = hbins[2] - hbins[1]

        # Normalize hpdf
        hpdf = hpdf / float(np.sum(hpdf * hbin_widths))

        # Handle Tails of PDF
        hpdf = np.hstack([0.0, hpdf, 0.0])
        hpdf_support = np.hstack(
            [
                hbin_centers[0] - bin_step_size,
                hbin_centers,
                hbin_centers[-1] + bin_step_size,
            ]
        )

        # hcdf = np.hstack([0.0, hcdf])
        domain_extension = 0.1
        precision = 1000
        old_support = np.array([X.min(), X.max()])

        support_extension = (domain_extension / 100) * abs(np.max(X) - np.min(X))
        # old_support = np.array([X.min(), X.max()])

        old_support = np.array([X.min(), X.max()])
        new_support = (1 + domain_extension) * (old_support - X.mean()) + X.mean()

        new_support = np.array(
            [X.min() - support_extension, X.max() + support_extension]
        )

        # Define New HPDF support
        hpdf_support_ext = np.hstack(
            [
                X.min() - support_extension,
                X.min(),
                hbin_centers + bin_step_size,
                X.max() + support_extension + bin_step_size,
            ]
        )

        # Define New HCDF
        hcdf_ext = np.hstack([0.0, 1.0 / X.shape[0], hcdf, 1.0])

        # Define New support for hcdf
        hcdf_support = np.linspace(hpdf_support_ext[0], hpdf_support_ext[-1], precision)
        self.hcdf_support = hcdf_support

        # Interpolate HCDF with new precision
        hcdf_ext = np.interp(hcdf_support, hpdf_support_ext, hcdf_ext)
        self.hcdf = hcdf_ext
        self.hpdf = hpdf
        self.hpdf_support = hpdf_support

        return self

    def transform(self, X):
        return np.interp(X, self.hcdf_support, self.hcdf)

    def inverse_transform(self, X):
        return np.interp(X, self.hcdf, self.hcdf_support)

    def log_abs_det_jacobian(self, X):
        return np.log(np.interp(X, self.hpdf_support, self.hpdf_support))

    def score_samples(self, X):
        raise NotImplementedError
