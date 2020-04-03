import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from scipy.stats import entropy as sci_entropy
from scipy.stats import norm, ortho_group
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_random_state
from sklearn.utils.validation import column_or_1d

from rbig.utils import (bin_estimation, check_bounds, check_floating,
                        make_finite, make_interior, make_interior_probability,
                        make_positive)

plt.style.use("ggplot")


class MarginalUniformization(BaseEstimator, TransformerMixin):
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

    def logdetjacobian(self, X):
        pass

    def logpdf(self, X):
        return np.log(self.hpdf[1:-1])

    def entropy(self, X, correction=True):
        return entropy(self.hpdf, correction=correction)


class HistogramUnivariateDensity(BaseEstimator, TransformerMixin):
    def __init__(self, bins="auto", bounds=0.1, alpha=1e-6):
        self.bins = bins
        self.bounds = bounds
        self.alpha = alpha

    def fit(self, X, y=None, histogram_params=None):

        # Check X
        X = self.check_X(X)

        # Check Bounds
        bounds = check_bounds(X, self.bounds)
        print(f"Bounds: {bounds}")

        # fit numpy histogram
        hist, bin_edges = np.histogram(X, bins=self.bins, range=bounds)
        print(f"BinEdges: {bin_edges.max(), bin_edges.min()}")
        print(f"hist: {hist.max(), hist.min()}")

        # ========================
        # Regularization
        # ========================
        hist = np.array(hist, dtype=float)
        hist += self.alpha

        # Convert to scipy RV
        self.rv = scipy.stats.rv_histogram((hist, bin_edges))
        return self

    def transform(self, X):
        X = self.check_X(X)
        return self.rv.cdf(X.ravel()).reshape((-1, 1))

    def inverse_transform(self, X):
        X = self.check_X(X)
        return self.rv.ppf(X.ravel()).reshape((-1, 1))

    def logdetjacobian(self, X):
        X = self.check_X(X)
        return self.rv.logpdf(X.ravel()).reshape((-1, 1))

    def get_support(self, X=None):
        # Assumes density is univariate
        return np.array([[self.rv.a, self.rv.b]])

    def sample(self, n_samples=1, random_state=None):

        rng = check_random_state(random_state)
        return np.array(self.rv.rvs(size=n_samples, random_state=rng)).reshape(
            (n_samples, 1)
        )

    def entropy(self, X):
        X = self.check_X(X)
        return self.rv.entropy(X.ravel()).reshape((-1, 1))

    def check_X(self, X, inverse=False):

        bounds = check_bounds(X, self.bounds)

        if np.any(X <= bounds[0]) or np.any(X >= bounds[1]):
            warnings.warn(
                BoundaryWarning(
                    "Input to random variable function has at least one value outside of bounds "
                    "but all inputs should be in (bounds[0], bounds[1]) exclusinve. Bounding "
                    "values away from bounds[0] and bounds[1]."
                )
            )

            X = make_interior(X, bounds)
        return X


class InverseCDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Check X
        # X = check_X(X)

        X = make_interior_probability(X)
        return scipy.stats.norm.ppf(X)

    def inverse_transform(self, X, y=None):
        return scipy.stats.norm.cdf(X)

    def logpdf(self, X, y=None):
        return np.log(self.hpdf)

    def entropy(self, X):
        return 0.5 + 0.5 * np.log(2 * np.pi) + np.log(1.0)


def get_data(seed=123, n_samples=1000, noise=0.25):
    rng = np.random.RandomState(seed=seed)

    X = np.abs(2 * rng.randn(n_samples, 1))
    Y = np.sin(X) + noise * rng.randn(n_samples, 1)

    return X, Y


def entropy(hist_counts, correction=None):

    # MLE Estimator with Miller-Maddow Correction
    if not (correction is None):
        correction = 0.5 * (np.sum(hist_counts > 0) - 1) / hist_counts.sum()
    else:
        correction = 0.0

    # Plut in estimator of entropy with correction
    return sci_entropy(hist_counts, base=2) + correction


def example():
    X, _ = get_data()

    seed = 123
    rng = np.random.RandomState(seed=seed)

    num_samples = 1000
    noise = 0.25

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(X, marker=".", linestyle="")
    ax.set_title("Original Data")
    plt.show()

    # Parameters
    alpha = 1e-6  # regularization parameter
    bounds = 0.01  # percentage extension
    bins = "auto"  # number of bins or bins estimator parameter

    histogram_params = None

    # Initialize Univariate Normalization
    uni_model = HistogramUnivariateDensity(alpha=alpha, bounds=bounds, bins=bins)

    # transform
    uni_model.fit(X)

    # Initialize Model
    uni_model = HistogramUnivariateDensity(alpha=alpha, bounds=bounds, bins=bins)

    # Fit Model to Data
    uni_model.fit(X)

    # Transform Data
    X_uni_approx = uni_model.transform(X)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(X_uni_approx, marker=".", linestyle="")
    ax.set_title("Data in Uniform Domain")
    plt.show()

    # Inverse Transform Data
    X_uni = rng.uniform(0, 1, (1000, 1))
    X_approx = uni_model.inverse_transform(X_uni)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(X_approx, marker=".", linestyle="")
    ax.set_title("Data Generated data domain")
    plt.show()

    # Generate samples from domain
    X_uni_samples = uni_model.sample(n_samples=1000)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(X_uni_samples, marker=".", linestyle="")
    ax.set_title("Samples from Uniform Domain")
    plt.show()

    # Calculate Log Jacobian
    mll = uni_model.logdetjacobian(X).sum()
    print(mll)

    # Calculate Entropy
    ent = uni_model.entropy(X).sum()
    print(ent)
    return None


def main():
    pass


if __name__ == "__main__":
    example()
    pass
