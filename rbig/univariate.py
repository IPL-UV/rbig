import numpy as np
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from sklearn.utils import check_random_state
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import column_or_1d
import scipy.stats
import matplotlib.pyplot as plt
from rbig.utils import (
    check_bounds,
    make_interior,
    check_floating,
    make_interior_probability,
    make_positive,
    make_finite,
)

plt.style.use("ggplot")


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

    def get_support(self, X):
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
        pass

    def transform(self, X, y=None):
        # Check X
        # X = check_X(X)

        X = make_interior_probability(X)
        return scipy.stats.norm.ppf(X)

    def inverse_transform(self, X, y=None):
        return scipy.stats.norm.cdf(X)

    def logdetjacobian(self, X, y=None):
        pass


def get_data(seed=123, n_samples=1000, noise=0.25):
    rng = np.random.RandomState(seed=seed)

    X = np.abs(2 * rng.randn(n_samples, 1))
    Y = np.sin(X) + noise * rng.randn(n_samples, 1)

    return X, Y


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
