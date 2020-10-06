import numpy as np

from rbig.density import (
    univariate_make_uniform,
    univariate_invert_uniformization,
    univariate_make_normal,
    univariate_invert_normalization,
)

rng = np.random.RandomState(123)


def test_univariate_uniformization():
    # create random data
    X = rng.randn(100)

    X_u, params = univariate_make_uniform(X, 0.1, 20)
    X_approx = univariate_invert_uniformization(X_u, params)

    np.testing.assert_array_almost_equal(X, X_approx)

    # create random data
    X = rng.randn(10_000)

    X_u, params = univariate_make_uniform(X, 0.1, 20)
    X_approx = univariate_invert_uniformization(X_u, params)

    np.testing.assert_array_almost_equal(X, X_approx)


def test_univariate_gaussianization():
    # create random data
    X = rng.randn(100)

    X_g, params = univariate_make_normal(X, 0.1, 20)
    X_approx = univariate_invert_normalization(X_g, params)

    np.testing.assert_array_almost_equal(X, X_approx)

    # create random data
    X = rng.randn(10_000)

    X_g, params = univariate_make_normal(X, 0.1, 20)
    X_approx = univariate_invert_normalization(X_g, params)

    np.testing.assert_array_almost_equal(X, X_approx)
