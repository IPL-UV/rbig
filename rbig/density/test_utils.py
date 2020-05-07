import pytest
from rbig.density.utils import kde_cdf
from scipy import stats
import numpy as np


def test_kde_cdf():
    # generate some data
    np.random.seed(1)
    data = stats.norm().rvs(20)

    # estimate with scipy kde
    scipy_est = stats.gaussian_kde(data, bw_method="scott",)

    # =============
    # TEST SCALAR
    # =============

    x_cdf = scipy_est.integrate_box_1d(-np.inf, data[0])
    # estimate with my function
    my_cdf = kde_cdf(data, data[0], scipy_est.factor)

    np.testing.assert_almost_equal(x_cdf, my_cdf)

    # =============
    # TEST VECTOR
    # =============

    # estimate with scipy function
    x_cdf = np.vectorize(lambda x: scipy_est.integrate_box_1d(-np.inf, x))(
        data
    )

    # estimate with my function
    my_cdf = kde_cdf(data, data, scipy_est.factor)

    np.testing.assert_almost_equal(x_cdf, my_cdf)
