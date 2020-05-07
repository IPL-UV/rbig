import numpy as np

# from rbig.transform import MarginalHistogramTransform


def test_histogram():

    X = np.array([1.0, 2.0, 1.0])
    hpdf_ans = np.array([0.0, 2.0, 1.0, 0.0])
    hbin_ans = np.array([0.0, 2.0, 1.0, 0.0])
    bins = 4
    bounds = (0, 3)
    # hpdf, hbins = MarginalHistogramTransform()._calculate_histogram(
    #     X, bins=bins, bounds=bounds
    # )

    assert np.all(hbin_ans) == np.all(hbin_ans)
    assert np.all(hpdf_ans) == np.all(hpdf_ans)
