import numpy as np
import statsmodels.api as sm
from sklearn.utils import check_array


def kde_entropy_uni(X: np.ndarray, **kwargs) -> float:

    # check input array
    X = check_array(X, ensure_2d=True)

    # initialize KDE
    kde_density = sm.nonparametric.KDEUnivariate(X)

    kde_density.fit(**kwargs)

    return kde_density.entropy
