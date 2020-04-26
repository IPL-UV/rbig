import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.utils import check_array
from typing import Optional, Dict, Union
from rbig.utils import get_support_reference, get_domain_extension
from scipy.interpolate import interp1d


def kde_entropy_uni(X: np.ndarray, **kwargs) -> float:

    # check input array
    X = check_array(X, ensure_2d=True)

    # initialize KDE
    kde_density = sm.nonparametric.KDEUnivariate(X)

    kde_density.fit(**kwargs)

    return kde_density.entropy
