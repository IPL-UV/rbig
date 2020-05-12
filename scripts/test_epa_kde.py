from scipy import stats
from pyprojroot import here
import sys

sys.path.append(str(here()))
import numpy as np
from rbig.density.kde import KDEEpanechnikov


def gen_uni_data(n_samples: int, a: float, seed: int = 123) -> np.ndarray:
    return stats.gamma(a=a).rvs(size=(n_samples, 1), random_state=seed)


# parameters
seed = 123
n_samples = 10_000
a = 4

# get some samples
X = gen_uni_data(n_samples, a, seed)
marg_kde_clf = KDEEpanechnikov(
    #     bw_method=bw_method,
    #     n_quantiles=n_quantiles,
    #     support_extension=support_extension,
    #     method=method,
    #     kernel=kernel,
    #     interp=interp,
    n_quantiles=1000,
    bw_method="scott",
    support_extension=10,
    kernel="epa",
)
marg_kde_clf.fit(X)
