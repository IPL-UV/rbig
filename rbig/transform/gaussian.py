from typing import Dict, Tuple, Callable
import numpy as np
from scipy import stats
from rbig.transform.uniform import uniform_transform_params
from rbig.utils import (
    make_interior_uniform_probability,
    make_interior,
    make_interior_log_prob,
)

BOUNDS_THRESHOLD = 1e-7


def gaussian_fit_transform(
    X: np.ndarray, method: str = "custom", params: Dict = {},
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

    # make uniform data
    params = uniform_transform_params(X, method=method, params=params)

    # do the forward transformations
    X_gauss = _gaussian_transform(X, params, inverse=False)

    return X_gauss, params


def _gaussian_transform(
    X: np.ndarray, params: Dict, inverse: bool = True
) -> np.ndarray:
    # get boundaries
    if not inverse:
        lower_bound_x = params["x_bounds"][0]
        upper_bound_x = params["x_bounds"][1]
        lower_bound_y = 0
        upper_bound_y = 1
    else:
        lower_bound_x = 0
        upper_bound_x = 1
        lower_bound_y = params["x_bounds"][0]
        upper_bound_y = params["x_bounds"][1]

        with np.errstate(invalid="ignore"):  # hide NAN comparison warnings
            X = stats.norm.cdf(X)

    # find indices of upper and lower indices that violate boundaries
    with np.errstate(invalid="ignore"):
        lower_bounds_idx = X - BOUNDS_THRESHOLD < lower_bound_x
        upper_bounds_idx = X + BOUNDS_THRESHOLD > upper_bound_x

    # get mask of values in domain
    isfinite_mask = ~np.isnan(X)

    X_finite = X[isfinite_mask]

    if not inverse:
        X[isfinite_mask] = params["uniform_cdf"](X_finite)
    else:
        X[isfinite_mask] = params["uniform_ppf"](X_finite)

    X[upper_bounds_idx] = upper_bound_y
    X[lower_bounds_idx] = lower_bound_y

    # match output distribution
    if not inverse:
        with np.errstate(invalid="ignore"):
            X = stats.norm.ppf(X)

            # find values to avoid mapping to infinity
            clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
            clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))

            #
            X = np.clip(X, clip_min, clip_max)

    return X


def gaussian_fit_transform_jacobian(
    X: np.ndarray, method: str = "custom", params: Dict = {},
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:

    # find the parameters
    X_gauss, params = gaussian_fit_transform(X, method=method, params=params)

    # Log PDF for Gaussianized data
    Xg_ldj = stats.norm.ppf(X_gauss)

    # LogPDF of uniformized data

    Xu_ldj = params["uniform_logpdf"](X)

    # log pdf of full transformation
    X_ldj = Xu_ldj - Xg_ldj

    return X_gauss, X_ldj, params


def gaussian_transform(
    X: np.ndarray, params: Dict = {}
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

    # # make uniform data
    # X_uniform = params["uniform_cdf"](X)

    # X_uniform = make_interior_uniform_probability(X_uniform, eps=BOUNDS_THRESHOLD)

    # try:
    #     assert X_uniform.min() > BOUNDS_THRESHOLD
    #     assert X_uniform.max() < 1 - BOUNDS_THRESHOLD
    # except:
    #     msg = f"\nInput Bounds: {X_uniform.min()}, {X_uniform.max()}"
    #     raise ValueError(f"Incorrect boundaries...:\n" + msg)
    # inverse gaussian cdf
    # return stats.norm.ppf(X_uniform)
    return _gaussian_transform(X, params, inverse=False)


def gaussian_transform_jacobian(
    X: np.ndarray, params: Dict = {}
) -> Tuple[np.ndarray, np.ndarray]:

    X = make_interior(X, bounds=params["x_bounds"], eps=1e-7)

    # LogPDF of uniformized data
    Xu_ldj = params["uniform_logpdf"](X.copy())
    Xu_ldj = make_interior_log_prob(Xu_ldj, BOUNDS_THRESHOLD)

    # fit the parameters
    X_gauss = _gaussian_transform(X.copy(), params=params, inverse=False)

    # Log PDF for Gaussianized data
    Xg_ldj = np.log(stats.norm.pdf(X_gauss))
    # print(np.percentile(Xg_ldj, [0.5, 0.95]))
    Xg_ldj = make_interior_log_prob(Xg_ldj, BOUNDS_THRESHOLD)

    # log pdf of full transformation
    X_ldj = Xu_ldj - Xg_ldj

    msg = f"\nInput Bounds: {X.min():.2f}, {X.max():.2f}"
    msg += f"\nParam Bounds: {params['x_bounds'][0]}, {params['x_bounds'][1]}"
    msg += f"\nUniform Dist: {Xu_ldj.min()}, {Xu_ldj.max()}"
    msg += f"\nX_g: {X_gauss.min()}, {X_gauss.max()}"
    msg += f"\nGaussian Dist: {Xg_ldj.min()}, {Xg_ldj.max()}"
    msg += f"\nFull Dist: {X_ldj.min()}, {X_ldj.max()}"

    # try:
    assert not np.isnan(X_ldj.min()) and not np.isinf(X_ldj.min()), msg
    assert not np.isnan(X_ldj.max()) and not np.isinf(X_ldj.max()), msg
    # except:

    #     raise ValueError(msg)
    # print(np.percentile(X_ldj, [0, 5, 50, 95, 100]))
    return X_gauss, X_ldj


def gaussian_inverse_transform(
    X: np.ndarray, params: Dict[str, Callable]
) -> np.ndarray:
    """Inverts the marginal gaussianization
    params are used from the uniform transform function.
    Parameters
    ----------
    X : np.ndarray, (n_samples)
        inputs to be transformed
    params : Dict[str, Callable]
        params to be used to do the transformation. Must have a parameter
        'uniform_cdf' to do the inverse transformation
    Returns
    -------
    Xtrans : np.ndarray, (n_samples)
        data in the original domain.
    """
    # transform data to the uniform domain
    # X = stats.norm.cdf(X)

    # transform data back to original domain
    # return params["uniform_ppf"](X)
    return _gaussian_transform(X, params=params, inverse=True)
