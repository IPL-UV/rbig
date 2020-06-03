from typing import Dict, Optional
from scipy.interpolate import interp1d
from sklearn.utils import check_array
import numpy as np
from rbig.information.entropy import marginal_entropy


def information_reduction(
    X: np.ndarray,
    Y: np.ndarray,
    method: str = "histogram",
    p_value: float = 0.25,
    kwargs: Optional[Dict] = {},
) -> float:
    """Calculates the information reduction between two datasets

    We find the sum of the marginal entropy measures for each dataset and 
    then we calculate the difference between the two.
    We assume that some transformation was applied to Y and it stems from X, 
        e.g. Y = AX.

    **Note**:
    * if X,Y is 1D then this is change in entropy
    * if X,Y is 2D then this is the change in mutual information
    * if X,Y is >=3D then this is the change in total correlation

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        the inputs of the array

    Y : np.narray, (n_samples, n_features)
        the transformed variable

    method : str, default='histogram'
        the univariate entropy estimator

    p_value : float, default=0.25
        a tolerance indicator for the change in entropy

    kwargs : Dict, default={}
        extra keyword arguments for the marginal estimators
        please see rbig.information.entropy.marginal_entropy for 
        more details

    Returns
    -------
    delta_I : float,
        the change in information (reduction of information)
        between the two inputs
    """
    X = check_array(X, ensure_2d=True)
    Y = check_array(Y, ensure_2d=True)

    # get tolerance dimensions
    n_samples, n_dimensions = X.shape
    tol = get_reduction_tolerance(n_samples)

    # calculate the marginal entropy

    H_x = marginal_entropy(X, method=method, **kwargs)
    H_y = marginal_entropy(Y, method=method, **kwargs)
    # print(H_x, H_y)

    # Find change in information content
    tc_term1 = np.sum(H_y) - np.sum(H_x)
    tc_term2 = np.sqrt(np.sum((H_x - H_y) ** 2))
    # print(tc_term1, tc_term2)

    if tc_term2 < np.sqrt(n_dimensions * p_value * tol ** 2) or tc_term1 < 0:
        delta_info = 0
    else:
        delta_info = tc_term1

    return delta_info


def get_reduction_tolerance(n_samples: int) -> float:
    """A tolerance indicator for the information reduction"""
    # xxx = np.logspace(2, 8, 7)
    # yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
    # tol = interp1d(xxx, yyy)(n_samples)
    xxx = np.logspace(2, 8, 7)
    yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
    tol_dimensions = np.interp(n_samples, xxx, yyy)
    return tol_dimensions
