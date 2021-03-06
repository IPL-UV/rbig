import numpy as np
from rbig.information.entropy import entropy_marginal


def information_reduction(x_data, y_data, tol_dimensions=None, correction=True):
    """Computes the multi-information (total correlation) reduction after a linear
    transformation
    
            Y = X * W
            II = I(X) - I(Y)
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data before the transformation, where n_samples is the number of samples
        and n_features is the number of features
    
    Y : array-like, shape (n_samples, n_features)
        Data after the transformation, where n_samples is the number of samples
        and n_features is the number of features
        
    tol_dimensions : float, optional
        Tolerance on the minimum multi-information difference
        
    Returns
    -------
    II : float
        The multi-information
        
    Information
    -----------
    Author: Valero Laparra
            Juan Emmanuel Johnson
    """
    # check that number of samples and dimensions are equal
    err_msg = "Number of samples for x and y should be equal."
    np.testing.assert_equal(x_data.shape, y_data.shape, err_msg=err_msg)

    n_samples, n_dimensions = x_data.shape

    # minimum multi-information heuristic
    if tol_dimensions is None or 0:
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        tol_dimensions = np.interp(n_samples, xxx, yyy)

    # preallocate data
    hx = np.zeros(n_dimensions)
    hy = np.zeros(n_dimensions)

    # calculate the marginal entropy
    hx = entropy_marginal(x_data, correction=correction)
    hy = entropy_marginal(y_data, correction=correction)

    # Information content
    I = np.sum(hy) - np.sum(hx)
    II = np.sqrt(np.sum((hy - hx) ** 2))

    p = 0.25
    if II < np.sqrt(n_dimensions * p * tol_dimensions ** 2) or I < 0:
        I = 0

    return I
