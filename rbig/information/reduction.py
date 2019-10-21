from .entropy import MarginalEntropy
from typing import Optional
import numpy as np
from sklearn.utils import check_array
from scipy.interpolate import interp1d


class InformationReduction(MarginalEntropy):
    """Calculates the information loss (or the information reduction) 
    between two datasets X and Y. We find the sum of the marginal entropy measures
    We assume that some transformation
    was applied to Y and it stems from X, e.g. Y = AX.

    **Note**: 
    * if X,Y is 1D then this is change in entropy
    * if X,Y is 2D then this is the change in mutual information
    * if X,Y is >=3D then this is the change in total correlation
    
    Parameters
    ----------
    tol : float, Optional, default=None
        The tolerance we allow it to be considered a change in total
        correlation of the data
        **Don't mess with this parameter unless you know what
        you're doing.**
    """

    def __init__(
        self,
        reduction_tol: Optional[float] = None,
        p_value: float = 0.25,
        univariate_method="histogram",
        bins: str = "auto",
        correction: bool = True,
        k: int = 10,
        kernel: str = "gau",
        bw: str = "normal_reference",
        kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(
            univariate_method=univariate_method,
            bins=bins,
            correction=correction,
            k=k,
            kernel=kernel,
            bw=bw,
            kwargs=kwargs,
        )
        self.reduction_tol = reduction_tol
        self.p_value = p_value

    def calculate_difference(self, X, Y):
        X = check_array(X)
        Y = check_array(Y)

        # get tolerance dimensions
        n_samples, n_dimensions = X.shape
        self.tol = self._get_tolerance(self.reduction_tol, n_samples)

        # calculate the marginal entropy

        H_x = self.entropy(X)
        H_y = self.entropy(Y)

        # Find change in information content
        I = np.sum(H_x) - np.sum(H_y)
        II = np.sqrt(np.sum(H_x - H_y) ** 2)

        if II < np.sqrt(n_dimensions * self.p_value * self.tol ** 2) or I < 0:
            I_final = 0
        else:
            I_final = I
        # print(I, II, I_final)
        return I_final

    def _get_tolerance(self, tol, n_samples):

        if tol is None or 0:
            xxx = np.logspace(2, 8, 7)
            yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
            tol = interp1d(xxx, yyy)(n_samples)
        return tol

