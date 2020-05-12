from typing import Callable, Optional, Union, Dict

import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn.utils import check_array, check_random_state
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

from rbig.transform.base import DensityMixin, BaseTransform
from rbig.transform.marginal import MarginalTransformation

# from rbig.density import Histogram
from rbig.transform.gauss_icdf import InverseGaussCDF

# from rbig.transform.histogram import ScipyHistogramUniformization
from rbig.utils import get_domain_extension, get_support_reference
import warnings
from sklearn.base import clone


class Gaussianization(BaseTransform, DensityMixin):
    """class to take a univariate Gaussianization
    
    This class composes a uniform transformer and a Inverse Gauss CDF
    transformation to make a Gaussianization transformation

    Parameters
    ----------
    uni_transformer : BaseTransform
        any base transformation that transforms data to a 
        uniform distribution.
    """

    def __init__(self, uni_transformer) -> None:
        self.uni_transformer = uni_transformer

    def fit(self, X: np.ndarray) -> None:
        """Fits the uniform transformation to the data
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            the input data to be transformed
        Returns
        -------
        self : instance of self
        """
        # check inputs
        X = check_array(X, ensure_2d=False, copy=True)

        # fit uniformization to data
        self.uni_transformer.fit(X)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Forward transformation of X.
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            the data to be transformed to the gaussian domain
        Returns
        -------
        Xtrans : np.ndarray, (n_samples, n_features)
            the transformed Gaussianized data
        """
        # check inputs
        X = check_array(X, ensure_2d=False, copy=True)

        # transform data to uniform domain
        X = self.uni_transformer.transform(X)

        # transform data to gaussian domain
        X = InverseGaussCDF().transform(X)

        # return gaussianized variable
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Performs the inverse transformation to original domain
        This transforms univariate Gaussian data to the original
        domain of the fitted transformation.
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Gaussian data
        Returns
        -------
        X : np.ndarray, (n_samples, n_features)
            the data in the original data domain.
        """
        # check inputs
        X = check_array(X, ensure_2d=False, copy=True)

        # transform data to uniform domain
        X = InverseGaussCDF().inverse_transform(X)

        # transform data to origin domain
        X = self.uni_transformer.inverse_transform(X)

        # return data from original domain
        return X

    def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Calculates the log-det-jacobian of the transformation
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            the data to be transformed
        
        Returns
        -------
        Xslogdet : np.ndarray, (n_samples, n_features)
            the log det-jacobian for each sample
        """
        # check array
        X = check_array(X.reshape(-1, 1), ensure_2d=True, copy=True)

        # find uniform probability
        u_log_prob = self.uni_transformer.log_abs_det_jacobian(X)

        # transform data into gaussian domain
        X_g = self.transform(X)

        # find gaussian probability
        g_log_prob = stats.norm().logpdf(X_g.squeeze())

        # return combined log det-jacobian
        return u_log_prob - g_log_prob

    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculates the log likelihood of the data.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            input data to calculate log-likelihood
        y : np.ndarray
            not used, for compatability only
        
        Returns
        -------
        ll : np.ndarray, (n_samples)
            log-likelhood of the data"""
        # transform data
        X_lprob = stats.norm().logpdf(self.transform(X)).squeeze()

        # independent feature transformation (sum the feature)
        X_ldj = self.log_abs_det_jacobian(X)

        return X_lprob + X_ldj


class MarginalGaussianization(MarginalTransformation):
    """Marginal Gaussianization transformation for any univariate transformer.
    This class wraps any univariate transformer to do a marginal Gaussianziation
    transformation. Includes a transform, inverse_transform and a
    log_det_jacobian method. It performs a feature-wise transformation on
    all fitted data. Includes a sampling method to con
    Parameters
    ----------
    Transformer : BaseTransform
        any base transform method
    Attributes
    ----------
    transforms_ : List[BaseTransform]
        a list of base transformers
    """

    def __init__(self, transformer: BaseTransform) -> None:
        super().__init__(Gaussianization(transformer))

    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculates the log likelihood of the data.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            input data to calculate log-likelihood
        y : np.ndarray
            not used, for compatability only
        
        Returns
        -------
        ll : np.ndarray, (n_samples)
            log-likelhood of the data"""
        # independent feature transformation (sum the feature)

        X_lprob = stats.norm().logpdf(self.transform(X)).squeeze()

        X_ldj = self.log_abs_det_jacobian(X)

        return (X_lprob + X_ldj).sum(-1)

    def sample(
        self, n_samples: int = 1, random_state: Optional[Union[RandomState, int]] = 123,
    ) -> np.ndarray:
        """Generate random samples from a uniform distribution.
        
        Parameters
        ----------
        n_samples : int, default=1
            The number of samples to generate. 
        
        random_state : int, RandomState,None, Optional, default=None
            The int to be used as a seed to generate the random 
            uniform samples.
        
        Returns
        -------
        X : np.ndarray, (n_samples, )
        """
        #
        rng = check_random_state(random_state)
        gauss_samples = rng.randn(n_samples, self.n_features_)

        X = self.inverse_transform(gauss_samples)
        return X
