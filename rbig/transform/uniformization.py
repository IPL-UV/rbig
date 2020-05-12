from typing import Callable, Optional, Tuple, Union, Dict

import numpy as np
from numpy.random import RandomState
from scipy import stats
from rbig.density.histogram import ScipyHistogram, QuantileHistogram
from rbig.density.kde import KDEFFT, KDEScipy, KDESklearn, KDEEpanechnikov

# Base classes
from sklearn.utils import check_array, check_random_state
from sklearn.base import clone

from rbig.transform.base import DensityMixin, BaseTransform
from rbig.transform.marginal import MarginalTransformation
import matplotlib.pyplot as plt

# from rbig.density.base
from rbig.utils import (
    get_domain_extension,
    bin_estimation,
    make_interior,
    check_input_output_dims,
)

BOUNDS_THRESHOLD = 1e-7


class HistogramUniformization(BaseTransform, DensityMixin):
    """Univariate histogram uniformization.
    This class performs uniformization using histograms. This class is
    a light wrapper around the scipy histogram function with the option
    to use quantiles to enhance the amount of support for the CDF
    function.
    
    Parameters
    ----------
    bins : Union[int, str], default='auto'
        the number of bins to estimated the histogram function.
        see `np.histogram` for more options.
    alpha : float, default=1e-5
        the amount of regularization to add to the estimated PDFs
        so that there are no zero probabilities.
    prob_tol : float, default=1e-7
        this controls how much we clip any data in the outside bounds
    n_quantiles : Optional[int], default=None
        the number of quantiles (support) to use for the CDF function.
        If None, then the number of support for the CDF/PPF function
        is equal to the number of bins used for the histogram.
    support_extension : int, default=10
        the amount to extend the support of the fitted data X. Affects
        the PDF,CDF, and PPF functions. Extending the support will allow
        more data to be interpolated.
        * this will widen the range for this histogram function.
        * this will widen the range for the quantile estimation
    subsample : int, default=10_000
        the number of subsamples to use for estimating the quantiles.
        Only used if the `n_quantiles` param is not None.
    random_state : int, default=123
        the random seed used for selecting the subset of points used
        to estimate the `n_quantiles`. Not used if `n_quantiles` is
        None.
    kwargs : Dict[str,Any], default={}
        any extra kwargs to be passed into the `np.histogram` estimator.
    
    Attributes
    ----------
    estimator_ : PDFEstimator
        a fitted histogram estimator with a PDFEstimator base class.
        This method will have pdf, logpdf, cdf and ppf functions.
    """

    def __init__(
        self,
        bins: Union[int, str] = "auto",
        alpha: float = 1e-5,
        prob_tol: float = 1e-7,
        n_quantiles: Optional[int] = None,
        support_extension: Union[float, int] = 10,
        subsample: Optional[int] = 10_000,
        random_state: int = 123,
        kwargs: Dict = {},
    ) -> None:
        self.bins = bins
        self.alpha = alpha
        self.prob_tol = prob_tol
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension
        self.subsample = subsample
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Used to fit the histogram estimator to data.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            the data to be fitted
        y : np.ndarray, (n_samples)
            not used, for compatibility only
        Returns
        -------
        self : instance of self
        """
        # fit superclass
        if self.n_quantiles is not None:
            self.estimator_ = QuantileHistogram(
                bins=self.bins,
                alpha=self.alpha,
                n_quantiles=self.n_quantiles,
                subsample=self.subsample,
                random_state=self.random_state,
                support_extension=self.support_extension,
                kwargs=self.kwargs,
            ).fit(X)
        else:
            self.estimator_ = ScipyHistogram(
                bins=self.bins,
                alpha=self.alpha,
                prob_tol=self.prob_tol,
                support_extension=self.support_extension,
                kwargs=self.kwargs,
            ).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Forward transform which is the CDF function of
        the samples

        $z=F_\theta(x)$
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
            incomining samples

        Returns
        -------
        X_trans : np.ndarray
            transformed data
        """
        X = check_array(X, ensure_2d=False, copy=True)

        return self.estimator_.cdf(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform which is the Inverse CDF function
        applied to the samples

            x = P^-1(z)
            P^-1() - Inverse CDF (PPF)

        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
        
        Returns
        -------
        X : np.ndarray, (n_samples, 1)
            Transformed data
        """
        X = check_array(X, ensure_2d=False, copy=True)

        return self.estimator_.ppf(X)

    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Returns the log likelihood. It
        calculates the mean of the log probability.
        """
        X = check_array(X, ensure_2d=False, copy=True)
        log_prob = self.log_abs_det_jacobian(X)
        # return log_prob
        return log_prob

    def log_abs_det_jacobian(self, X: np.ndarray) -> float:
        """Returns the log-determinant-jacobian of transformation
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, 1)
         incoming samples
        
        Returns
        -------
        X_jacobian : (n_samples, n_features),
            the mean of the log probability
        """
        X = check_array(X, ensure_2d=False, copy=True)

        return self.estimator_.logpdf(X)

    def sample(
        self,
        n_samples: int = 1,
        random_state: Optional[Union[RandomState, int]] = None,
    ) -> np.ndarray:
        """Generate random samples from this.
        
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

        U = rng.rand(n_samples)

        X = self.inverse_transform(U)
        return X


class KDEUniformization(HistogramUniformization):
    """Univariate kde uniformization.
    This class performs uniformization using kernel density functions. 
    This class is a light wrapper around a few KDE models:

    * scipy - exact kde function
    * statsmodels - fft kde function (fast)
    * sklearn - kde with k-nearest neighbors
    * kdepy - with the option for Epanechnikov kernel (fastest)
    
    to use quantiles to enhance the amount of support for the CDF
    function.
    Once the KDE parameters are fitted, most methods use an interpolation
    function for the CDF and the PPF.
    
    Parameters
    ----------
    method : str, default='fft'
        the method to use for the kde estimation

        * 'fft' (default) - uses the fast statsmodels implementation
        * 'exact' - uses the exact scipy implemention
        * 'epa' - uses the fast/scalable KDEpy implementation
        * 'knn' - uses the sklearn implementation

    bw_method : str, default='scott'
        bandwidth for the kernel estimation. almost all methods have
        scott so that's the default. For the 'epa' kernel and suspected
        multimodel distributions, then it's recommended to use the
        'ISJ' method. For more options, see the individual implementations.
    n_quantiles : Optional[int], default=None
        the number of quantiles (support) to use for the PDF/CDF/PPF functions.
    support_extension : int, default=10
        the amount to extend the support of the fitted data X. Affects
        the PDF,CDF, and PPF functions. Extending the support will allow
        more data to be interpolated.
        * this will widen the range for this kde grid function.
        * this will widen the range for the quantile estimation
    kernel : str, default='gaussian'
        the standard kernel is the Gaussian
    interp : bool, default=True
        decides whether to interpolate the pdf estimation
        only affects the "exact", "knn" "fft" methods
    random_state : int, default=123
        the random seed used for selecting the subset of points used
        to estimate the `n_quantiles`. Not used if `n_quantiles` is
        None.
    kwargs : Dict[str,Any], default={}
        any extra kwargs to be passed into the `np.histogram` estimator.
    
    Attributes
    ----------
    estimator_ : PDFEstimator
        a fitted histogram estimator with a PDFEstimator base class.
        This method will have pdf, logpdf, cdf and ppf functions.
    """

    def __init__(
        self,
        method: str = "fft",
        bw_method: str = "scott",
        n_quantiles: int = 1_000,
        support_extension: Union[float, int] = 10,
        interp: bool = True,
        algorithm: str = "kd_tree",
        kernel: str = "gaussian",
        metric: str = "euclidean",
        norm: int = 2,
        kwargs: Dict = {},
    ) -> None:
        self.bw_method = bw_method
        self.n_quantiles = n_quantiles
        self.support_extension = support_extension
        self.interp = interp
        self.method = method
        self.algorithm = algorithm
        self.metric = metric
        self.kernel = kernel
        self.norm = norm
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        # fit superclass
        if self.method in ["knn"]:
            self.estimator_ = KDESklearn(
                n_quantiles=self.n_quantiles,
                algorithm=self.algorithm,
                interp=self.interp,
                kernel=self.kernel,
                metric=self.metric,
                support_extension=self.support_extension,
            ).fit(X)
        elif self.method in ["fft"]:
            self.estimator_ = KDEFFT(
                bw_method=self.bw_method,
                interp=self.interp,
                n_quantiles=self.n_quantiles,
                support_extension=self.support_extension,
            ).fit(X)
        elif self.method in ["exact"]:
            self.estimator_ = KDEScipy(
                bw_method=self.bw_method,
                interp=self.interp,
                n_quantiles=self.n_quantiles,
                support_extension=self.support_extension,
            ).fit(X)
        elif self.method in ["epa"]:
            self.estimator_ = KDEEpanechnikov(
                kernel=self.kernel,
                bw_method=self.bw_method,
                n_quantiles=self.n_quantiles,
                support_extension=self.support_extension,
                norm=self.norm,
            ).fit(X)
        else:
            raise ValueError(f"Unrecognized method: {self.method}")
        return self


class MarginalUniformization(MarginalTransformation):
    """Marginal Uniformization transformation for any univariate transformer.
    This class wraps any univariate transformer to do a marginal Uniformization
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
    n_features_ : int
        number of features in the marginal transformation
    """

    def __init__(self, transformer: BaseTransform) -> None:
        super().__init__(transformer)

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
        return self.log_abs_det_jacobian(X).sum(-1)

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
        uniform_samples = rng.rand(n_samples, self.n_features_)

        X = self.inverse_transform(uniform_samples)
        return X
