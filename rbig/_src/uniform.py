from typing import Union, NamedTuple
from scipy.stats import rv_histogram
import numpy as np
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

class MarginalHistogramUniformization:
    name: str= "marghistuni"
 
    
    def __init__(self, X: np.ndarray, bins: Union[int,str]="auto", alpha: float=1e-10, bound_ext: float=0.1):
        
        estimators = []
        
        
        for iX in X.T:
            diff = iX.max() - iX.min()
            lower_bound = iX.min() - bound_ext * diff
            upper_bound = iX.max() + bound_ext * diff

            # create histogram 
            hist = np.histogram(iX, bins=bins, range=(lower_bound, upper_bound))

            # create histogram object
            i_estimator = rv_histogram(hist)

            # add some regularization
            i_estimator._hpdf += alpha
            
            estimators.append(i_estimator)
            
        self.estimators = estimators
        
    def forward(self, X):
        
        Z = np.zeros_like(X)
        for idim, iX in enumerate(X.T):
            Z[:, idim] = self.estimators[idim].cdf(iX)
        
        return Z
    
    def inverse(self, Z):
        
        X = np.zeros_like(Z)
        
        for idim, iZ in enumerate(Z.T):
            
            X[:, idim] = self.estimators[idim].ppf(iZ)
        
        return X
    
    def gradient(self, X):
        
        X_grad = np.zeros_like(X)
        
        for idim, iX in enumerate(X.T):
            X_grad[:, idim] = self.estimators[idim].logpdf(iX)
        X_grad = X_grad.sum(axis=-1)
        return X_grad




class KDEParams(NamedTuple):
    support : np.ndarray
    pdf_est : np.ndarray
    cdf_est : np.ndarray

class MarginalKDEUniformization:
    name: str= "marghistuni"
 
    
    def __init__(self, X: np.ndarray, grid_size: int=50, n_quantiles:int=50, bound_ext: float=0.1, fft: bool=True):
        
        estimators = []
        
        # estimate bandwidth
        bw = np.power(X.shape[0], -1 / (X.shape[1] + 4.0))
        
        
        for iX in X.T:
            

            # create histogram 
            estimator = sm.nonparametric.KDEUnivariate(iX.squeeze())
            
            estimator.fit(
                    kernel="gau", bw=bw, fft=fft, gridsize=grid_size,
                )
            
            # estimate support
            diff = iX.max() - iX.min()
            lower_bound = iX.min() - bound_ext * diff
            upper_bound = iX.max() + bound_ext * diff
            support = np.linspace(lower_bound, upper_bound, n_quantiles)
            
            # estimate empirical pdf from data
            hpdf = estimator.evaluate(support)
            
            # estimate empirical cdf from data
            hcdf = ECDF(iX)(support)
            
            kde_params = KDEParams(support=support, pdf_est=np.log(hpdf), cdf_est=hcdf)
            estimators.append(kde_params)
            
        self.estimators = estimators
        
    def forward(self, X):
        
        Z = np.zeros_like(X)
        for idim, iX in enumerate(X.T):
            iparams = self.estimators[idim]
            Z[:, idim] = np.interp(iX, xp=iparams.support, fp=iparams.cdf_est)
        
        return Z
    
    def inverse(self, Z):
        
        X = np.zeros_like(Z)
        
        for idim, iZ in enumerate(Z.T):
            
            iparams = self.estimators[idim]
            X[:, idim] = np.interp(iZ, xp=iparams.cdf_est, fp=iparams.support)
        
        return X
    
    def gradient(self, X):
        
        X_grad = np.zeros_like(X)
        
        for idim, iX in enumerate(X.T):
            
            iparams = self.estimators[idim]
            X_grad[:, idim] = np.interp(iX, xp=iparams.support, fp=iparams.pdf_est)
            
        X_grad = X_grad.sum(axis=-1)
        return X_grad