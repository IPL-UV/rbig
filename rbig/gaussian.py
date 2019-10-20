import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from scipy import stats
from rbig.utils import make_interior_probability
from .base import ScoreMixin
from sklearn.preprocessing import QuantileTransformer


class QuantileGaussian(BaseEstimator, TransformerMixin, ScoreMixin):
    def __init__(
        self,
        n_quantiles: int = 1_000,
        subsample: int = int(1e5),
        random_state: int = 123,
    ) -> None:
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None) -> None:

        # initialize quantile transformer
        transformer = QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution="normal",
            ignore_implicit_zeros=False,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        # fit to data
        transformer.fit(X)

        # save transformer to class
        self.transformer = transformer

        return self

    def transform(self, X, y=None):

        return self.transformer.transform(X)

    def inverse_transform(self, X, y=None):
        return self.transformer.inverse_transform(X)

    def score_samples(self, X, y=None):

        # transform data, invCDF(X)
        x_ = self.inverse_transform(X)

        # get - log probability, - log PDF( invCDF (x) )
        independent_log_prob = -stats.norm.logpdf(x_)

        # sum of log-likelihood is product of indepenent likelihoods
        return independent_log_prob.sum(axis=1)


class InverseCDF(BaseEstimator, TransformerMixin, ScoreMixin):
    """Independent inverse CDF transformer"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Applies the"""

        # Check array
        X = check_array(X)

        # TODO: Check X interval

        # TODO: Make interior probability
        X = make_interior_probability(X)

        # Apply inverse CDF of Gaussian
        return stats.norm.ppf(X)

    def inverse_transform(self, X, y=None):

        X = check_array(X)

        # TODO: check x interval

        # Apply Gaussian CDF
        return stats.norm.cdf(X)

    def score_samples(self, X, y=None):
        """Finds the log determinant of the data.
        
        Note: 
        d invCDF(X) = 1 / PDF( invCDF(X) ) 
                    = - log PDF( invCDF(X) )
                    = log ( J^-1 )
        because the Jacobian is diagonal
        """

        X = check_array(X)

        # TODO: Check interval

        # TODO: Make interior probability

        return stats.norm.logpdf(self.inverse_transform(X))

    def entropy(self, X):
        return 0.5 + 0.5 * np.log(2 * np.pi) + np.log(1.0)

