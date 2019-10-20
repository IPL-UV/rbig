from sklearn.base import BaseEstimator, TransformerMixin
from .base import ScoreMixin


class MarginalTransformation(BaseEstimator, TransformerMixin, ScoreMixin):
    """This performs a univariate transformation on a datasets.
    
    Assuming that the data is independent across features, this
    applies a transformation on each feature independently. The inverse 
    transformation is the marginal cdf applied to each of the features
    independently and the inverse transformation is the marginal inverse
    cdf applied to the features independently.
    """

    def __init__(self,):
        pass

    def fit(self, X, y=None):
        pass

    def _fit(self, X):
        pass

    def transform(self, X, y=None):
        pass

    def inverse_transform(self, X, y=None):
        pass

    def score_samples(self, X, y=None):
        pass

    def entropy(self, X, y=None):
        pass

    def _marginal_cdf(self, X, y=None):
        pass

    def _marginal_inverse_cdf(self, X, y=None):
        pass

