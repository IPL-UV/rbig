import numpy as np
from sklearn.utils import check_random_state


class ScoreMixin(object):
    """Mixin for :func:`score` that returns mean of :func:`score_samples`."""

    def score(self, X, y=None):
        """Return the mean log likelihood (or log(det(Jacobian))).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.
        y : None, default=None
            Not used but kept for compatibility.
        Returns
        -------
        log_likelihood : float
            Mean log likelihood data points in X.
        """
        return np.mean(self.score_samples(X, y))


class DensityMixin(object):
    """Mixin for :func:`sample` that returns the """

    def sample(self, n_samples=1, random_state=None):

        # random state
        rng = check_random_state(random_state)

        # get features
        d_dimensions = get_n_features(self)

        # Gaussian samples
        X_gauss = rng.randn(n_samples, d_dimensions)

        # Inverse transformation
        X = self.inverse_transform(X_gauss)
        return X


def get_n_features(model):

    # Check if RBIG block
    if hasattr(model, "d_dimensions"):
        return model.d_dimensions

    # Check if Model with RBIG block as attribute
    elif hasattr(model.transforms[0], "d_dimensions"):
        return model.transforms[0].d_dimensions

    else:
        raise ValueError("No model density (or block density) has been found.")
