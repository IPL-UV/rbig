import numpy as np


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
