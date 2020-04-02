from typing import Optional, Union, Callable
import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn.utils import check_random_state, check_array

BOUNDS_THRESHOLD = 1e-7
CLIP_MIN = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
CLIP_MAX = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))


class InverseGaussCDF:
    def __init__(self, alpha: float = 1e-10) -> None:
        self.alpha = alpha

    def transform(self, X: np.ndarray):
        X = check_array(X, ensure_2d=False, copy=True)
        X_trans = stats.norm.ppf(X)

        # clip infinite values
        X_trans = np.clip(X_trans, CLIP_MIN, CLIP_MAX)

        return X_trans

    def inverse_transform(self, X: np.ndarray):
        X = check_array(X, ensure_2d=False, copy=True)
        return stats.norm.cdf(X)

    def _transform(
        self,
        X: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        inverse: bool = False,
    ) -> np.ndarray:
        """Internal function that handles the boundary issues that can occur
        when doing the transformations. Wraps the transform and inverse
        transform functions.
        """

        # perform transformation
        X = f(X)

        if inverse == False:
            # get boundaries
            X = np.clip(X, CLIP_MIN, CLIP_MAX)

        return X

    def logpdf(self, X: np.ndarray):
        X = check_array(X, ensure_2d=False, copy=True)
        return -stats.norm.logpdf(stats.norm.ppf(X))

    def abs_det_jacobian(self, X: np.ndarray):
        X = check_array(X, ensure_2d=False, copy=True)
        return 1 / stats.norm.pdf(self.transform(X))

    def log_abs_det_jacobian(self, X: np.ndarray):
        X = check_array(X, ensure_2d=False, copy=True)
        return -stats.norm.logpdf(self.transform(X))

    def logprob(self, X: np.ndarray):
        X = check_array(X, ensure_2d=False, copy=True)
        return stats.norm.pdf(X) + self.log_abs_det_jacobian(X)

    def prob(self, X: np.ndarray):
        X = check_array(X, ensure_2d=False, copy=True)
        return stats.norm.pdf(X) * self.abs_det_jacobian(X)

    def sample(
        self, n_samples: int = 1, random_state: Optional[Union[RandomState, int]] = None
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

        U = rng.randn(n_samples)

        X = self.inverse_transform(U)
        return X


def main():
    from scipy import stats
    import matplotlib.pyplot as plt

    seed = 123
    n_samples = 1000

    # initialize data distribution
    data_dist = stats.uniform()

    # get some samples
    X_samples = data_dist.rvs(size=n_samples)
    # X_samples = np.array([1.0, 2.0, 1.0])

    # ========================
    # Transform Data Samples
    # ========================

    # transform data
    Xg = InverseGaussCDF().transform(X_samples)

    fig, ax = plt.subplots()
    ax.hist(Xg)
    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$\psi(u)$")
    plt.show()

    # =========================
    # Generate Gaussian Samples
    # =========================
    g_samples = stats.norm().rvs(size=1000)
    Xu_approx = InverseGaussCDF().inverse_transform(g_samples)

    fig, ax = plt.subplots()
    ax.hist(Xu_approx)
    ax.set_xlabel(r"$\psi^{-1}(g)$")
    ax.set_ylabel(r"u")
    plt.show()

    # ========================
    # Evaluate Probability
    # ========================
    print(X_samples[:10].shape)
    x_prob = InverseGaussCDF().prob(X_samples[:10])
    # x_prob *= stats.norm().rvs(X_samples[:10].shape[0])
    data_prob = data_dist.pdf(X_samples[:10])
    print(x_prob)
    print(data_prob)

    # # ========================
    # # Evaluate Log Probability
    # # ========================
    # x_score = hist_clf.score(X_samples)
    # data_score = -np.log(data_dist.pdf(X_samples)).mean()
    # print(x_score, data_score)

    return None


if __name__ == "__main__":
    main()
