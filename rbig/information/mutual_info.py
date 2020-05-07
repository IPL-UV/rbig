from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array


import numpy as np
from typing import Optional
from rbig.information.entropy import univariate_entropy, multivariate_entropy
from sklearn.utils import check_array


def univariate_mutual_info(
    X: np.ndarray, Y: np.ndarray, method: str = "knn", **kwargs
) -> float:

    # check input array
    X = check_array(X, dtype=np.float, ensure_2d=True)
    Y = check_array(Y, dtype=np.float, ensure_2d=True)

    # H(X), entropy
    H_x = univariate_entropy(X, method=method, **kwargs)

    # H(Y), entropy
    H_y = univariate_entropy(Y, method=method, **kwargs)

    # H(X,Y), joint entropy
    H_xy = multivariate_entropy(np.hstack([X, Y]), method=method, **kwargs)

    return H_x + H_y - H_xy


class MutualInformation(BaseEstimator):
    def __init__(
        self, estimator: str = "knn", kwargs: Optional[dict] = None
    ) -> None:
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> BaseEstimator:
        """
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Data to be estimated.
        """
        X = check_array(X)
        if self.estimator == "knn":
            self.model = (
                KNNEstimator(**self.kwargs)
                if self.kwargs is not None
                else KNNEstimator()
            )
        elif self.estimator in ["rbig", "kde", "histogram"]:
            raise NotImplementedError(
                f"{self.estimator} is not implemented yet."
            )

        else:
            raise ValueError(f"Unrecognized estimator: {self.estimator}")
        if Y is not None:
            Y = check_array(Y)
            self._fit_mutual_info(X, Y)
        else:
            raise ValueError(f"X dims are less than 2. ")

        return self

    def _fit_multi_info(self, X: np.ndarray) -> float:

        # fit full
        model_full = self.model.fit(X)
        H_x = model_full.score(X)

        # fit marginals
        H_x_marg = 0
        for ifeature in X.T:

            model_marginal = self.model.fit(ifeature)
            H_x_marg += model_marginal.score(ifeature)

        # calcualte the multiinformation
        self.MI = H_x_marg - H_x

        return H_x_marg - H_x

    def _fit_mutual_info(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> None:

        # MI for X
        model_x = self.model.fit(X)
        H_x = model_x.score(X)
        print("Marginal:", H_x)

        # MI for Y
        model_y = self.model.fit(Y)
        H_y = model_y.score(Y)
        print("Marginal:", H_y)

        # MI for XY
        model_xy = self.model.fit(np.hstack([X, Y]))
        H_xy = model_xy.score(X)
        print("Full:", H_xy)

        # save the MI
        self.MI = H_x + H_y - H_xy

        return self.MI

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        return self.MI


class TotalCorrelation(BaseEstimator):
    def __init__(
        self, estimator: str = "knn", kwargs: Optional[dict] = None
    ) -> None:
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> BaseEstimator:
        """
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Data to be estimated.
        """
        X = check_array(X)

        if self.estimator == "knn":
            self.model = (
                KNNEstimator(**self.kwargs)
                if self.kwargs is not None
                else KNNEstimator()
            )
        elif self.estimator in ["rbig", "kde", "histogram"]:
            raise NotImplementedError(
                f"{self.estimator} is not implemented yet."
            )

        else:
            raise ValueError(f"Unrecognized estimator: {self.estimator}")

        if y is None and X.shape[1] > 1:

            self._fit_multi_info(X)
        else:
            raise ValueError(f"X dims are less than 2. ")

        return self

    def _fit_multi_info(self, X: np.ndarray) -> float:

        # fit full
        model_full = self.model.fit(X)
        H_x = model_full.score(X)
        print("Full:", H_x)
        # fit marginals
        H_x_marg = 0
        for ifeature in X.T:

            model_marginal = self.model.fit(ifeature[:, None])

            H_xi = model_marginal.score(ifeature[:, None])
            print("Marginal:", H_xi)
            H_x_marg += H_xi

        # calcualte the multiinformation
        self.MI = H_x_marg - H_x

        return self

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        return self.MI


# class RBIGMI(object):
#     """ Rotation-Based Iterative Gaussian-ization (RBIG) applied to two
#     multidimensional variables (RBIGMI). Applies the RBIG algorithm to
#     the two multidimensional variables independently, then applies another
#     RBIG algorithm on the two Gaussian-ized datasets.

#     Parameters
#     ----------
#     n_layers : int, optional (default 1000)
#         The number of steps to run the sequence of marginal gaussianization
#         and then rotation

#     rotation_type : {'PCA', 'random'}
#         The rotation applied to the marginally Gaussian-ized data at each iteration.
#         - 'pca'     : a principal components analysis rotation (PCA)
#         - 'random'  : random rotations
#         - 'ica'     : independent components analysis (ICA)

#     pdf_resolution : int, optional (default 1000)
#         The number of points at which to compute the gaussianized marginal pdfs.
#         The functions that map from original data to gaussianized data at each
#         iteration have to be stored so that we can invert them later - if working
#         with high-dimensional data consider reducing this resolution to shorten
#         computation time.

#     pdf_extension : int, optional (default 0.1)
#         The fraction by which to extend the support of the Gaussian-ized marginal
#         pdf compared to the empirical marginal PDF.

#     verbose : int, optional
#         If specified, report the RBIG iteration number every
#         progress_report_interval iterations.

#     zero_tolerance : int, optional (default=60)
#         The number of layers where the total correlation should not change
#         between RBIG iterations. If there is no zero_tolerance, then the
#         method will stop iterating regardless of how many the user sets as
#         the n_layers.

#     rotation_kwargs : dict, optional (default=None)
#         Any extra keyword arguments that you want to pass into the rotation
#         algorithms (i.e. ICA or PCA). See the respective algorithms on
#         scikit-learn for more details.

#     random_state : int, optional (default=None)
#         Control the seed for any randomization that occurs in this algorithm.

#     entropy_correction : bool, optional (default=True)
#         Implements the shannon-millow correction to the entropy algorithm

#     Attributes
#     ----------

#     rbig_model_X : RBIG() object
#         The RBIG model fitted

#     rbig_model_Y :


#     rbig_model_XY :


#     References
#     ----------
#     * Original Paper : Iterative Gaussianization: from ICA to Random Rotations
#         https://arxiv.org/abs/1602.00229

#     """

#     def __init__(
#         self,
#         n_layers=50,
#         rotation_type="PCA",
#         pdf_resolution=1000,
#         pdf_extension=None,
#         random_state=None,
#         verbose=None,
#         tolerance=None,
#         zero_tolerance=100,
#     ):
#         self.n_layers = n_layers
#         self.rotation_type = rotation_type
#         self.pdf_resolution = pdf_resolution
#         self.pdf_extension = pdf_extension
#         self.random_state = random_state
#         self.verbose = verbose
#         self.tolerance = tolerance
#         self.zero_tolerance = zero_tolerance

#     def fit(self, X, Y):
#         """Inputs for the RBIGMI algorithm.

#         Parameters
#         ----------
#         X : array, (n1_samples, d1_dimensions)

#         Y : array, (n2_samples, d2_dimensions)

#         Note: The number of dimensions and the number of samples
#         do not have to be the same.

#         """

#         # Initialize RBIG class I
#         self.rbig_model_X = RBIG(
#             n_layers=self.n_layers,
#             rotation_type=self.rotation_type,
#             pdf_resolution=self.pdf_resolution,
#             pdf_extension=self.pdf_extension,
#             verbose=self.verbose,
#             random_state=self.random_state,
#             zero_tolerance=self.zero_tolerance,
#             tolerance=self.tolerance,
#         )

#         # fit and transform model to the data
#         X_transformed = self.rbig_model_X.fit_transform(X)

#         # Initialize RBIG class II
#         self.rbig_model_Y = RBIG(
#             n_layers=self.n_layers,
#             rotation_type=self.rotation_type,
#             pdf_resolution=self.pdf_resolution,
#             pdf_extension=self.pdf_extension,
#             verbose=self.verbose,
#             random_state=self.random_state,
#             zero_tolerance=self.zero_tolerance,
#             tolerance=self.tolerance,
#         )

#         # fit model to the data
#         Y_transformed = self.rbig_model_Y.fit_transform(Y)

#         # Stack Data
#         XY_transformed = np.hstack([X_transformed, Y_transformed])

#         # Initialize RBIG class I & II
#         self.rbig_model_XY = RBIG(
#             n_layers=self.n_layers,
#             rotation_type=self.rotation_type,
#             random_state=self.random_state,
#             zero_tolerance=self.zero_tolerance,
#             tolerance=self.tolerance,
#         )

#         # Fit RBIG model to combined dataset
#         self.rbig_model_XY.fit(XY_transformed)

#         return self

#     def mutual_information(self):
#         """Given that the algorithm has been fitted to two datasets, this
#         returns the mutual information between the two multidimensional
#         datasets.

#         Returns
#         -------
#         mutual_info : float
#             The mutual information between the two multidimensional
#             variables.
#         """
#         return self.rbig_model_XY.residual_info.sum()
