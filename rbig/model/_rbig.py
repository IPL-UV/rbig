from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.utils import check_random_state, check_array
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from scipy.stats import norm, uniform, ortho_group, entropy as sci_entropy
from scipy.interpolate import interp1d
from rbig.information.total_corr import information_reduction
from rbig.information.entropy import entropy_marginal
from rbig.utils import make_cdf_monotonic
from sklearn.decomposition import PCA
import sys
import logging
from rbig.transform.gaussian import (
    gaussian_transform,
    gaussian_fit_transform,
    gaussian_inverse_transform,
    gaussian_transform_jacobian,
)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s: %(levelname)s: %(message)s",
)
logger = logging.getLogger()
# logger.setLevel(logging.INFO)


class RBIG(BaseEstimator, TransformerMixin):
    """ Rotation-Based Iterative Gaussian-ization (RBIG). This algorithm transforms
    any multidimensional data to a Gaussian. It also provides a sampling mechanism
    whereby you can provide multidimensional gaussian data and it will generate
    multidimensional data in the original domain. You can calculate the probabilities
    as well as have access to a few information theoretic measures like total
    correlation and entropy.

    Parameters
    ----------
    n_layers : int, optional (default 1000)
        The number of steps to run the sequence of marginal gaussianization
        and then rotation
    rotation_type : {'PCA', 'random'}
        The rotation applied to the marginally Gaussian-ized data at each iteration.
        - 'pca'     : a principal components analysis rotation (PCA)
        - 'random'  : random rotations
        - 'ica'     : independent components analysis (ICA)
    pdf_resolution : int, optional (default 1000)
        The number of points at which to compute the gaussianized marginal pdfs.
        The functions that map from original data to gaussianized data at each
        iteration have to be stored so that we can invert them later - if working
        with high-dimensional data consider reducing this resolution to shorten
        computation time.
    method : str, default='custom'
    pdf_extension : int, optional (default 0.1)
        The fraction by which to extend the support of the Gaussian-ized marginal
        pdf compared to the empirical marginal PDF.
    verbose : int, optional
        If specified, report the RBIG iteration number every
        progress_report_interval iterations.
    zero_tolerance : int, optional (default=60)
        The number of layers where the total correlation should not change
        between RBIG iterations. If there is no zero_tolerance, then the
        method will stop iterating regardless of how many the user sets as
        the n_layers.
    rotation_kwargs : dict, optional (default=None)
        Any extra keyword arguments that you want to pass into the rotation
        algorithms (i.e. ICA or PCA). See the respective algorithms on 
        scikit-learn for more details.
    random_state : int, optional (default=None)
        Control the seed for any randomization that occurs in this algorithm.
    entropy_correction : bool, optional (default=True)
        Implements the shannon-millow correction to the entropy algorithm

    Attributes
    ----------
    gauss_data : array, (n_samples x d_dimensions)
        The gaussianized data after the RBIG transformation

    residual_info : array, (n_layers)
        The cumulative amount of information between layers. It should exhibit
        a curve with a plateau to indicate convergence.

    rotation_matrix = dict, (n_layers)
        A rotation matrix that was calculated and saved for each layer.

    gauss_params = dict, (n_layers)
        The cdf and pdf for the gaussianization parameters used for each layer.

    References
    ----------
    * Original Paper : Iterative Gaussianization: from ICA to Random Rotations
        https://arxiv.org/abs/1602.00229

    * Original MATLAB Implementation
        http://isp.uv.es/rbig.html

    * Original Python Implementation
        https://github.com/spencerkent/pyRBIG
    """

    def __init__(
        self,
        n_layers: int = 1000,
        rotation_type: str = "PCA",
        method: str = "custom",
        pdf_resolution: int = 1000,
        pdf_extension: int = 10,
        random_state: Optional[int] = None,
        verbose: Optional[int] = None,
        tolerance: int = None,
        zero_tolerance: int = 60,
        entropy_correction: bool = True,
        rotation_kwargs: Dict = {},
        base="gauss",
        n_quantiles: int = 1_000,
    ) -> None:
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.method = method
        self.pdf_resolution = pdf_resolution
        self.pdf_extension = pdf_extension
        self.random_state = random_state
        self.verbose = verbose
        self.tolerance = tolerance
        self.zero_tolerance = zero_tolerance
        self.entropy_correction = entropy_correction
        self.rotation_kwargs = rotation_kwargs
        self.base = base
        self.n_quantiles = n_quantiles

    def fit(self, X):
        """ Fit the model with X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, ensure_2d=True)
        self._fit(X)
        return self

    def _fit(self, data):
        """ Fit the model with data.
        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        data = check_array(data, ensure_2d=True)

        if self.pdf_extension is None:
            self.pdf_extension = 10

        if self.pdf_resolution is None:
            self.pdf_resolution = 2 * np.round(np.sqrt(data.shape[0]))
        self.X_fit_ = data
        gauss_data = np.copy(data)

        n_samples, n_dimensions = np.shape(data)

        if self.zero_tolerance is None:
            self.zero_tolerance = self.n_layers + 1

        if self.tolerance is None:
            self.tolerance = self._get_information_tolerance(n_samples)

        logging.debug("Data (shape): {}".format(np.shape(gauss_data)))

        # Initialize stopping criteria (residual information)
        self.residual_info = list()
        self.gauss_params = list()
        self.rotation_matrix = list()

        # Loop through the layers
        logging.debug("Running: Looping through the layers...")

        for layer in range(self.n_layers):

            if self.verbose > 2:
                print("Completed {} iterations of RBIG.".format(layer + 1))

            # ------------------
            # Gaussian(-ization)
            # ------------------
            layer_params = list()

            for idim in range(n_dimensions):

                gauss_data[:, idim], params = gaussian_fit_transform(
                    gauss_data[:, idim],
                    method=self.method,
                    params={
                        "support_extension": self.pdf_extension,
                        "n_quantiles": self.n_quantiles,
                    },
                )

                # gauss_data[:, idim], params = self.univariate_make_normal(
                #     gauss_data[:, idim], self.pdf_extension, self.pdf_resolution
                # )
                if self.verbose > 2:
                    logging.info(
                        f"Gauss Data (After Marginal): {gauss_data.min()}, {gauss_data.max()}"
                    )

                # append the parameters
                layer_params.append(params)

            self.gauss_params.append(layer_params)
            gauss_data_prerotation = gauss_data.copy()
            if self.verbose > 2:
                logging.info(
                    f"Gauss Data (prerotation): {gauss_data.min()}, {gauss_data.max()}"
                )

            # --------
            # Rotation
            # --------
            if self.rotation_type == "random":

                rand_ortho_matrix = ortho_group.rvs(n_dimensions)
                gauss_data = np.dot(gauss_data, rand_ortho_matrix)
                self.rotation_matrix.append(rand_ortho_matrix)

            elif self.rotation_type.lower() == "pca":

                # Initialize PCA model
                pca_model = PCA(random_state=self.random_state, **self.rotation_kwargs)

                logging.debug("Size of gauss_data: {}".format(gauss_data.shape))
                gauss_data = pca_model.fit_transform(gauss_data)
                self.rotation_matrix.append(pca_model.components_.T)

            else:
                raise ValueError(
                    f"Rotation type '{self.rotation_type}' not recognized."
                )

            # --------------------------------
            # Information Reduction
            # --------------------------------
            self.residual_info.append(
                information_reduction(
                    gauss_data, gauss_data_prerotation, self.tolerance
                )
            )

            # --------------------------------
            # Stopping Criteria
            # --------------------------------
            if self._stopping_criteria(layer):
                break
            else:
                pass
        self.residual_info = np.array(self.residual_info)
        self.gauss_data = gauss_data
        self.mutual_information = np.sum(self.residual_info)
        self.n_layers = len(self.gauss_params)

        return self

    def _stopping_criteria(self, layer):
        """Stopping criteria for the the RBIG algorithm.
        
        Parameter
        ---------
        layer : int

        Returns
        -------
        verdict = 
        
        """
        stop_ = False

        if layer > self.zero_tolerance:
            aux_residual = np.array(self.residual_info)

            if np.abs(aux_residual[-self.zero_tolerance :]).sum() == 0:
                logging.debug("Done! aux: {}".format(aux_residual))

                # delete the last 50 layers for saved parameters
                self.rotation_matrix = self.rotation_matrix[:-50]
                self.gauss_params = self.gauss_params[:-50]

                stop_ = True
            else:
                stop_ = False

        return stop_

    def transform(self, X):
        """Complete transformation of X given the learned Gaussianization parameters.
        This assumes that the data follows a similar distribution as the data that
        was original used to fit the RBIG Gaussian-ization parameters.
        Parameters
        ----------
        X : array, (n_samples, n_dimensions)
            The data to be transformed (Gaussianized)
        Returns
        -------
        X_transformed : array, (n_samples, n_dimensions)
            The new transformed data in the Gaussian domain
        """
        X = check_array(X, ensure_2d=True, copy=True)

        for igauss, irotation in zip(self.gauss_params, self.rotation_matrix):

            # ----------------------------
            # Marginal Gaussianization
            # ----------------------------

            for idim in range(X.shape[1]):

                X[:, idim] = gaussian_transform(X[:, idim], igauss[idim])

            # ----------------------
            # Rotation
            # ----------------------
            X = np.dot(X, irotation)

        return X

    def inverse_transform(self, X):
        """Complete transformation of X in the  given the learned Gaussianization parameters.

        Parameters
        ----------
        X : array, (n_samples, n_dimensions)
            The X that follows a Gaussian distribution to be transformed
            to data in the original input space.

        Returns
        -------
        X_input_domain : array, (n_samples, n_dimensions)
            The new transformed X in the original input space.

        """
        X = check_array(X, ensure_2d=True, copy=True)

        for igauss, irotation in zip(
            self.gauss_params[::-1], self.rotation_matrix[::-1]
        ):

            # ----------------------
            # Rotation
            # ----------------------
            X = np.dot(X, irotation.T)

            # ----------------------------
            # Marginal Gaussianization
            # ----------------------------

            for idim in range(X.shape[1]):

                X[:, idim] = gaussian_inverse_transform(X[:, idim], igauss[idim])

        return X

    def _get_information_tolerance(self, n_samples):
        """Precompute some tolerances for the tails."""
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]

        return interp1d(xxx, yyy)(n_samples)

    def jacobian(self, X: np.ndarray):
        """Calculates the jacobian matrix of the X.

        Parameters
        ----------
        X : array, (n_samples, n_features)
            The input array to calculate the jacobian using the Gaussianization params.

        return_X_transform : bool, default: False
            Determines whether to return the transformed Data. This is computed along
            with the Jacobian to save time with the iterations

        Returns
        -------
        jacobian : array, (n_samples, n_features, n_features)
            The jacobian of the data w.r.t. each component for each direction

        X_transformed : array, (n_samples, n_features) (optional)
            The transformed data in the Gaussianized space
        """
        X = check_array(X, ensure_2d=True, copy=True)
        n_samples, n_components = X.shape

        X_logdetjacobian = np.zeros((n_samples, n_components, self.n_layers))

        for ilayer, (igauss, irotation) in enumerate(
            zip(self.gauss_params, self.rotation_matrix)
        ):
            # ----------------------------
            # Marginal Gaussianization
            # ----------------------------

            for idim in range(X.shape[1]):

                # marginal gaussian transformation
                (
                    X[:, idim],
                    X_logdetjacobian[:, idim, ilayer],
                ) = gaussian_transform_jacobian(X[:, idim], igauss[idim])

            # ----------------------
            # Rotation
            # ----------------------
            X = np.dot(X, irotation)
        return X, X_logdetjacobian

    def log_det_jacobian(self, X: np.ndarray):
        """Calculates the jacobian matrix of the X.

        Parameters
        ----------
        X : array, (n_samples, n_features)
            The input array to calculate the jacobian using the Gaussianization params.

        return_X_transform : bool, default: False
            Determines whether to return the transformed Data. This is computed along
            with the Jacobian to save time with the iterations

        Returns
        -------
        jacobian : array, (n_samples, n_features, n_features)
            The jacobian of the data w.r.t. each component for each direction

        X_transformed : array, (n_samples, n_features) (optional)
            The transformed data in the Gaussianized space
        """
        X = check_array(X, ensure_2d=True, copy=True)

        X += 1e-1 * np.random.rand(X.shape[0], X.shape[1])
        n_samples, n_components = X.shape

        X_logdetjacobian = np.zeros((n_samples, n_components))
        X_ldj = np.zeros((n_samples, n_components))
        self.jacs_ = list()
        self.jacs_sum_ = list()

        for ilayer, (igauss, irotation) in enumerate(
            zip(self.gauss_params, self.rotation_matrix)
        ):
            # ----------------------------
            # Marginal Gaussianization
            # ----------------------------

            for idim in range(X.shape[1]):

                # marginal gaussian transformation
                (X[:, idim], X_ldj[:, idim],) = gaussian_transform_jacobian(
                    X[:, idim], igauss[idim]
                )

                # print(
                #     X_logdetjacobian[:, idim].min(),
                #     X_logdetjacobian[:, idim].max(),
                #     X_ldj.min(),
                #     X_ldj.max(),
                # )
                msg = f"X: {np.min(X[:, idim]):.5f}, {np.max(X[:, idim]):.5f}"
                msg += f"\nLayer: {ilayer, idim}"
                assert not np.isinf(X_logdetjacobian).any(), msg
                # X_ldj = np.clip(X_ldj, -2, 2)
            # ----------------------
            # Rotation
            # ----------------------
            X_logdetjacobian += X_ldj.copy()
            # X_logdetjacobian = np.clip(X_logdetjacobian, -10, 10)
            self.jacs_.append(np.percentile(X_ldj, [0, 5, 10, 50, 90, 95, 100]))
            self.jacs_sum_.append(
                np.percentile(X_logdetjacobian, [0, 5, 10, 50, 90, 95, 100])
            )
            X = np.dot(X, irotation)

        return X, X_logdetjacobian

    def predict_proba(self, X):
        """ Computes the probability of the original data under the generative RBIG
        model.

        Parameters
        ----------
        X : array, (n_samples x n_components)
            The points that the pdf is evaluated

        n_trials : int, (default : 1)
            The number of times that the jacobian is evaluated and averaged

        TODO: make sure n_trials is an int
        TODO: make sure n_trials is 1 or more

        chunksize : int, (default: 2000)
            The batchsize to calculate the jacobian matrix.

        TODO: make sure chunksize is an int
        TODO: make sure chunk size is greater than 0

        domain : {'input', 'gauss', 'both'}
            The domain to calculate the PDF.
            - 'input' : returns the original domain (default)
            - 'gauss' : returns the gaussian domain
            - 'both'  : returns both the input and gauss domain

        Returns
        -------
        prob_data_input_domain : array, (n_samples, 1)
            The probability
        """
        X = check_array(X, ensure_2d=True, copy=True)

        # get transformation and jacobian
        Z, X_ldj = self.log_det_jacobian(X)
        logging.debug(f"Z: {np.percentile(Z, [0, 5, 50, 95, 100])}")

        # calculate the probability
        Z_logprob = stats.norm.logpdf(Z)

        logging.debug(f"Z_logprob: {np.percentile(Z_logprob, [0, 5, 50, 95, 100])}")
        logging.debug(f"X_ldj: {np.percentile(X_ldj, [0, 5, 50, 95, 100])}")

        # calculate total probability
        X_logprob = (Z_logprob + X_ldj).sum(-1)
        logging.debug(f"X_logprob: {np.percentile(X_logprob, [0, 5, 50, 95, 100])}")
        X_prob = np.exp(X_logprob)

        logging.debug(f"XProb: {np.percentile(X_prob, [0, 5, 50, 95, 100])}")

        return X_prob.reshape(-1, 1)

    def entropy(self, correction=None):

        # TODO check fit
        if (correction is None) or (correction is False):
            correction = self.entropy_correction
        return (
            entropy_marginal(self.X_fit_, correction=correction).sum()
            - self.mutual_information
        )

    def total_correlation(self):

        # TODO check fit
        return self.residual_info.sum()
