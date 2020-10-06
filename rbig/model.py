import sys
import warnings

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import norm, ortho_group
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as mi_score
from sklearn.utils import check_array, check_random_state

from rbig.information.entropy import entropy_marginal
from rbig.information.total_corr import information_reduction
from rbig.density import univariate_invert_normalization, univariate_make_normal

warnings.filterwarnings("ignore")  # get rid of annoying warnings


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
        n_layers=1000,
        rotation_type="PCA",
        pdf_resolution=1000,
        pdf_extension=None,
        random_state=None,
        verbose: int = 0,
        tolerance=None,
        zero_tolerance=60,
        entropy_correction=True,
        rotation_kwargs=None,
        base="gauss",
    ):
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.pdf_resolution = pdf_resolution
        self.pdf_extension = pdf_extension
        self.random_state = random_state
        self.verbose = verbose
        self.tolerance = tolerance
        self.zero_tolerance = zero_tolerance
        self.entropy_correction = entropy_correction
        self.rotation_kwargs = rotation_kwargs
        self.base = base

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

        data = check_array(data, ensure_2d=True, copy=True)

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

        # Initialize stopping criteria (residual information)
        self.residual_info = list()
        self.gauss_params = list()
        self.rotation_matrix = list()

        # Loop through the layers
        for layer in range(self.n_layers):

            if self.verbose > 1:
                print("Completed {} iterations of RBIG.".format(layer + 1))

            # ------------------
            # Gaussian(-ization)
            # ------------------
            layer_params = list()

            for idim in range(n_dimensions):

                gauss_data[:, idim], temp_params = univariate_make_normal(
                    gauss_data[:, idim], self.pdf_extension, self.pdf_resolution
                )

                # append the parameters
                layer_params.append(temp_params)

            self.gauss_params.append(layer_params)
            gauss_data_prerotation = gauss_data.copy()
            if self.verbose == 2:
                print(gauss_data.min(), gauss_data.max())

            # --------
            # Rotation
            # --------
            if self.rotation_type == "random":

                rand_ortho_matrix = ortho_group.rvs(n_dimensions)
                gauss_data = np.dot(gauss_data, rand_ortho_matrix)
                self.rotation_matrix.append(rand_ortho_matrix)

            elif self.rotation_type.lower() == "pca":

                # Initialize PCA model
                if self.rotation_kwargs is not None:
                    pca_model = PCA(
                        random_state=self.random_state, **self.rotation_kwargs
                    )
                else:
                    pca_model = PCA(random_state=self.random_state)

                gauss_data = pca_model.fit_transform(gauss_data)
                self.rotation_matrix.append(pca_model.components_.T)

            else:
                raise ValueError(
                    "Rotation type " + self.rotation_type + " not recognized"
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
        n_dimensions = np.shape(X)[1]
        X_transformed = np.copy(X)

        for layer in range(self.n_layers):

            # ----------------------------
            # Marginal Uniformization
            # ----------------------------
            data_layer = X_transformed

            for idim in range(n_dimensions):

                # marginal uniformization
                # data_layer[:, idim] = univariate_make_normal(
                #     data_layer[:, idim], self.gauss_params[layer][idim]
                # )
                data_layer[:, idim] = interp1d(
                    self.gauss_params[layer][idim]["uniform_cdf_support"],
                    self.gauss_params[layer][idim]["uniform_cdf"],
                    # fill_value="extrapolate",
                )(data_layer[:, idim])

                # marginal gaussianization
                data_layer[:, idim] = norm.ppf(data_layer[:, idim])

            # ----------------------
            # Rotation
            # ----------------------
            X_transformed = np.dot(data_layer, self.rotation_matrix[layer])

        return X_transformed

    def inverse_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
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
        n_dimensions = np.shape(X)[1]
        X_input_domain = check_array(X, ensure_2d=True, copy=True)

        for layer in range(self.n_layers - 1, -1, -1):

            if self.verbose > 1:
                print("Completed {} inverse iterations of RBIG.".format(layer + 1))

            X_input_domain = np.dot(X_input_domain, self.rotation_matrix[layer].T)

            temp = X_input_domain
            for idim in range(n_dimensions):
                temp[:, idim] = univariate_invert_normalization(
                    temp[:, idim], self.gauss_params[layer][idim]
                )
            X_input_domain = temp

        return X_input_domain

    def _get_information_tolerance(self, n_samples):
        """Precompute some tolerances for the tails."""
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]

        return interp1d(xxx, yyy)(n_samples)

    def jacobian(self, X, return_X_transform=False):
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
        n_samples, n_components = X.shape

        # initialize jacobian matrix
        jacobian = np.zeros((n_samples, n_components, n_components))

        X_transformed = X.copy()

        XX = np.zeros(shape=(n_samples, n_components))
        XX[:, 0] = np.ones(shape=n_samples)

        # initialize gaussian pdf
        gaussian_pdf = np.zeros(shape=(n_samples, n_components, self.n_layers))
        igaussian_pdf = np.zeros(shape=(n_samples, n_components))

        # TODO: I feel like this is repeating a part of the transform operation

        for ilayer in range(self.n_layers):

            for idim in range(n_components):

                # Marginal Uniformization
                data_uniform = interp1d(
                    self.gauss_params[ilayer][idim]["uniform_cdf_support"],
                    self.gauss_params[ilayer][idim]["uniform_cdf"],
                    fill_value="extrapolate",
                )(X_transformed[:, idim])

                # Marginal Gaussianization
                igaussian_pdf[:, idim] = norm.ppf(data_uniform)

                # Gaussian PDF
                gaussian_pdf[:, idim, ilayer] = interp1d(
                    self.gauss_params[ilayer][idim]["empirical_pdf_support"],
                    self.gauss_params[ilayer][idim]["empirical_pdf"],
                    fill_value="extrapolate",
                )(X_transformed[:, idim]) * (1 / norm.pdf(igaussian_pdf[:, idim]))

            XX = np.dot(gaussian_pdf[:, :, ilayer] * XX, self.rotation_matrix[ilayer])

            X_transformed = np.dot(igaussian_pdf, self.rotation_matrix[ilayer])
        jacobian[:, :, 0] = XX

        if n_components > 1:

            for idim in range(n_components):

                XX = np.zeros(shape=(n_samples, n_components))
                XX[:, idim] = np.ones(n_samples)

                for ilayer in range(self.n_layers):

                    XX = np.dot(
                        gaussian_pdf[:, :, ilayer] * XX, self.rotation_matrix[ilayer]
                    )

                jacobian[:, :, idim] = XX

        if return_X_transform:
            return jacobian, X_transformed
        else:
            return jacobian

    def predict_proba(self, X, n_trials=1, chunksize=2000, domain="input"):
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
        prob_data_input_domain : array, (n_samples)
            The probability
        """
        component_wise_std = np.std(X, axis=0) / 20

        n_samples, n_components = X.shape

        prob_data_gaussian_domain = np.zeros(shape=(n_samples, n_trials))
        prob_data_input_domain = np.zeros(shape=(n_samples, n_trials))

        for itrial in range(n_trials):

            jacobians = np.zeros(shape=(n_samples, n_components, n_components))

            if itrial < n_trials:
                data_aux = X + component_wise_std[None, :]
            else:
                data_aux = X

            data_temp = np.zeros(data_aux.shape)

            # for start_idx, end_idx in generate_batches(n_samples, chunksize):

            # (
            #     jacobians[start_idx:end_idx, :, :],
            #     data_temp[start_idx:end_idx, :],
            # ) = self.jacobian(
            #     data_aux[start_idx:end_idx, :], return_X_transform=True
            # )
            jacobians, data_temp = self.jacobian(data_aux, return_X_transform=True)
            # set all nans to zero
            jacobians[np.isnan(jacobians)] = 0.0

            # get the determinant of all jacobians
            det_jacobians = np.linalg.det(jacobians)

            # Probability in Gaussian Domain
            prob_data_gaussian_domain[:, itrial] = np.prod(
                (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power(data_temp, 2)), axis=1
            )

            # set all nans to zero
            prob_data_gaussian_domain[np.isnan(prob_data_gaussian_domain)] = 0.0

            # compute determinant for each sample's jacobian
            prob_data_input_domain[:, itrial] = prob_data_gaussian_domain[
                :, itrial
            ] * np.abs(det_jacobians)

            # set all nans to zero
            prob_data_input_domain[np.isnan(prob_data_input_domain)] = 0.0

        # Average all the jacobians we calculate
        prob_data_input_domain = prob_data_input_domain.mean(axis=1)
        prob_data_gaussian_domain = prob_data_gaussian_domain.mean(axis=1)
        det_jacobians = det_jacobians.mean()

        # save the jacobians
        self.jacobians = jacobians
        self.det_jacobians = det_jacobians

        if domain == "input":
            return prob_data_input_domain
        elif domain == "transform":
            return prob_data_gaussian_domain
        elif domain == "both":
            return prob_data_input_domain, prob_data_gaussian_domain

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
