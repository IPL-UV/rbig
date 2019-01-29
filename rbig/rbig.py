import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score as mi_score
from scipy.stats import norm, ortho_group, entropy as sci_entropy
from scipy.interpolate import interp1d
import warnings
import logging

warnings.filterwarnings('ignore')  # get rid of annoying warnings
logging.basicConfig(filename="rbig_demo.log",
                    level=logging.DEBUG,
                    format="%(asctime)s: %(name)-12s %(levelname)-8s: %(message)s",
                    filemode='w')


class RBIG(BaseEstimator, TransformerMixin):
    """ Rotation-Based Iterative Gaussian-ization (RBIG_)
    Parameters
    ----------
    n_layers : int, optional (default 50)
        The number of steps to run the sequence of marginal gaussianization
        and then rotation

    rotation_type : {'PCA', 'random'}
        The rotation applied to the Gaussian-ized data at each iteration.
        - 'pca'     : a principal components analysis rotation (PCA)
        - 'random'  : random rotations

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
    def __init__(self, n_layers=1000, rotation_type='PCA', pdf_resolution=1000,
                 pdf_extension=None, random_state=None, verbose=None, tolerance=None,
                 zero_tolerance=100, entropy_algo='standard'):
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.pdf_resolution = pdf_resolution
        self.pdf_extension = pdf_extension
        self.random_state = random_state
        self.verbose = verbose
        self.tolerance = tolerance
        self.zero_tolerance = zero_tolerance
        self.entropy_algo = entropy_algo
        
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
        self._fit(X)
        return self

    def _fit(self, data):
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

        if self.pdf_extension is None:
            self.pdf_extension = 2 * np.round(np.sqrt(data.shape[0]))

        self.X_fit_ = data
        gauss_data = np.copy(data)
        
        n_samples, n_dimensions = np.shape(data)
        
        if self.zero_tolerance is None:
            self.zero_tolerance = self.n_layers + 1
        
        if self.tolerance is None:
            self.tolerance = self._get_information_tolerance(n_samples)
        
        logging.debug('Data (shape): {}'.format(np.shape(gauss_data)))

        # Initialize stopping criteria (residual information)
        residual_info = list()
        gauss_params = list()
        rotation_matrix = list()

        # Loop through the layers
        logging.debug('Running: Looping through the layers...')
        for layer in range(self.n_layers):

            if self.verbose is not None:
                print("Completed {} iterations of RBIG.".format(layer + 1))

            # ------------------
            # Gaussian(-ization)
            # ------------------
            layer_params = list()

            for idim in range(n_dimensions):
                
                gauss_data[:, idim], temp_params = univariate_make_normal(
                    gauss_data[:, idim],
                    self.pdf_extension,
                    self.pdf_resolution
                )

                # append the parameters
                layer_params.append(temp_params)

            gauss_params.append(layer_params)
            gauss_data_prerotation = gauss_data.copy()
            # --------
            # Rotation
            # --------
            if self.rotation_type == 'random':

                rand_ortho_matrix = ortho_group.rvs(n_dimensions)
                gauss_data = np.dot(gauss_data, rand_ortho_matrix)
                rotation_matrix.append(rand_ortho_matrix)

            elif self.rotation_type.lower() == 'ica':
                
                # initialize model fastica model
                ica_model = FastICA()

                # fit-transform data
                gauss_data = ica_model.fit_transform(gauss_data)

                # save rotation matrix
                rotation_matrix.append(ica_model.components_.T)


            elif self.rotation_type.lower() == 'pca':

                if n_dimensions > n_samples or n_dimensions > 10 ** 6:
                    # If the dimensionality of each datapoint is high, we probably
                    # want to compute the SVD of the data directly to avoid forming a huge
                    # covariance matrix
                    _, _, V = np.linalg.svd(gauss_data, full_matrices=True)
                    
                else:
                    # the SVD is more numerically stable then eig so we'll use it on the 
                    # covariance matrix directly
                    # cov_data = np.dot(gauss_data.T, gauss_data) / n_samples
                    # _, _, V = np.linalg.svd(cov_data, full_matrices=True)
                    pca_model = PCA()
                    
                
                # logging.debug('Size of V: {}'.format(V.shape))
                logging.debug('Size of gauss_data: {}'.format(gauss_data.shape))
                # print(V.shape)
                # gauss_data = np.dot(gauss_data, V.T)
                # rotation_matrix.append(V.T)
                gauss_data = pca_model.fit_transform(gauss_data)
                rotation_matrix.append(pca_model.components_.T)

            else:
                raise ValueError('Rotation type ' + self.rotation_type + ' not recognized')
                
            # --------------------------------
            # Information Reduction (Emmanuel)
            # --------------------------------
            residual_info.append(information_reduction(gauss_data, 
                                                       gauss_data_prerotation,
                                                       self.tolerance))
            
            
            # Transform Residual Information
            if layer > self.zero_tolerance:
                aux_residual = np.array(residual_info)
                if (aux_residual[-self.zero_tolerance:].sum() == 0):
                    logging.debug('Done! aux: {}'.format(aux_residual))
                    
                    residual_info = residual_info[:-self.zero_tolerance]
                    logging.debug('Res Info: {}'.format(len(residual_info)))
                    rotation_matrix = rotation_matrix[:-self.zero_tolerance]
                    logging.debug('Rotation Matrix: {}'.format(len(rotation_matrix)))
                    gauss_params = gauss_params[:-self.zero_tolerance]
                    logging.debug('Gauss Param: {}'.format(len(gauss_params)))
                    break
                else:
                    pass

        # save necessary parameters
        try:
            self.information_loss = aux_residual
        except UnboundLocalError:
            self.information_loss = np.array(residual_info)
        self.gauss_data = gauss_data
        self.residual_info = np.array(residual_info)
        self.mutual_information = np.sum(residual_info)
        self.rotation_matrix = rotation_matrix
        self.gauss_params = gauss_params
        self.n_layers = len(gauss_params)

        return self

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
                data_layer[:, idim] = interp1d(
                    self.gauss_params[layer][idim]['uniform_cdf_support'],
                    self.gauss_params[layer][idim]['uniform_cdf'],
                    fill_value='extrapolate')(
                        data_layer[:, idim]
                    )

                # marginal gaussianization
                data_layer[:, idim] = norm.ppf(
                    data_layer[:, idim]
                )
                
            # ----------------------
            # Rotation
            # ----------------------
            X_transformed = np.dot(
                data_layer, self.rotation_matrix[layer]
            )

        return X_transformed

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
        n_dimensions = np.shape(X)[1]
        X_input_domain = np.copy(X)
        
        for layer in range(self.n_layers - 1, -1, -1):

            if self.verbose is not None:
                print("Completed {} inverse iterations of RBIG.".format(layer + 1))

            X_input_domain = np.dot(X_input_domain, self.rotation_matrix[layer].T)
            
            temp = X_input_domain
            for idim in range(n_dimensions):
                temp[:, idim] = univariate_invert_normalization(
                    temp[:, idim],
                    self.gauss_params[layer][idim]
                )
            X_input_domain = temp

        return X_input_domain

    def _get_information_tolerance(self, n_samples):
        """Precompute some tolerances for the tails."""
        xxx = np.logspace(2, 8, 6)
        yyy = [0.1571, 0.0468, 0.0046, 0.0014, 0.0001, 0.00001]

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
                    self.gauss_params[ilayer][idim]['uniform_cdf_support'],
                    self.gauss_params[ilayer][idim]['uniform_cdf'],
                    fill_value='extrapolate')(
                        X_transformed[:, idim]
                    )

                # Marginal Gaussianization
                igaussian_pdf[:, idim] = norm.ppf(data_uniform)

                # Gaussian PDF
                gaussian_pdf[:, idim, ilayer] = interp1d(
                    self.gauss_params[ilayer][idim]['empirical_pdf_support'],
                    self.gauss_params[ilayer][idim]['empirical_pdf'],
                    fill_value='extrapolate')(
                        X_transformed[:, idim]
                    ) * (1 / norm.pdf(igaussian_pdf[:, idim]))


            XX = np.dot(gaussian_pdf[:, :, ilayer] * XX, self.rotation_matrix[ilayer])

            X_transformed = np.dot(igaussian_pdf, self.rotation_matrix[ilayer])
        jacobian[:, :, 0] = XX

        if n_components > 1:

            for idim in range(n_components):

                XX = np.zeros(shape=(n_samples, n_components))
                XX[:, idim] = np.ones(n_samples)

                for ilayer in range(self.n_layers):

                    XX = np.dot(gaussian_pdf[:, :, ilayer]*XX, self.rotation_matrix[ilayer])

                jacobian[:, :, idim] = XX

        if return_X_transform:
            return jacobian, X_transformed
        else:
            return jacobian

    def predict_proba(self, X, n_trials=1, chunksize=2000, domain='input'):
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

            for start_idx, end_idx in generate_batches(n_samples, chunksize):

                jacobians[start_idx:end_idx, :, :], data_temp[start_idx:end_idx, :] = \
                    self.jacobian(data_aux[start_idx:end_idx, :], return_X_transform=True)

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
            prob_data_input_domain[:, itrial] = (
                prob_data_gaussian_domain[:, itrial] * np.abs(det_jacobians)
            )

            # set all nans to zero
            prob_data_input_domain[np.isnan(prob_data_input_domain)] = 0.0



        # Average all the jacobians we calculate
        prob_data_input_domain = prob_data_input_domain.mean(axis=1)
        prob_data_gaussian_domain = prob_data_gaussian_domain.mean(axis=1)
        det_jacobians = det_jacobians.mean()

        # save the jacobians
        self.jacobians = jacobians
        self.det_jacobians = det_jacobians

        if domain == 'input':
            return prob_data_input_domain
        elif domain == 'transform':
            return prob_data_gaussian_domain
        elif domain == 'both':
            return prob_data_input_domain, prob_data_gaussian_domain

    def entropy(self, correction=False):

        #TODO check fit

        return entropy_marginal(self.X_fit_, correction=correction).sum() - self.mutual_information

    def total_correlation(self):

        #TODO check fit
        return self.residual_info.sum()


class RBIGMI(object):
    def __init__(self, n_layers=50, rotation_type='PCA', pdf_resolution=1000,
                 pdf_extension=None, random_state=None, verbose=None,
                 tolerance=None, zero_tolerance=100):
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.pdf_resolution = pdf_resolution
        self.pdf_extension = pdf_extension
        self.random_state = random_state
        self.verbose = verbose
        self.tolerance = tolerance
        self.zero_tolerance = zero_tolerance
    
    def fit(self, X, Y):


        # Initialize RBIG class I
        self.rbig_model_X = RBIG(n_layers=self.n_layers, 
                                 rotation_type=self.rotation_type, 
                                 random_state=self.random_state,
                                 zero_tolerance=self.zero_tolerance,
                                  tolerance=self.tolerance)

        # fit and transform model to the data
        X_transformed = self.rbig_model_X.fit_transform(X)

        # Initialize RBIG class II
        self.rbig_model_Y = RBIG(n_layers=self.n_layers, 
                                 rotation_type=self.rotation_type, 
                                 random_state=self.random_state,
                                 zero_tolerance=self.zero_tolerance,
                                  tolerance=self.tolerance)

        # fit model to the data
        Y_transformed = self.rbig_model_Y.fit_transform(Y)

        # Stack Data
        XY_transformed = np.hstack([X_transformed, Y_transformed])

        # Initialize RBIG class I & II
        self.rbig_model_XY = RBIG(n_layers=self.n_layers, 
                                 rotation_type=self.rotation_type, 
                                 random_state=self.random_state,
                                 zero_tolerance=self.zero_tolerance,
                                  tolerance=self.tolerance)
        


        # Fit RBIG model to combined dataset
        self.rbig_model_XY.fit(XY_transformed)

        return self

    def mutual_information(self):

        return self.rbig_model_XY.residual_info.sum()

def univariate_make_normal(uni_data, extension, precision):
    """
    Takes univariate data and transforms it to have approximately normal dist
    We do this through the simple composition of a histogram equalization
    producing an approximately uniform distribution and then the inverse of the
    normal CDF. This will produce approximately gaussian samples.
    Parameters
    ----------
    uni_data : ndarray
      The univariate data [Sx1] where S is the number of samples in the dataset
    extension : float
      Extend the marginal PDF support by this amount.
    precision : int
      The number of points in the marginal PDF

    Returns
    -------
    uni_gaussian_data : ndarray
      univariate gaussian data
    params : dictionary
      parameters of the transform. We save these so we can invert them later
    """
    data_uniform, params = univariate_make_uniform(uni_data.T, extension, precision)
    return norm.ppf(data_uniform).T, params

def univariate_make_uniform(uni_data, extension, precision):
    """
    Takes univariate data and transforms it to have approximately uniform dist
    Parameters
    ----------
    uni_data : ndarray
      The univariate data [1xS] where S is the number of samples in the dataset
    extension : float
      Extend the marginal PDF support by this amount. Default 0.1
    precision : int
      The number of points in the marginal PDF
    Returns
    -------
    uni_uniform_data : ndarray
      univariate uniform data
    transform_params : dictionary
    parameters of the transform. We save these so we can invert them later
    """
    n_samps = len(uni_data)
    support_extension = \
      (extension / 100) * abs(np.max(uni_data) - np.min(uni_data))

    # not sure exactly what we're doing here, but at a high level we're
    # constructing bins for the histogram
    bin_edges = np.linspace(np.min(uni_data), np.max(uni_data),
                           np.sqrt(n_samps) + 1)
    bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)

    counts, _ = np.histogram(uni_data, bin_edges)

    bin_size = bin_edges[2] - bin_edges[1]
    pdf_support = np.hstack((bin_centers[0] - bin_size, bin_centers,
                           bin_centers[-1] + bin_size))
    empirical_pdf = np.hstack((0.0, counts / (np.sum(counts) * bin_size), 0.0))
    #^ this is unnormalized
    c_sum = np.cumsum(counts)
    cdf = (1 - 1 / n_samps) * c_sum / n_samps

    incr_bin = bin_size / 2

    new_bin_edges = np.hstack((np.min(uni_data) - support_extension,
                             np.min(uni_data),
                             bin_centers + incr_bin,
                             np.max(uni_data) + support_extension + incr_bin))

    extended_cdf = np.hstack((0.0, 1.0 / n_samps, cdf, 1.0))
    new_support = np.linspace(new_bin_edges[0], new_bin_edges[-1], precision)
    learned_cdf = interp1d(new_bin_edges, extended_cdf)
    uniform_cdf = make_cdf_monotonic(learned_cdf(new_support))
    #^ linear interpolation
    uniform_cdf /= np.max(uniform_cdf)
    uni_uniform_data = interp1d(new_support, uniform_cdf)(uni_data)

    return uni_uniform_data, {'empirical_pdf_support': pdf_support,
                            'empirical_pdf': empirical_pdf,
                            'uniform_cdf_support': new_support,
                            'uniform_cdf': uniform_cdf}

def univariate_invert_normalization(uni_gaussian_data, trans_params):
    """
    Inverts the marginal normalization
    See the companion, univariate_make_normal.py, for more details
    """
    uni_uniform_data = norm.cdf(uni_gaussian_data)
    uni_data = univariate_invert_uniformization(uni_uniform_data, trans_params)
    return uni_data

def univariate_invert_uniformization(uni_uniform_data, trans_params):
    """
    Inverts the marginal uniformization transform specified by trans_params
    See the companion, univariate_make_normal.py, for more details
    """
    # simple, we just interpolate based on the saved CDF
    return interp1d(trans_params['uniform_cdf'],
                  trans_params['uniform_cdf_support'])(uni_uniform_data)

def make_cdf_monotonic(cdf):
    """
    Take a cdf and just sequentially readjust values to force monotonicity
    There's probably a better way to do this but this was in the original
    implementation. We just readjust values that are less than their predecessors
    Parameters
    ----------
    cdf : ndarray
      The values of the cdf in order (1d)
    """
    # laparra's version
    corrected_cdf = cdf.copy()
    for i in range(1, len(corrected_cdf)):
        if corrected_cdf[i] <= corrected_cdf[i-1]:
            if abs(corrected_cdf[i-1]) > 1e-14:
                corrected_cdf[i] = corrected_cdf[i-1] + 1e-14
            elif corrected_cdf[i-1] == 0:
                corrected_cdf[i] = 1e-80
            else:
                corrected_cdf[i] = (corrected_cdf[i-1] +
                                    10**(np.log10(abs(corrected_cdf[i-1]))))
    return corrected_cdf

#     # my version
#     # I think actually i need to make sure i is strictly increasing....
#     return np.maximum.accumulate(cdf)

# def entropy_hist(data):
    
#     # data dimensions
#     n_samples, d_dimensions = data.shape
    
#     # preallocate data
#     H = np.zeros(d_dimensions)    
#     # number of bins
#     n_bins = int(round(np.sqrt(n_samples))) + 1
#     print('n Bins: {}'.format(n_bins))
    
#     for idim in np.arange(0, d_dimensions):

#         # calculate entropy in X direction
#         [hist, bin_edges] = np.histogram(a=data[:, idim], bins=n_bins)
#         bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#         delta = bin_centers[2] - bin_centers[1]
# #         print(delta)
# #         print(hist)
#         h = entropy(hist)
        
# #         # MLE estimator with miller-maddow correction
# #         idx = np.where(hist_counts > 0)
# #         constant = 0.5 * (np.sum(hist_counts[idx]) - 1.0) / np.sum(hist_counts)
# #         hist_counts = hist_counts / np.sum(hist_counts)
    
# #         H = - np.sum(hist_counts[idx] * np.log2(hist_counts[idx])) + constant
#         print('Little h: {}'.format(h))
#         H[idim] = h + np.log2(delta)
        
#     return H
        
def entropy_multi(x_data, tol_dimensions=None, algorithm='standard'):
    
    n_samples, n_dimensions = x_data.shape
    
    # minimum multi-information heuristic
    if tol_dimensions is None or 0:
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        tol_dimensions = np.interp(n_samples, xxx, yyy)
    
#     if algorithm == 'standard':
#         entropy = sci_entropy()
#     elif algorithm == 'miller-maddow':
#         entropy = entropy()
#     else:
#         raise ValueError('Unrecognized entropy algorithm')
    # preallocate data
    H = np.zeros(n_dimensions)
    
    # number of bins
    n_bins = int(round(np.sqrt(n_samples))) + 1
    
    
    print('here')
    # loop through dimensions
    for idim in np.arange(0, n_dimensions):

        # calculate entropy in X direction
        Raux = np.linspace(x_data[:, idim].min(), x_data[:, idim].max(), n_bins)
        [hist_x, bin_edges_x] = np.histogram(a=x_data[:, idim], bins=Raux)
        bin_centers_x = 0.5 * (bin_edges_x[1:] + bin_edges_x[:-1])
        delta_x = bin_centers_x[2] - bin_centers_x[1]
        if algorithm == 'standard':
            correction = None
            # h = sci_entropy(hist_x, base=2)
            # h = entropy
        elif algorithm == 'miller-maddow':
            # h = entropy(hist_x)
            correction = True
        else:
            raise ValueError('Unrecognized entropy algorithm: {}...'
                             .format(algorithm))
        H[idim] = entropy(hist_x, correction) + np.log2(delta_x)
    
    return H
    
        
def entropy_marginal(data, correction=None):
    
    # Get dimensions of the data
    n_samples, d_dimensions = data.shape
    
    # get number of bins
    n_bins = int(round(np.sqrt(n_samples))) + 1
    
    H = np.zeros(d_dimensions)
    
    # loop through dimensions
    for idimension in range(d_dimensions):
        
        # get histogram counts and edges for data
        Raux = np.linspace(data[:, idimension].min(), data[:, idimension].max(), n_bins)
        [hist_counts, bin_edges] = np.histogram(a=data[:, idimension], bins=Raux)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        delta = bin_centers[2] - bin_centers[1]
#         print(f'Delta: {delta:.4f}')
        # MLE Estimator with Miller-Maddow Correction
        idx = np.where(hist_counts > 0)
        
        
        
        
        if correction:
            constant = 0.5 * (np.count_nonzero(hist_counts[idx]) - 1.0) / hist_counts.sum()
        else:
            constant = 0.0
        
        hist_counts = hist_counts / np.sum(hist_counts)
        
        h = - np.sum(hist_counts[idx] * np.log2(hist_counts[idx])) + constant
#         print(f'Little h: {h:.3f}')
        H[idimension] = h + np.log2(delta)
        
        
        
    return H
    

def information_reduction(x_data, y_data, tol_dimensions=None, correction=None):
    """Computes the multi-information (total correlation) reduction after a linear
    transformation
    
            Y = X * W
            II = I(X) - I(Y)
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data before the transformation, where n_samples is the number of samples
        and n_features is the number of features
    
    Y : array-like, shape (n_samples, n_features)
        Data after the transformation, where n_samples is the number of samples
        and n_features is the number of features
        
    tol_dimensions : float, optional
        Tolerance on the minimum multi-information difference
        
    Returns
    -------
    II : float
        The multi-information
        
    Information
    -----------
    Author: Valero Laparra
            Juan Emmanuel Johnson
    """
    # check that number of samples and dimensions are equal
    err_msg = 'Number of samples for x and y should be equal.'
    np.testing.assert_equal(x_data.shape, y_data.shape, err_msg=err_msg)
    
    n_samples, n_dimensions = x_data.shape
    
    # minimum multi-information heuristic
    if tol_dimensions is None or 0:
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        tol_dimensions = np.interp(n_samples, xxx, yyy)

    # preallocate data
    hx = np.zeros(n_dimensions)
    hy = np.zeros(n_dimensions)
    
    # number of bins
    n_bins = int(round(np.sqrt(n_samples))) + 1
#     print('Bins: ', n_bins)
    # loop through dimensions


    # calculate the entropy in
    # print('Marginal..')
    hx = entropy_marginal(x_data, correction=correction)
    hy = entropy_marginal(y_data, correction=correction)

    # for idim in np.arange(0, n_dimensions):

    #     # calculate entropy in X direction
    #     Raux = np.linspace(x_data[:, idim].min(), x_data[:, idim].max(), n_bins)
    #     [hist_x, bin_edges_x] = np.histogram(a=x_data[:, idim], bins=Raux)
    #     bin_centers_x = 0.5 * (bin_edges_x[1:] + bin_edges_x[:-1])
    #     delta_x = bin_centers_x[2] - bin_centers_x[1]
    #     hx[idim] = sci_entropy(hist_x) + np.log2(delta_x)

    #     # calculate entropy in Y direction
    #     [hist_y, bin_edges_y] = np.histogram(a=y_data[:, idim], bins=n_bins)
    #     bin_centers_y = 0.5 * (bin_edges_y[1:] + bin_edges_y[:-1])
    #     delta_y = bin_centers_y[2] - bin_centers_y[1]
    #     hy[idim] = sci_entropy(hist_y) + np.log2(delta_y)

    I = np.sum(hy) - np.sum(hx)
#     print('Data:')
#     print(hx, hy)
    II = np.sqrt(np.sum((hy - hx) ** 2))
#     print(I, II)
    p = 0.25
    if II < np.sqrt(n_dimensions * p * tol_dimensions ** 2) or I < 0:
        I = 0
    return I

def entropy(hist_counts, correction=None):
    # MLE estimator with miller-maddow correction
    idx = np.where(hist_counts > 0)
    constant = 0.5 * (np.count_nonzero(idx) - 1.0) / np.sum(hist_counts)
    hist_counts = hist_counts / np.sum(hist_counts)
    
    H = - np.sum(hist_counts[idx] * np.log2(hist_counts[idx])) + constant
    return H

def generate_batches(n_samples, batch_size):
    """A generator to split an array of 0 to n_samples
    into an array of batch_size each.

    Parameters
    ----------
    n_samples : int
        the number of samples

    batch_size : int,
        the size of each batch


    Returns
    -------
    start_index, end_index : int, int
        the start and end indices for the batch

    Source:
        https://github.com/scikit-learn/scikit-learn/blob/master
        /sklearn/utils/__init__.py#L374
    """
    start_index = 0

    # calculate number of batches
    n_batches = int(n_samples // batch_size)

    for _ in range(n_batches):

        # calculate the end coordinate
        end_index = start_index + batch_size

        # yield the start and end coordinate for batch
        yield start_index, end_index

        # start index becomes new end index
        start_index = end_index

    # special case at the end of the segment
    if start_index < n_samples:

        # yield the remaining indices
        yield start_index, n_samples


def main():
    pass


if __name__ == "__main__":
    main()
