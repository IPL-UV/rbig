import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score as mi_score
from scipy.stats import norm, ortho_group
from scipy.interpolate import interp1d
import warnings
import logging

warnings.filterwarnings('ignore')  # get rid of annoying warnings
logging.basicConfig(filename="rbig_demo.log",
                    level=logging.DEBUG,
                    format="%(asctime)s: %(name)-12s %(levelname)-8s: %(message)s",
                    filemode='w')


class RBIG(object):
    def __init__(self, n_layers=50, rotation_type='PCA', pdf_resolution=1000,
                 pdf_extension=0.1, random_state=None, verbose=None):
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.pdf_resolution = pdf_resolution
        self.pdf_extension = pdf_extension
        self.random_state = random_state
        self.verbose = verbose
        """ Rotation-Based Iterative Gaussian-ization (RBIG_)
        Parameters
        ----------
        n_layers : int, optional (default 50)
            The number of steps to run the sequence of marginal gaussianization
            and then rotation
        
        rotation_type : str {'PCA', 'random'}
            The rotation applied to the Gaussian-ized data at each iteration.
            PCA : 
            random :
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
        """

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
        
        gauss_data = np.copy(data)
        
        n_samples, n_dimensions = np.shape(gauss_data)
        
        logging.debug('Data (shape): {}'.format(np.shape(gauss_data)))

        # Initialize stopping criteria (residual information)
        residual_info = np.empty(shape=(self.n_layers))
        mutual_information = np.empty(shape=(self.n_layers, n_dimensions))
        gauss_params = [None] * self.n_layers
        rotation_matrix = [None] * self.n_layers

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

            gauss_params[layer] = layer_params
            gauss_data_prerotation = gauss_data.copy()
            # --------
            # Rotation
            # --------
            if self.rotation_type == 'random':

                rand_ortho_matrix = ortho_group.rvs(n_dimensions)
                gauss_data = np.dot(gauss_data, rand_ortho_matrix)
                rotation_matrix[layer] = rand_ortho_matrix

            elif self.rotation_type == 'PCA':
                if n_dimensions > n_samples or n_dimensions > 10 ** 6:
                    # If the dimensionality of each datapoint is high, we probably
                    # want to compute the SVD of the data directly to avoid forming a huge
                    # covariance matrix
                    _, _, V = np.linalg.svd(gauss_data, full_matrices=True)
                    
                else:
                    # the SVD is more numerically stable then eig so we'll use it on the 
                    # covariance matrix directly
                    cov_data = np.dot(gauss_data.T, gauss_data) / n_samples
                    _, _, V = np.linalg.svd(cov_data, full_matrices=True)
                
                logging.debug('Size of V: {}'.format(V.shape))
                logging.debug('Size of gauss_data: {}'.format(gauss_data.shape))

                gauss_data = np.dot(gauss_data, V.T)
                rotation_matrix[layer] = V.T

            else:
                raise ValueError('Rotation type ' + self.rotation_type + ' not recognized')
                
            # --------------------------------
            # Information Reduction (Emmanuel)
            # --------------------------------
            residual_info[layer] = information_reduction(gauss_data, gauss_data_prerotation)
            
        # save necessary parameters
        self.gauss_data = gauss_data
        self.residual_info = residual_info
        self.rotation_matrix = rotation_matrix
        self.gauss_params = gauss_params

        return self

    def transform(self, data):

        # get the dimensions of the data
        n_samples, n_dimensions = np.shape(data)
        rbig_transformed = np.copy(data)

        for layer in range(self.n_layers):
            
            # ----------------------------
            # Marginal Uniformization
            # ----------------------------
            data_layer = rbig_transformed.T
            for idim in range(n_dimensions):
                
                
                # marginal uniformization                
                data_layer[idim, :] = interp1d(
                    self.gauss_params[layer][idim]['uniform_cdf_support'],
                    self.gauss_params[layer][idim]['uniform_cdf'],
                    fill_value='extrapolate')(
                        data_layer[idim, :]
                    )

                # marginal gaussianization
                data_layer[idim, :] = norm.ppf(
                    data_layer[idim, :]
                )
                
            # ----------------------
            # Rotation
            # ----------------------
            rbig_transformed = np.dot(
                data_layer.T, self.rotation_matrix[layer] 
            )
        return rbig_transformed

    def inverse_transform(self, data):
        n_samples, n_dimensions = np.shape(data)
        sampled_data = np.copy(data)

        total_iters = 0
        for layer in range(self.n_layers - 1, -1, -1):

            if self.verbose is not None:
                print("Completed {} inverse iterations of RBIG.".format(layer + 1))

            sampled_data = np.dot(sampled_data, self.rotation_matrix[layer])
            
            temp = sampled_data.T
            for idim in range(n_dimensions):
                temp[idim, :] = univariate_invert_normalization(
                    temp[idim, :],
                    self.gauss_params[layer][idim]
                )
            sampled_data = temp.T

        return sampled_data

    def _get_information_tolerance(self, n_samples):
        """Precompute some tolerances for the tails."""
        xxx = np.linspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0046, 0.0014, 0.0001, 0.00001]
        tol_d = np.interp(n_samples, xxx, yyy)

        return tol_d

    def jacobian(self, data):
        """Calculates the jacobian matrix"""
        n_samples, n_components = data.shape

        # initialize jacobian matrix
        jacobian = np.zeros((n_samples, n_components, n_components))

        data_rbig = data.copy()

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
                        data_rbig[:, idim]
                    )

                # Marginal Gaussianization
                igaussian_pdf[:, idim] = norm.ppf(data_uniform)

                # Gaussian PDF
                gaussian_pdf[:, idim, ilayer] = interp1d(
                    self.gauss_params[ilayer][idim]['empirical_pdf_support'],
                    self.gauss_params[ilayer][idim]['empirical_pdf'],
                    fill_value='extrapolate')(
                        data_rbig[:, idim]
                    ) * (1 / norm.pdf(igaussian_pdf[:, idim]))


            XX = np.dot(gaussian_pdf[:, :, ilayer] * XX, self.rotation_matrix[ilayer])

            data_rbig = np.dot(igaussian_pdf, self.rotation_matrix[ilayer])
        jacobian[:, :, 0] = XX

        if n_components > 1:

            for idim in range(n_components):

                XX = np.zeros(shape=(n_samples, n_components))
                XX[:, idim] = np.ones(n_samples)

                for ilayer in range(self.n_layers):

                    XX = np.dot(gaussian_pdf[:, :, ilayer]*XX, self.rotation_matrix[ilayer])

                jacobian[:, :, idim] = XX
        return jacobian, data_rbig

    def estimate_prob(self, data, n_trials=1, chunksize=2000, domain='input'):

        component_wise_std = np.std(data, axis=0) / 20

        n_samples, n_components = data.shape

        prob_data_gaussian_domain = np.zeros(shape=(n_samples, n_trials))
        prob_data_input_domain = np.zeros(shape=(n_samples, n_trials))

        for itrial in range(n_trials):

            jacobians = np.zeros(shape=(n_samples, n_components, n_components))

            if itrial < n_trials:
                data_aux = data + component_wise_std[None, :]
            else:
                data_aux = data

            data_temp = np.zeros(data_aux.shape)

            for start_idx, end_idx in generate_batches(n_samples, chunksize):

                jacobians[start_idx:end_idx, :, :], data_temp[start_idx:end_idx, :] = \
                    self.jacobian(data_aux[start_idx:end_idx, :])

            # set all nans to zero
            jacobians[np.isnan(jacobians)] = 0.0

            # print(f"Jacobians - min: {jacobians.min()}, max: {jacobians.max()}")
            # print(f"Shape of Jacobians: {jacobians.shape}")
            det_jacobians = np.linalg.det(jacobians)
            # print('Det:', det_jacobians.shape)

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
        # print(prob_data_input_domain.shape)
        # print(prob_data_gaussian_domain.shape)
        # print(det_jacobians.shape)
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

    # my version
    # I think actually i need to make sure i is strictly increasing....
    # return np.maximum.accumulate(cdf)

def information_reduction(x_data, y_data, tol_dimensions=None):
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
    n_bins = int(round(np.sqrt(n_samples)))
    
    # loop through dimensions
    for idim in np.arange(0, n_dimensions):

        # calculate entropy in X direction
        [hist_x, bin_edges_x] = np.histogram(a=x_data[:, idim], bins=n_bins)
        bin_centers_x = 0.5 * (bin_edges_x[1:] + bin_edges_x[:-1])
        delta_x = bin_centers_x[2] - bin_centers_x[1]
        hx[idim] = entropy(hist_x) + np.log2(delta_x)

        # calculate entropy in Y direction
        [hist_y, bin_edges_y] = np.histogram(a=y_data[:, idim], bins=n_bins)
        bin_centers_y = 0.5 * (bin_edges_y[1:] + bin_edges_y[:-1])
        delta_y = bin_centers_y[2] - bin_centers_y[1]
        hy[idim] = entropy(hist_y) + np.log2(delta_y)

    I = np.sum(hy) - np.sum(hx)
    II = np.sqrt(np.sum(hy - hx) ** 2)
    p = 0.25

    if np.abs(II) < np.sqrt(n_dimensions * p * tol_dimensions ** 2):
        I = 0

    return I

def entropy(hist_counts, correction=None):
    # MLE estimator with miller-maddow correction
    idx = np.where(hist_counts > 0)
    constant = 0.5 * (np.sum(hist_counts[idx]) - 1.0) / np.sum(hist_counts)
    hist_counts = hist_counts / np.sum(hist_counts)
    
    H = -np.sum(hist_counts[idx] * np.log2(hist_counts[idx])) + constant
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
