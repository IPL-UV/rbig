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

warnings.filterwarnings('ignore') # get rid of annoying warnings
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

        n_samples, n_dimensions = np.shape(data.T)
        gauss_data = np.copy(data)

        residual_info = np.empty(shape=(self.n_layers, n_dimensions))
        gauss_params = [None] * self.n_layers
        rotation_matrix = [None] * self.n_layers

        # Loop through the layers
        for layer in range(self.n_layers):

            if self.verbose is not None:
                print("Completed {} iterations of RBIG.".format(layer+1))

            # ------------------
            # Gaussian(-ization)
            # ------------------
                layer_params = list()

                for idim in range(n_dimensions):

                    gauss_data[idim, :], temp_params = univariate_make_normal(
                        gauss_data[idim, :],
                        self.pdf_extension,
                        self.pdf_resolution
                    )

                    # append the parameters
                    layer_params.append(temp_params)
                    
                gauss_params[layer] = layer_params
            # --------
            # Rotation
            # --------
            if self.rotation_type == 'random':
                rand_ortho_matrix = ortho_group.rvs(n_dimensions)
                gauss_data = np.dot(rand_ortho_matrix, gauss_data)
                rotation_matrix[layer] = rand_ortho_matrix

            elif self.rotation_type == 'PCA':
                if n_dimensions > n_samples or n_dimensions > 10**6:
                    # If the dimensionality of each datapoint is high, we probably
                    # want to compute the SVD of the data directly to avoid forming a huge
                    # covariance matrix
                    U, _, _ = np.linalg.svd(gauss_data, full_matrices=True)
                else:
                    # the SVD is more numerically stable then eig so we'll use it on the 
                    # covariance matrix directly
                    U, _, _ = np.linalg.svd(
                        np.dot(gauss_data, gauss_data.T) / n_samples,
                        full_matrices=True)

                gauss_data = np.dot(U.T, gauss_data)
                rotation_matrix[layer] = U.T

            else:
                raise ValueError('Rotation type ' + self.rotation_type + ' not recognized')
            

            # ------------------
            # Mutual Information
            # ------------------

            for idim in range(0, n_dimensions):
                residual_info[layer, idim] = \
                    mi_score(gauss_data[idim, :].T, gauss_data[idim, :].T)

        # save necessary parameters
        self.gauss_data = gauss_data
        self.residual_info = residual_info
        self.multi_information_ = np.sum(residual_info, axis=1)
        self.rotation_matrix = rotation_matrix
        self.gauss_params = gauss_params

        return self

    def transform(self, data):

        # get the dimensions of the data
        n_samples, n_dimensions = np.shape(data.T)
        rbig_transformed = np.copy(data)

        for layer in range(self.n_layers):

            for idim in range(n_dimensions):

                # marginal uniformization
                rbig_transformed[idim, :] = interp1d(
                    self.gauss_params[layer][idim]['uniform_cdf_support'],
                    self.gauss_params[layer][idim]['uniform_cdf'],
                    fill_value='extrapolate')(
                        rbig_transformed[idim, :]
                    )

                # marginal gaussianization
                rbig_transformed[idim] = norm.ppf(
                    rbig_transformed[idim, :]
                )

                # rotation
                rbig_transformed = np.dot(
                    self.rotation_matrix[layer], rbig_transformed
                )

            return rbig_transformed

    def inverse_transform(self, data):
        n_dimensions, n_samples = np.shape(data)
        sampled_data = np.copy(data)

        total_iters = 0
        for layer in range(self.n_layers-1, -1, -1):

            if self.verbose is not None:
                print("Completed {} inverse iterations of RBIG.".format(layer+1))

            sampled_data = np.dot(self.rotation_matrix[layer], sampled_data)

            for idim in range(n_dimensions):
                sampled_data[idim, :] = univariate_invert_normalization(
                    sampled_data[idim, :],
                    self.gauss_params[layer][idim]
                )

        return sampled_data
    # def inverse_transform(self, data):

        # get the dimensions of the data
        data_transformed = np.copy(data)

        # Loop through the layers (backwards)
        for layer in np.arange(0, self.n_layers)[::-1]:

            # PCA Rotation
            PCA_Model = self.pca_models[layer]

            data_transformed = PCA_Model.inverse_transform(data_transformed)

            # Gaussian Transformation
            GaussianModel = self.gauss_models[layer]

            data_transformed = GaussianModel.inverse_transform(data_transformed)

        return data_transformed


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
  data_uniform, params = univariate_make_uniform(uni_data, extension, precision)
  return norm.ppf(data_uniform), params

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

def information_reduction(x_data, y_data, tol_dimensions, random_state=None):

    # get dimensions of x,y data
    n_xdimensions, n_xsamples = x_data.shape
    n_ydimensions, n_ysamples = y_data.shape

    # check if same dimensions
    if n_xsamples is not n_ysamples:
        raise ValueError('number of x samples is not equal to number of y samples.')

    # preallocate data
    hx = np.zeros(n_xdimensions)
    hy = hx.copy()

    # loop through dimensions
    for n in np.arange(0, n_xdimensions):

        # calculate entropy in X direction
        [hist_x, bin_edges_x] = np.histogram(a=x_data[n, :], bins=np.sqrt(n_xsamples))
        delta = bin_edges_x[2] - bin_edges_x[1]
        hx[n] = entropy(hist_x) + np.log(delta)

        # calculate entropy in Y direction
        [hist_y, bin_edges_y] = np.histogram(a=y_data[n, :], bins=np.sqrt(n_ysamples))
        delta_y = bin_edges_y[2] - bin_edges_y[1]
        hy[n] = entropy(hist_y) + np.log(delta_y)

    I = np.sum(hy) - np.sum(hx)
    II= np.sqrt(np.sum(hy - hx) ** 2)
    p = 0.25

    if np.abs(II) < np.sqrt(n_xdimensions * p * tol_dimensions ** 2):
        I = 0

    return I

def entropy(counts):
    # MLE estimator with miller-maddow correction
    constant = 0.5 * (np.sum(counts[counts > 0]) - 1) / np.sum(counts)
    print(constant)
    hist_counts = counts / np.sum(counts)
    idx = np.where(hist_counts != 0)
    H = -np.sum(counts[idx != 0]) * np.log(counts[idx != 0]) + constant

    return H

def generate_data(num_points=1000, noise=0.2, random_state=None):

    generator = check_random_state(random_state)

    data_auxilary = generator.randn(1, num_points)

    data = np.cos(data_auxilary)
    data = np.vstack((data, np.sinc(data_auxilary)))

    data = data + noise * generator.rand(2, num_points)
    data = np.dot(np.array([[0.5, 0.5], [-0.5, 0.5]]), data).T

    logging.debug('Size of data_auxilary: {}'.format(data_auxilary.shape))
    logging.debug('Size of data: {}'.format(data.shape))

    return data

def plot_toydata(data, title=None):

    fig, ax = plt.subplots()

    ax.scatter(data[:, 0], data[:, 1], s=10, c='k', label='Toy Data')

    if title is not None:
        ax.set_title(title)

    plt.show()

    return None

def run_demo():


    logging.info('Running: run_demo ...')
    logging.info('Calling: generate_data ...')

    num_points = 2000
    noise = 0.2
    random_state = 123

    data = generate_data(num_points=num_points,
                         noise=noise,
                         random_state=random_state)

    # split into training and testing
    train_percent = 0.5
    x_train, x_test = train_test_split(data,
                                       train_size=train_percent,
                                       random_state=random_state)

    logging.info('Calling: plot_toydata() ...')

    # logging.info('Plotting toy data ...')
    # title = 'Raw Toy Data'
    # plot_toydata(x_train, title)

    # initialize RBIG function
    logging.info('Initialzing RBIG Function ...')

    n_layers = 75
    rotation_type = 'PCA'
    pdf_resolution = 1000
    pdf_extension = 0.1
    verbose = 1
    random_state = 123

    RBIG_model = RBIG(n_layers=n_layers,
                      rotation_type=rotation_type,
                      pdf_resolution=pdf_resolution,
                      pdf_extension=pdf_extension,
                      verbose=verbose,
                      random_state=random_state)

    # Fit to data
    logging.info('Calling: RBIG function ...')

    RBIG_model.fit(x_train.T)

    print('Shape of Gauss Data: {}'.format(RBIG_model.gauss_data.shape))
    print('Shape of Multi Info: {}'.format(RBIG_model.multi_information_.shape))
    print('Shape of Rotations: {}'.format(len(RBIG_model.rotation_matrix)))
    print('Shape of Gauss Parameters: {}'.format(len(RBIG_model.gauss_params)))
    x_transformed = RBIG_model.gauss_data

    print('Shape of Transformed Gauss Data: {}'.format(x_transformed.shape))


    title = 'Original Transformed Data'
    plot_toydata(x_transformed.T, title)

    # Transform New Data
    logging.info('Calling: rbig transformation function ...')

    x_test_transformed = RBIG_model.transform(x_test.T)

    print('Shape of TransformedGauss Data: {}'.format(x_test_transformed.shape))


    title = 'New Transformed Data'
    plot_toydata(x_test_transformed.T, title)

    logging.info('Calling: rbig inverse transformation function ...')

    # create a random matrix with a normal distribution
    generator = check_random_state(random_state)
    x_random = generator.randn(num_points, x_train.shape[1])

    

    x_synthetic = RBIG_model.inverse_transform(x_random.T)

    title = 'Synthetic Data'
    plot_toydata(x_synthetic.T, title)

    return None

if __name__ == "__main__":

    run_demo()
