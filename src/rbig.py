import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score as mi_score
from scipy.stats import norm
import warnings
import logging

warnings.filterwarnings('ignore') # get rid of annoying warnings
logging.basicConfig(filename="rbig_demo.log",
                    level=logging.DEBUG,
                    format="%(asctime)s: %(name)-12s %(levelname)-8s: %(message)s",
                    filemode='w')

class RBIG(object):
    def __init__(self, precision=1000, subsample=1000,n_layers=1000, 
                 random_state=None):
        self.precision = precision
        self.subsample = subsample
        self.n_layers = n_layers
        self.random_state = random_state

    def fit(self, data):

        n_samples, n_dimensions = data.shape

        data_transformed = np.copy(data)
        residual_info = np.empty(shape=(self.n_layers, n_dimensions))
        gauss_models = [None] * self.n_layers
        pca_models = [None] * self.n_layers

        # loop through the layers
        for layer in np.arange(0, self.n_layers):

            # ------------------
            # Gaussian(-ization)
            # ------------------

            # Initialize Gaussian(-nizer) Class
            Gaussianizer = QuantileTransformer(n_quantiles=self.precision,
                                               subsample=self.subsample,
                                               random_state=self.random_state,
                                               output_distribution='normal')

            # Fit and Transform Gaussian(-izer) Model the data
            data_gauss = Gaussianizer.fit_transform(data_transformed)

            # save gauss model
            gauss_models[layer] = Gaussianizer

            # --------
            # Rotation
            # --------

            # Initialize PCA Model
            PCA_model = PCA()

            # Fit and Transform PCA Model to data
            data_transformed = PCA_model.fit_transform(data_gauss)

            # save the PCA Model
            pca_models[layer] = PCA_model

            # ------------------
            # Mutual Information
            # ------------------

            for idim in range(0, n_dimensions):

                residual_info[layer, idim] = \
                    mi_score(data_gauss[:, idim], data_transformed[:, idim])

        # save necessary parameters
        self.data_transformed_ = data_transformed
        self.residual_info = residual_info
        self.multi_information_ = np.sum(residual_info, axis=1)
        self.pca_models = pca_models
        self.gauss_models = gauss_models


        return self

    def transform(self, data):

        # get the dimensions of the data
        data_transformed = np.copy(data)


        # Loop through the layers (forwards)
        for layer in range(0, self.n_layers):

            # Gaussian Transformation
            GaussianModel = self.gauss_models[layer]

            data_transformed = GaussianModel.transform(data_transformed)

            # PCA Rotation
            PCA_Model = self.pca_models[layer]

            data_transformed = PCA_Model.transform(data_transformed)

        return data_transformed

    def inverse_transform(self, data):

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
    title = 'Raw Toy Data'
    plot_toydata(x_train, title)

    # initialize RBIG function
    logging.info('Initialzing RBIG Function ...')

    n_quantiles = 1000
    subsample = 1000
    random_state = 123

    RBIG_model = RBIG(precision=n_quantiles,
                      subsample=subsample,
                      random_state=random_state)

    # Fit to data
    logging.info('Calling: RBIG function ...')

    RBIG_model.fit(x_train)

    x_transformed = RBIG_model.data_transformed_

    title = 'Original Transformed Data'
    plot_toydata(x_transformed, title)

    # Transform New Data
    logging.info('Calling: rbig transformation function ...')

    x_test_transformed = RBIG_model.transform(x_test)

    title = 'New Transformed Data'
    plot_toydata(x_test_transformed, title)

    logging.info('Calling: rbig inverse transformation function ...')

    # create a random matrix with a normal distribution
    generator = check_random_state(random_state)
    x_random = generator.randn(num_points, x_train.shape[1])

    x_synthetic = RBIG_model.inverse_transform(x_random)

    title = 'Synthetic Data'
    plot_toydata(x_synthetic, title)

    return None

if __name__ == "__main__":

    run_demo()
