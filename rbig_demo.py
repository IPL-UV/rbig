import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from scipy.stats import norm
from scipy import linalg
import logging

logging.basicConfig(filename="logs/rbig_demo.log",
                    level=logging.DEBUG,
                    format="%(asctime)s: %(name)-12s %(levelname)-8s: %(message)s",
                    filemode='w')

class RBIG(object):
    def __init__(self, precision=1000, porc=10, transformation='pca',
                 n_layers=1000, tol_samples=None, tol_dimensions=None,
                 random_state=None):
        self.precision = precision
        self.porc = porc
        self.transformation = transformation
        self.n_layers = n_layers
        self.tol_samples = tol_samples
        self.tol_dimensions = tol_dimensions
        self.random_state = random_state

    def run_demo(self, num_points=1000, noise=0.2, random_state=None):

        logging.info('Running: run_demo ...')
        logging.info('Calling: generate_data ...')

        self.data = self.generate_data(num_points=num_points,
                                       noise=noise,
                                       random_state=random_state)

        logging.info('Calling: rbig function ...')

        logging.info('Calling: rbig transformation function ...')

        logging.info('Calling: rbig inverse transformation function ...')


        return None

    def generate_data(self, num_points=1000, noise=0.2, random_state=None):

        generator = check_random_state(random_state)

        data_auxilary = generator.randn(1, num_points)

        data = np.cos(data_auxilary)
        data = np.vstack((data, np.sinc(data_auxilary)))

        data = data + noise * generator.rand(2, num_points)
        data = np.dot(np.array([[0.5, 0.5], [-0.5, 0.5]]), data).T

        logging.debug('Size of data_auxilary: {}'.format(data_auxilary.shape))
        logging.debug('Size of data: {}'.format(data.shape))
        logging.debug('data : {}'.format(data[:10]))

        return data

    def plot_toydata(self):

        if not hasattr(self, 'data'):
            self.generate_data()

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(self.data[:, 0], self.data[:, 1], s=10, c='k', label='Toy Data')
        ax.set_title('Toy Data: Raw')

        plt.show()

        return None

    def rbig_apply(self, data, transform):

        logging.info('Running: rbig apply algorithm ...')
        # get dimensions of information
        # get size of the X data
        n_samples, n_dimensions = data.shape

        precision = transform['precision']

        # for computational speed
        logging.info('Modifying data for computational speed ...')
        n_dimensions = 500000
        mmod = np.mod(n_samples, n_dimensions)
        fflor = np.floor(n_samples / n_dimensions)

        # initialize the data
        data_transformed = np.zeros(data.shape)

        for j_sample in np.arange(0, fflor * n_dimensions, n_dimensions):

            data_0 = data[j_sample:j_sample + n_dimensions, :]

            for i_sample in np.range(0, len(transform['data'])):
                data_0 = self._marginal_gaussianization(data_0[:, n_dimensions],
                                                  transform['data'][i_sample],
                                                  precision)

            V = transform['v'][n_samples]
            data_0 = V * data_0

        data_transformed[j_sample:j_sample + n_dimensions] = data_0

        if mmod > 0:
            data_0 = data[fflor * n_dimensions:]

            for i_sample in np.arange(0, len(transform['data'])):
                data_0 = self._marginal_gaussianization(data_0[:, n_dimensions],
                                                  transform['data'][i_sample],
                                                  precision)

            V = transform['v'][n_samples]
            data_0 = V * data_0

        data_transformed[j_sample:j_sample + n_dimensions] = data_0

        return data_transformed

    # def inv_rbig_apply(self, data, transform):
    #
    #     precision = transform.precision[0]
    #     n_dimensions = data.shape
    #     data_0 = data
    #
    #     for i_sample in np.arange(0, transform.shape[0])[-1::]:

    def _calculate_tolerance(self, n_errors=1000):

        n_errors = 1000
        ee = np.zeros(n_errors)

        for rep in np.arange(0, n_errors):
            # HX
            [counts, centers] = np.histogram(a=np.random.randn(1, np.round(self.n_samples)),
                                             bins=np.int(np.round(np.sqrt(self.n_samples))))

            delta = centers[1] - centers[0]

            constant = 0.5 * (np.sum(counts[counts > 0]) - 1) / np.sum(counts)
            counts = counts / np.sum(counts)
            hx = -np.sum(counts[counts != 0] * np.log(counts[counts != 0])) + constant
            hx += delta

            # HY
            [counts, centers] = np.histogram(a=np.random.randn(1, np.round(self.n_samples)),
                                             bins=np.int(np.round(np.sqrt(self.n_samples))))
            #     centers = centers[:-1] = np.diff(centers)/2
            delta = centers[1] - centers[0]

            constant = 0.5 * (np.sum(counts[counts > 0]) - 1) / np.sum(counts)
            counts = counts / np.sum(counts)
            hy = -np.sum(counts[counts != 0] * np.log(counts[counts != 0])) + constant
            hy += delta

            ee[rep] = hy - hx

        self.tol_dimensions = np.mean(ee)
        self.tol_samples = np.std(ee)

        return None

    def _marginal_gaussianization(self, data, porc=None, precision=None):

        if precision is None:
            precision = np.round(np.sqrt(len(data)) + 1)

        data_uniform = self._marginal_uniformization(data, T, precision)

        return norm.ppf(data_uniform)

    def _marginal_uniformization(self, data, porc=None, precision=None):

        if precision is None:
            precision = 1000

        return np.interp(T.R, T.C, data)

    def _inv_marginal_gaussianization(self, data, precision=None):

        return None

    def _inv_marginal_uniformization(self, data, precision=None):

        return None

    def rbig(self, data, n_samples=1000):

        n_samples, n_dimensions = data.shape



        transform = []
        transform_t = []
        projection_mat = []
        residual_info = []

        # calculate the tolerance for samples and dimensions
        if self.tol_samples or self.tol_dimensions is None:
            tolerance_m, tolerance_d = calculate_tolerance(n_samples=1000)

        # loop through the layers
        for i_layer in np.arange(0, self.n_layers):

            # loop through each dimension
            for idimension in np.arange(0, n_dimensions):

                # perform gaussian marginalization
                data[:, idimension], T = \
                    self._marginal_gaussianization(data[:, idimension],
                                                   self.porc,
                                                   self.precision)


                # store the transformation
                transform_t.append(T)

            transform.append(transform_t)


            # rotation
            if self.transformation in ['rnd', 'RND']:
                V = np.random(n_dimensions)
                V = V * linalg.inv(V.T * V)                         # orthogonalization
                V = V / (np.abs(linalg.det(V))**(1/V.shape[1]))     # normalization
                data = V.T * data

            elif self.transformation in ['pca', 'PCA']:
                C = data * data.T / data.shape[0]
                V, D = linalg.eig(C)
                data = V.T * data

            else:
                raise ValueError('Unrecognized method. Need to '
                                 'use PCA or rnd.')

            projection_mat.append(V)

            # multi-information reduction
            residual_info.append(information_reduction(data, data, self.tol_samples,
                                                       self.tol_dimensions))






        return None


def information_reduction(data, aux_data, tol_samples, tol_dimensions,
                          random_state=None):
    generator = check_random_state(random_state)
    dimensions, n_samples = data.shape

    hx = np.zeros(n)
    for n in np.arange(0, dimensions):
        # calculate entropy in X direction
        [p, R] = np.histogram(a=data[n, :], np.sqrt(n_samples))
        delta = R[2] - R[1]
        hx[n] = entropy(p) + np.log(delta)

        # calculate entropy in Y direction
        [p, R] = np.histogram(a=Y[n, :], np.sqrt(n_samples))
        delta = R[2] - R[1]
        hy[n]

    I = np.sum(hy) - np.sum(hx)
    II = np.sqrt(np.sum((hy - hx) ** 2))
    p = 0.25

    if np.abs(II) < np.sqrt(dimensions * p * tol_d ** 2):
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

def rbig(data, precision=100, domain_prnt=10, transformation='pca',
         number_layers=1000, tolerance_m=None, tolerance_d=None,
         verbose=0):

    # get size of the X data
    n_samples, dimensions = data.shape

    logging.debug('Sizes: n_samples: {}'.format(n_samples))
    logging.debug('Sizes: dimensions: {}'.format(dimensions))


    if tolerance_m or tolerance_d is None:
        temp_m, temp_d = _calculate_tolerance(n_samples)

        if tolerance_m is None:
            tolerance_m = temp_m

        if tolerance_d is None:
            tolerance_d = temp_d

    if verbose:
        print('Tolerance m: ', tolerance_m)
        print('Tolerance d: ', tolerance_d)
    return

def calculate_tolerance(n_samples):

    n_errors = 1000
    ee = np.zeros(n_errors)

    for rep in np.arange(0, n_errors):
        # HX
        [counts, centers] = np.histogram(a=np.random.randn(1, np.round(n_samples)),
                                         bins=np.int(np.round(np.sqrt(n_samples))))

        #     centers = centers[:-1] = np.diff(centers)/2
        delta = centers[1] - centers[0]

        constant = 0.5 * (np.sum(counts[counts > 0]) - 1) / np.sum(counts)
        counts = counts / np.sum(counts)
        hx = -np.sum(counts[counts != 0] * np.log(counts[counts != 0])) + constant
        hx += delta

        # HY
        [counts, centers] = np.histogram(a=np.random.randn(1, np.round(n_samples)),
                                         bins=np.int(np.round(np.sqrt(n_samples))))
        #     centers = centers[:-1] = np.diff(centers)/2
        delta = centers[1] - centers[0]

        constant = 0.5 * (np.sum(counts[counts > 0]) - 1) / np.sum(counts)
        counts = counts / np.sum(counts)
        hy = -np.sum(counts[counts != 0] * np.log(counts[counts != 0])) + constant
        hy += delta

        ee[rep] = hy - hx

    tolerance_m = np.mean(ee)
    tolerance_d = np.std(ee)

    return (tolerance_m, tolerance_d)


def run_rbigdemo():
    demo = RBIG()
    demo.run_demo()

if __name__ == "__main__":

    run_rbigdemo()
