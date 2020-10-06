import numpy as np
import sys
import warnings
from scipy.interpolate import interp1d
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
from sklearn.utils import check_array

from rbig.information.entropy import entropy_marginal
from rbig.density import univariate_make_normal


def information_reduction(x_data, y_data, tol_dimensions=None, correction=True):
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
    err_msg = "Number of samples for x and y should be equal."
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

    # calculate the marginal entropy
    hx = entropy_marginal(x_data, correction=correction)
    hy = entropy_marginal(y_data, correction=correction)

    # Information content
    I = np.sum(hy) - np.sum(hx)
    II = np.sqrt(np.sum((hy - hx) ** 2))

    p = 0.25
    if II < np.sqrt(n_dimensions * p * tol_dimensions ** 2) or I < 0:
        I = 0

    return I


class RBIGTotalCorr:
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

            for idim in range(n_dimensions):

                gauss_data[:, idim], _ = univariate_make_normal(
                    gauss_data[:, idim], self.pdf_extension, self.pdf_resolution
                )

            gauss_data_prerotation = gauss_data.copy()
            if self.verbose == 2:
                print(gauss_data.min(), gauss_data.max())

            # --------
            # Rotation
            # --------
            if self.rotation_type == "random":

                rand_ortho_matrix = ortho_group.rvs(n_dimensions)
                gauss_data = np.dot(gauss_data, rand_ortho_matrix)

            elif self.rotation_type.lower() == "pca":

                # Initialize PCA model
                if self.rotation_kwargs is not None:
                    pca_model = PCA(
                        random_state=self.random_state, **self.rotation_kwargs
                    )
                else:
                    pca_model = PCA(random_state=self.random_state)

                gauss_data = pca_model.fit_transform(gauss_data)

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
        self.mutual_information = np.sum(self.residual_info)
        self.n_layers = len(self.residual_info)

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

    def _get_information_tolerance(self, n_samples):
        """Precompute some tolerances for the tails."""
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]

        return interp1d(xxx, yyy)(n_samples)

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
