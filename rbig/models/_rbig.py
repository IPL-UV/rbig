from sklearn.base import BaseEstimator, TransformerMixin


class RBIG(BaseEstimator, TransformerMixin):
    """ Rotation-Based Iterative Gaussianization (RBIG).
    This algorithm transforms any multidimensional data to a Gaussian.
    It also provides a sampling mechanism whereby you can provide
    multidimensional gaussian data and it will generate multidimensional
    data in the original domain. You can calculate the probabilities as
    well as have access to a few information theoretic measures like total
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

    def __init__(self) -> None:
        pass
