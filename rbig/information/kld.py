# class RBIGKLD(object):
#     """ Rotation-Based Iterative Gaussian-ization (RBIG) applied to two
#     multidimensional variables to find the Kullback-Leibler Divergence (KLD) between
#     X and Y

#         KLD(X||Y) = int_R P_X(R) log P_Y(R) / P_X(R) dR


#     Note: as with the normal KLD,the KLD using RBIG is not symmetric.

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
#         pdf_resolution=None,
#         pdf_extension=10,
#         random_state=None,
#         verbose=None,
#         tolerance=None,
#         zero_tolerance=100,
#         increment=1.5,
#     ):
#         self.n_layers = n_layers
#         self.rotation_type = rotation_type
#         self.pdf_resolution = pdf_resolution
#         self.pdf_extension = pdf_extension
#         self.random_state = random_state
#         self.verbose = verbose
#         self.tolerance = tolerance
#         self.zero_tolerance = zero_tolerance
#         self.increment = increment

#     def fit(self, X, Y):

#         # Check Arrays
#         X = check_array(X, ensure_2d=True)
#         Y = check_array(Y, ensure_2d=True)

#         mv_g = None

#         # Loop Until convergence
#         try:
#             while mv_g is None:

#                 if self.verbose:
#                     print(f"PDF Extension: {self.pdf_extension}%")

#                 try:

#                     # initialize RBIG transform for Y
#                     self.rbig_model_Y = RBIG(
#                         n_layers=self.n_layers,
#                         rotation_type=self.rotation_type,
#                         random_state=self.random_state,
#                         zero_tolerance=self.zero_tolerance,
#                         tolerance=self.tolerance,
#                         pdf_extension=self.pdf_extension,
#                     )

#                     # fit RBIG model to Y
#                     self.rbig_model_Y.fit(Y)

#                     # Transform X using rbig_model_Y
#                     X_transformed = self.rbig_model_Y.transform(X)

#                     # Initialize RBIG transform for X_transformed
#                     self.rbig_model_X_trans = RBIG(
#                         n_layers=self.n_layers,
#                         rotation_type=self.rotation_type,
#                         random_state=self.random_state,
#                         zero_tolerance=self.zero_tolerance,
#                         tolerance=self.tolerance,
#                         pdf_extension=self.pdf_extension,
#                     )

#                     # Fit RBIG model to X_transformed
#                     self.rbig_model_X_trans.fit(X_transformed)

#                     # Get mutual information
#                     mv_g = self.rbig_model_X_trans.residual_info.sum()

#                 except:
#                     self.pdf_extension = self.increment * self.pdf_extension
#         except KeyboardInterrupt:
#             print("Interrupted!")

#         self.mv_g = mv_g
#         if self.verbose == 2:
#             print(f"mv_g: {mv_g}")
#             print(f"m_g: {neg_entropy_normal(X_transformed)}")
#         self.kld = mv_g + neg_entropy_normal(X_transformed).sum()

#         return self

#     def get_kld(self):

#         return self.kld
