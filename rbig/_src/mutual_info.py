# class RBIGMI(object):
#     """ Rotation-Based Iterative Gaussian-ization (RBIG) applied to two
#     multidimensional variables (RBIGMI). Applies the RBIG algorithm to
#     the two multidimensional variables independently, then applies another
#     RBIG algorithm on the two Gaussian-ized datasets.

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
#         pdf_resolution=1000,
#         pdf_extension=None,
#         random_state=None,
#         verbose=0,
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
#         self.increment = 1.5

#     def fit(self, X, Y):
#         """Inputs for the RBIGMI algorithm.

#         Parameters
#         ----------
#         X : array, (n1_samples, d1_dimensions)

#         Y : array, (n2_samples, d2_dimensions)

#         Note: The number of dimensions and the number of samples
#         do not have to be the same.

#         """

#         # Loop Until Convergence
#         X = check_array(X, ensure_2d=True, copy=True)
#         Y = check_array(Y, ensure_2d=True, copy=True)
#         fitted = None
#         try:
#             while fitted is None:

#                 if self.verbose:
#                     print(f"PDF Extension: {self.pdf_extension}%")

#                 try:
#                     # Initialize RBIG class I
#                     self.rbig_model_X = RBIG(
#                         n_layers=self.n_layers,
#                         rotation_type=self.rotation_type,
#                         pdf_resolution=self.pdf_resolution,
#                         pdf_extension=self.pdf_extension,
#                         verbose=self.verbose,
#                         random_state=self.random_state,
#                         zero_tolerance=self.zero_tolerance,
#                         tolerance=self.tolerance,
#                     )

#                     # fit and transform model to the data
#                     X_transformed = self.rbig_model_X.fit_transform(X)

#                     # Initialize RBIG class II
#                     self.rbig_model_Y = RBIG(
#                         n_layers=self.n_layers,
#                         rotation_type=self.rotation_type,
#                         pdf_resolution=self.pdf_resolution,
#                         pdf_extension=self.pdf_extension,
#                         verbose=self.verbose,
#                         random_state=self.random_state,
#                         zero_tolerance=self.zero_tolerance,
#                         tolerance=self.tolerance,
#                     )

#                     # fit model to the data
#                     Y_transformed = self.rbig_model_Y.fit_transform(Y)

#                     # Stack Data
#                     if self.verbose:
#                         print(X_transformed.shape, Y_transformed.shape)

#                     XY_transformed = np.hstack([X_transformed, Y_transformed])

#                     # Initialize RBIG class I & II
#                     self.rbig_model_XY = RBIG(
#                         n_layers=self.n_layers,
#                         rotation_type=self.rotation_type,
#                         random_state=self.random_state,
#                         zero_tolerance=self.zero_tolerance,
#                         tolerance=self.tolerance,
#                         pdf_resolution=self.pdf_resolution,
#                         pdf_extension=self.pdf_extension,
#                         verbose=self.verbose,
#                     )

#                     # Fit RBIG model to combined dataset
#                     self.rbig_model_XY.fit(XY_transformed)
#                     fitted = True
#                 except:
#                     self.pdf_extension = self.increment * self.pdf_extension

#         except KeyboardInterrupt:
#             print("Interrupted!")

#         return self

#     def mutual_information(self):
#         """Given that the algorithm has been fitted to two datasets, this
#         returns the mutual information between the two multidimensional
#         datasets.

#         Returns
#         -------
#         mutual_info : float
#             The mutual information between the two multidimensional
#             variables.
#         """
#         return self.rbig_model_XY.residual_info.sum()
