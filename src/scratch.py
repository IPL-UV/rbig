
# def _calculate_tolerance(self, n_errors=1000):

#     ee = np.zeros(n_errors)

#     for rep in np.arange(0, n_errors):
#         # HX
#         [counts, centers] = np.histogram(a=np.random.randn(1, np.round(self.n_samples)),
#                                             bins=np.int(np.round(np.sqrt(self.n_samples))))

#         delta = centers[1] - centers[0]

#         constant = 0.5 * (np.sum(counts[counts > 0]) - 1) / np.sum(counts)
#         counts = counts / np.sum(counts)
#         hx = -np.sum(counts[counts != 0] * np.log(counts[counts != 0])) + constant
#         hx += delta

#         # HY
#         [counts, centers] = np.histogram(a=np.random.randn(1, np.round(self.n_samples)),
#                                             bins=np.int(np.round(np.sqrt(self.n_samples))))
#         #     centers = centers[:-1] = np.diff(centers)/2
#         delta = centers[1] - centers[0]

#         constant = 0.5 * (np.sum(counts[counts > 0]) - 1) / np.sum(counts)
#         counts = counts / np.sum(counts)
#         hy = -np.sum(counts[counts != 0] * np.log(counts[counts != 0])) + constant
#         hy += delta

#         ee[rep] = hy - hx

#     self.tol_dimensions = np.mean(ee)
#     self.tol_samples = np.std(ee)

#     return None

# def _marginal_gaussianization(self, data, precision=None):

        if precision is None:
            precision = np.round(np.sqrt(len(data)) + 1)

        data_uniform, params = marginal_uniformization(data, self.domain_prnt, precision)

        return norm.ppf(data_uniform), params



# def rbig_apply(self, data, transform):
#
#     logging.info('Running: rbig apply algorithm ...')
#     # get dimensions of information
#     # get size of the X data
#     n_samples, n_dimensions = data.shape
#
#     precision = transform['precision']
#
#     # for computational speed
#     logging.info('Modifying data for computational speed ...')
#     n_dimensions = 500000
#     mmod = np.mod(n_samples, n_dimensions)
#     fflor = np.floor(n_samples / n_dimensions)
#
#     # initialize the data
#     data_transformed = np.zeros(data.shape)
#
#     for j_sample in np.arange(0, fflor * n_dimensions, n_dimensions):
#
#         data_0 = data[j_sample:j_sample + n_dimensions, :]
#
#         for i_sample in np.range(0, len(transform['data'])):
#             data_0 = self._marginal_gaussianization(data_0[:, n_dimensions],
#                                               transform['data'][i_sample],
#                                               precision)
#
#         V = transform['v'][n_samples]
#         data_0 = V * data_0
#
#     data_transformed[j_sample:j_sample + n_dimensions] = data_0
#
#     if mmod > 0:
#         data_0 = data[fflor * n_dimensions:]
#
#         for i_sample in np.arange(0, len(transform['data'])):
#             data_0 = self._marginal_gaussianization(data_0[:, n_dimensions],
#                                               transform['data'][i_sample],
#                                               precision)
#
#         V = transform['v'][n_samples]
#         data_0 = V * data_0
#
#     data_transformed[j_sample:j_sample + n_dimensions] = data_0
#
#     return data_transformed

# def inv_rbig_apply(self, data, transform):
#
#     precision = transform.precision[0]
#     n_dimensions = data.shape
#     data_0 = data
#
#     for i_sample in np.arange(0, transform.shape[0])[-1::]:


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


def marginal_uniformization(data, domain_prnt=None, precision=None):

    if precision is None:
        precision = 1000

    params = {}

    # make aux probs and bin edges
    probability_aux = (domain_prnt / 100) * np.abs(data.max() - data.min())
    bin_edges_aux = np.linspace(data.min(), data.max(),
                        num=2 * np.sqrt(len(data)) + 1)

    bin_edges = 0.5 * (bin_edges_aux[:-1]+ bin_edges_aux[1:])

    [hist, bin_edges] = np.histogram(a=data, bins=bin_edges)

    # get bin centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    bin_delta = bin_centers[2] - bin_centers[1]

    bin_center_ant = np.hstack((bin_centers[0]- bin_delta,
                                bin_centers,
                                bin_centers[-1] + bin_delta))
    hist_ant = np.hstack((0,
                          hist / (np.sum(hist) * (bin_centers[3] - bin_centers[2])), 0))

    params['bin_center_ant'] = bin_center_ant
    params['prob_ant'] = hist_ant

    C = np.cumsum(hist)
    N = C.max()

    C = (1 - 1/N) * C / N

    bin_incr = (bin_edges[1] - bin_edges[0]) / 2

    bin_centers = np.hstack((data.min() - probability_aux,
                             data.min(),
                             bin_centers[:-1] + bin_incr,
                             data.max(),
                             probability_aux + bin_incr))

    C = np.hstack((0, 1 / N, C, 1))

    Range_2 = np.linspace(bin_centers[0], bin_centers[-1], num=precision)

    C_2 = make_monotonic(np.interp( Range_2, bin_centers, C))
    C_2 = C_2 / C_2.max()
    x_lin = np.interp(data, Range_2, C_2)

    params['C_2'] = C_2
    params['bin_centers'] = bin_centers

    return x_lin, params


def make_monotonic(f):


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

    fn = f
    for nn in np.arange(2, len(fn)):
        if fn[nn] <= fn[nn - 1]:
            if abs(fn[nn - 1]) > 1e-14:
                fn[nn] = fn[nn - 1] + 1.0e-14
            elif fn[nn - 1] == 0:
                fn[nn] = 1e-80
            else:
                fn[nn] == fn[nn - 1] + 10 ** (np.log(abs(fn[nn - 1])))

    return fn