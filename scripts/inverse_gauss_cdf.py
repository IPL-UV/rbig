import sys

sys.path.insert(0, "/home/emmanuel/code/projects/rbig")
sys.path.insert(0, "/Users/eman/Documents/code_projects/rbig")

from rbig.transform import InverseGaussCDF

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

seed = 123
n_samples = 1_000
a = 10
nbins = 100

# initialize data distribution
data_dist = stats.uniform()

# get some samples
Xu_samples = data_dist.rvs(size=n_samples)
# X_samples = np.array([1.0, 2.0, 1.0])

# initialize HistogramClass
icdf_clf = InverseGaussCDF()

# fit to data
icdf_clf.fit(Xu_samples)

# ========================
# Transform Data Samples
# ========================

# transform data
Xg = icdf_clf.transform(Xu_samples)

fig, ax = plt.subplots()
ax.hist(Xg, nbins)
ax.set_xlabel(r"$F(x)$")
ax.set_ylabel(r"$p(u)$")
plt.show()

# ========================
# Generate Uniform Samples
# ========================
Xu_approx = icdf_clf.inverse_transform(Xg)

fig, ax = plt.subplots()
ax.hist(Xu_approx, nbins)
ax.set_title("Generate Samples (transformed)")
ax.set_xlabel(r"$F^{-1}(u)$")
ax.set_ylabel(r"$p(x_d)$")
plt.show()

# ========================
# Generate Uniform Samples
# ========================
X_approx = icdf_clf.sample(1000)

fig, ax = plt.subplots()
ax.hist(X_approx, nbins)
ax.set_title("Generate Samples (from Function)")
ax.set_xlabel(r"$F^{-1}(\hat{u})$")
ax.set_ylabel(r"$p(x_d)$")
plt.show()


# ========================
# Evaluate Jacobian
# ========================

x_der = icdf_clf.abs_det_jacobian(Xu_samples)
fig, ax = plt.subplots()

ax.hist(x_der, nbins)
ax.set_title("Generate Samples (transformed)")
ax.set_xlabel(r"$F^{-1}(u)$")
ax.set_ylabel(r"$p(x_d)$")
plt.show()


# ========================
# Evaluate Log Probability
# ========================
x_der = icdf_clf.log_abs_det_jacobian(Xu_samples)

fig, ax = plt.subplots()
ax.hist(x_der, nbins)
ax.set_title("Generate Samples (transformed)")
ax.set_xlabel(r"$F^{-1}(u)$")
ax.set_ylabel(r"$\log p(x_d)$")
plt.show()
