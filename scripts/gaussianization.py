import sys

sys.path.insert(0, "/home/emmanuel/code/projects/rbig")


from rbig.transform import HistogramGaussianization

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


plt.style.use(["seaborn-paper"])

seed = 123
n_samples = 1_000
a = 10
nbins = int(np.sqrt(n_samples))

# initialize data distribution
rng = np.random.RandomState(seed=seed)
data_dist = stats.gamma(a=a)

# get some samples
# X_samples = data_dist.rvs(size=n_samples, random_state=seed)
X_samples = np.abs(2 * rng.randn(1, n_samples))
X_samples = np.sin(X_samples) + 0.25 * rng.randn(1, n_samples)

print("Generated Samples...")

# fig, ax = plt.subplots()
# ax.hist(X_samples, nbins)
# ax.set_xlabel(r"$\mathcal{X}$")
# ax.set_ylabel(r"$p_\theta(x)$")
# plt.show()

# initialize HistogramClass

histgauss_clf = HistogramGaussianization(nbins=nbins, log=False)

# fit to data
histgauss_clf.fit(X_samples)

print("Fit Histogram to Data...")

# ========================
# Transform Data Samples
# ========================

# transform data
Xg, Xg_der = histgauss_clf.transform(X_samples, return_jacobian=True)

print("Transformed data (w. Jacobian)")
print(Xg.min(), Xg.max())

# Plot Transform
fig, ax = plt.subplots()
ax.hist(Xg)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\psi(x)$")
plt.show()


# Log probability
Xg_lder = histgauss_clf.abs_det_jacobian(X_samples, log=True)

print("Transformed data (w. log Jacobian)")

# Plot the derivative
# fig, ax = plt.subplots(ncols=2)
# ax[0].hist(Xg_der, nbins)
# ax[0].set_xlabel(r"$x$")
# ax[0].set_ylabel(r"$\psi'(x)$")
# ax[1].hist(Xg_lder, nbins)
# ax[1].set_xlabel(r"$x$")
# ax[1].set_ylabel(r"$\log\psi'(x)$")
# plt.tight_layout()
# plt.show()


# ========================
# Inverse Transform
# ========================

# transform data
X_gauss = stats.norm(loc=0, scale=1).rvs(n_samples)
X_approx = histgauss_clf.inverse_transform(X_gauss)

# fig, ax = plt.subplots()
# ax.hist(X_approx, nbins, label="Generated")
# ax.hist(X_samples, nbins, label="Real")
# ax.set_xlabel(r"$X_\mathcal{G}$")
# ax.set_ylabel(r"$\psi^{-1}(x)$")
# ax.legend()
# plt.tight_layout()
# plt.show()


# ========================
# Evaluate Probability
# ========================
x_lprob = stats.norm(loc=0, scale=1).logpdf(Xg) + Xg_lder
x_prob = stats.norm(loc=0, scale=1).pdf(Xg) * Xg_der

# Plot the derivative
fig, ax = plt.subplots(ncols=2)
ax[0].hist(x_prob, nbins, label="Approximate")
ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$\psi'(x)$")
# ax[1].hist(data_dist.pdf(X_samples), nbins, label="Real")
# ax[1].set_xlabel(r"$x$")
# ax[1].set_ylabel(r"$\psi'(x)$")
plt.tight_layout()
plt.show()

# Plot the log derivative
fig, ax = plt.subplots(ncols=2)
ax[0].hist(x_lprob, nbins, label="Approximate")
ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$\log\psi'(x)$")
# ax[1].hist(data_dist.logpdf(X_samples), nbins, label="Real")
# ax[1].set_xlabel(r"$x$")
# ax[1].set_ylabel(r"$\log\psi'(x)$")
plt.tight_layout()
plt.show()

# # ========================
# # Evaluate Probability
# # ========================
# x_prob = histgauss_clf.pdf(X_samples)

# fig, ax = plt.subplots()
# ax.hist(x_prob, nbins, label="Transformed")
# ax.hist(data_dist.pdf(X_samples), nbins, label="Data")
# ax.set_xlabel(r"$\mathcal{X}$")
# ax.set_ylabel(r"$p_\theta(x)$")
# ax.legend()
# plt.show()


# # ========================
# # Evaluate Log Probability
# # ========================
# x_prob = histgauss_clf.logpdf(X_samples)
# data_prob = data_dist.logpdf(X_samples)

# fig, ax = plt.subplots()
# ax.hist(x_prob, nbins, label="Transformed")
# ax.hist(data_prob, nbins, label="Data")
# ax.set_xlabel(r"$\mathcal{X}$")
# ax.set_ylabel(r"$p_\theta(x)$")
# ax.legend()
# plt.show()

# ========================
# Evaluate Score
# ========================
x_prob = histgauss_clf.score(X_samples)
# data_prob = data_dist.logpdf(X_samples).mean()
print(x_prob)
# print(data_prob)
