import sys

sys.path.insert(0, "/home/emmanuel/code/projects/rbig")
sys.path.insert(0, "/Users/eman/Documents/code_projects/rbig")


from rbig.transform.gaussianization import HistogramGaussianization

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
X_samples = data_dist.rvs(size=(n_samples, 1), random_state=seed)
# X_samples = np.abs(2 * rng.randn(n_samples, 1))
# X_samples = np.sin(X_samples) + 0.25 * rng.randn(n_samples, 1)

print(X_samples.shape)

print("Generated Samples...")

fig, ax = plt.subplots()
ax.hist(X_samples, nbins)
ax.set_xlabel(r"$\mathcal{X}$")
ax.set_ylabel(r"$p_\theta(x)$")
plt.show()

# initialize HistogramClass

histgauss_clf = HistogramGaussianization(nbins=nbins)

# fit to data
histgauss_clf.fit(X_samples)

print("Fit Histogram to Data...")

# ========================
# Transform Data Samples
# ========================
print("Forward Transformer")
# transform data
Xg = histgauss_clf.transform(X_samples)


print(Xg.min(), Xg.max())

# Plot Transform
fig, ax = plt.subplots()
ax.set_title("Forward Transformation")
ax.hist(Xg)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\psi(x)$")
plt.show()


# ========================
# Inverse Transform
# ========================

print("Inverse Transformation")
# transform data
X_gauss = stats.norm(loc=0, scale=1).rvs(size=(n_samples, 1))
X_approx = histgauss_clf.inverse_transform(X_gauss)
# print(X_approx.min(), X_approx.max())


fig, ax = plt.subplots()
ax.set_title("Inverse Transformation")
ax.hist(X_approx, nbins, label="Generated")
ax.hist(X_samples, nbins, label="Real")
ax.set_xlabel(r"$X_\mathcal{G}$")
ax.set_ylabel(r"$\psi^{-1}(x)$")
ax.legend()
plt.tight_layout()
plt.show()

# ========================
# Evaluate Jacobian
# ========================
print("Jacobian")
x_jacobian = histgauss_clf.log_abs_det_jacobian(X_samples)

print(x_jacobian.min(), x_jacobian.max())

fig, ax = plt.subplots()
ax.hist(x_jacobian, nbins, label="Gaussianized(X)")
ax.set_title("X Jacobian")
ax.set_xlabel(r"$\mathcal{X}$")
ax.set_ylabel(r"$p_\theta(x)$")
ax.legend()
plt.show()

# ========================
# Evaluate Probability
# ========================
print("Log Probability")
x_logprob = histgauss_clf.score_samples(X_samples)

print(x_logprob.min(), x_logprob.max())

fig, ax = plt.subplots()
ax.hist(x_logprob, nbins, label="Log Probabilty")
ax.hist(data_dist.logpdf(X_samples), nbins, label="Data")
ax.set_xlabel(r"$\mathcal{X}$")
ax.set_ylabel(r"$p_\theta(x)$")
ax.legend()
plt.show()

# ==================================
# Evaluate Negative Log-Likelihood
# ==================================
x_prob = histgauss_clf.score(X_samples)
data_prob = data_dist.logpdf(X_samples).mean()
print(x_prob)
print(data_prob)
