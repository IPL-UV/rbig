import sys

sys.path.insert(0, "/home/emmanuel/code/projects/rbig")
sys.path.insert(0, "/Users/eman/Documents/code_projects/rbig")


from rbig.density import Histogram

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

seed = 123
n_samples = 1_000
a = 10

# initialize data distribution
data_dist = stats.gamma(a=a)

# get some samples
X_samples = data_dist.rvs(size=n_samples)
# X_samples = np.array([1.0, 2.0, 1.0])

# initialize HistogramClass
nbins = int(np.sqrt(n_samples))
bounds = None
hist_clf = Histogram(nbins=nbins)

# fit to data
hist_clf.fit(X_samples)

# ========================
# Transform Data Samples
# ========================

# transform data
Xu = hist_clf.transform(X_samples)

fig, ax = plt.subplots()
ax.hist(Xu, nbins)
ax.set_xlabel(r"$F(x)$")
ax.set_ylabel(r"$p(u)$")
plt.show()

# ========================
# Generate Uniform Samples
# ========================
u_samples = stats.uniform().rvs(size=1000)
X_approx = hist_clf.inverse_transform(u_samples)

fig, ax = plt.subplots()
ax.hist(X_approx, nbins)
ax.set_title("Generate Samples (transformed)")
ax.set_xlabel(r"$F^{-1}(u)$")
ax.set_ylabel(r"$p(x_d)$")
plt.show()

# ========================
# Generate Uniform Samples
# ========================
X_approx = hist_clf.sample(1000)

fig, ax = plt.subplots()
ax.hist(X_approx, nbins)
ax.set_title("Generate Samples (from Function)")
ax.set_xlabel(r"$F^{-1}(u)$")
ax.set_ylabel(r"$p(x_d)$")
plt.show()


# ========================
# Evaluate Probability
# ========================
print(X_samples[:10].shape)
x_prob = hist_clf.pdf(X_samples[:10])
data_prob = data_dist.pdf(X_samples[:10])
print(x_prob)
print(data_prob)

# ========================
# Evaluate Log Probability
# ========================
print(X_samples[:10].shape)
x_prob = hist_clf.score_samples(X_samples[:10])
data_prob = data_dist.logpdf(X_samples[:10])
print(x_prob)
print(data_prob)


# ========================
# Log-Likelihood
# ========================
x_score = hist_clf.score(X_samples)
data_score = np.log(data_dist.pdf(X_samples)).mean()
print(x_score, data_score)
