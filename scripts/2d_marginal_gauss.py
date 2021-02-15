import sys

sys.path.insert(0, "/home/emmanuel/code/projects/rbig")
sys.path.insert(0, "/Users/eman/Documents/code_projects/rbig")


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from rbig.transform import HistogramGaussianization
import seaborn as sns

plt.style.use(["seaborn-paper", "fivethirtyeight"])

seed = 123
rng = np.random.RandomState(seed=seed)

n_samples = 1_000
# initialize data distribution
data_dist = stats.uniform()

# get some samples
# data = data_dist.rvs(size=n_samples).reshape(-1, 1)

x = np.abs(2 * rng.randn(1, n_samples))
y = np.sin(x) + 0.25 * rng.randn(1, n_samples)
data = np.vstack((x, y)).T


# plt.figure()
# pts = sns.jointplot(x=data[:, 0], y=data[:, 1],)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Original data")
# plt.tight_layout()
# plt.show()

plt.figure()
plt.hist2d(data[:, 0], data[:, 1], bins=100, normed=True, cmap="Blues")
cb = plt.colorbar()
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots()
# ax.scatter(data[:, 0], data[:, 1], s=1)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_title("Original Data")
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(ncols=2)
# ax[0].hist(data[:, 0], bins=100)
# ax[0].set_xlabel("X")
# ax[0].set_ylabel(r"$p_\theta(X)$")
# ax[1].hist(data[:, 1], bins=100)
# ax[1].set_xlabel("Y")
# ax[1].set_ylabel(r"$p_\theta(Y)$")
# plt.tight_layout()
# plt.show()

# =========================
# Marginal Transformation
# =========================

# RBIG Transformation 1
mg_clf = HistogramGaussianization().fit(data)

X_mg = mg_clf.transform(data)

pts = sns.jointplot(x=X_mg[:, 0], y=X_mg[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Transformed data")
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots(ncols=2)
# ax[0].hist(X_mg[:, 0], bins=100)
# ax[0].set_xlabel("X")
# ax[0].set_ylabel(r"$p_\theta(X)$")
# ax[1].hist(X_mg[:, 1], bins=100)
# ax[1].set_xlabel("Y")
# ax[1].set_ylabel(r"$p_\theta(Y)$")
# plt.tight_layout()
# plt.show()


# =========================
# Inverse Transformation
# =========================

X_approx = mg_clf.inverse_transform(X_mg)

pts = sns.jointplot(x=X_approx[:, 0], y=X_approx[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Approximate data")
plt.tight_layout()
plt.show()

# =========================
# Jacobian
# =========================

print("Jacobian")
X_mg_der = mg_clf.log_abs_det_jacobian(data)
print(X_mg_der.min(), X_mg_der.max())

pts = sns.jointplot(x=X_mg_der[:, 0], y=X_mg_der[:, 1])
plt.xlabel("dX")
plt.ylabel("dY")
plt.title("Jacobian")
plt.tight_layout()
plt.show()


# =========================
# Probability Density
# =========================

print("Log Probability")
X_logprob = mg_clf.score_samples(data)

print(X_logprob.min(), X_logprob.max())
fig, ax = plt.subplots()
pts = ax.scatter(data[:, 0], data[:, 1], s=1, c=X_logprob, cmap="Blues")
plt.colorbar(pts)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Transformed data")
plt.tight_layout()
plt.show()

print("Probability")
x_prob = np.exp(X_logprob)
print(x_prob.min(), x_prob.max())
fig, ax = plt.subplots()
pts = ax.scatter(data[:, 0], data[:, 1], s=1, c=x_prob, cmap="Blues")
plt.colorbar(pts)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Transformed data")
plt.tight_layout()
plt.show()

# =========================
# Negative Log-Likelihood
# =========================

print("Log Probability")
x_nll = mg_clf.score(data)
print(x_nll)
