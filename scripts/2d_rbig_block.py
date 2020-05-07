import sys

sys.path.insert(0, "/home/emmanuel/code/projects/rbig")
sys.path.insert(0, "/Users/eman/Documents/code_projects/rbig")


import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy import stats
from rbig.transform import HistogramGaussianization, OrthogonalTransform
import seaborn as sns

plt.style.use(["seaborn-paper"])


seed = 123
rng = np.random.RandomState(seed=seed)

n_samples = 10_000
# initialize data distribution
data_dist = stats.uniform()

# get some samples
# data = data_dist.rvs(size=n_samples).reshape(-1, 1)

x = np.abs(2 * rng.randn(1, n_samples))
y = np.sin(x) + 0.25 * rng.randn(1, n_samples)
data = np.vstack((x, y)).T

pts = sns.jointplot(x=data[:, 0], y=data[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle("Data")
plt.tight_layout()
plt.show()


# =========================
# RBIG Flow
# =========================


# 1. Marginal Gaussianization
mg_clf = HistogramGaussianization().fit(data)

X_mg = mg_clf.transform(data)

pts = sns.jointplot(x=X_mg[:, 0], y=X_mg[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle("Transformed data (MG)")
plt.tight_layout()
plt.show()


# 2. Orthogonal Rotation
rot_clf = OrthogonalTransform().fit(X_mg)

X_rot = rot_clf.transform(X_mg)

pts = sns.jointplot(x=X_rot[:, 0], y=X_rot[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle("Transformed data (MG + R)")
plt.tight_layout()
plt.show()

# =========================
# RBIG Inverse Flow
# =========================


# RBIG - Rotation
X_mg_approx = rot_clf.inverse_transform(X_rot)

pts = sns.jointplot(x=X_mg_approx[:, 0], y=X_mg_approx[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle("Inverse Transformed data (R^-1)")
plt.tight_layout()
plt.show()


# RBIG - Marginal Gaussianization
X_approx = mg_clf.inverse_transform(X_mg_approx)


pts = sns.jointplot(x=X_approx[:, 0], y=X_approx[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Inverse Transformed data (R^-1 + MG^-1)")
plt.tight_layout()
plt.show()


# # =========================
# # RBIG Flow - Jacobian
# # =========================


# 1. Marginal Gaussianization
mg_clf = HistogramGaussianization().fit(data)

X_mg = mg_clf.transform(data)
dX_mg = mg_clf.log_abs_det_jacobian(data)

pts = sns.jointplot(x=dX_mg[:, 0], y=dX_mg[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle("Transformed data (MG) - Jacobian")
plt.tight_layout()
plt.show()


# RBIG - Rotation
rot_clf = OrthogonalTransform().fit(X_mg)

X_rot = rot_clf.transform(X_mg)
dX_rot = rot_clf.log_abs_det_jacobian(X_rot)

pts = sns.jointplot(x=dX_rot[:, 0], y=dX_rot[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle("Transformed data (MG + R) - Jacobian")
plt.tight_layout()
plt.show()


# RBIG - Rotation + MG
dX = dX_rot + dX_mg

pts = sns.jointplot(x=dX[:, 0], y=dX[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle("Transformed data (MG + R) - Jacobian")
plt.tight_layout()
plt.show()


# ==================================
# CALCULATE LOG PROBABILITIES
# ==================================


# 1. Marginal Gaussianization
mg_clf = HistogramGaussianization().fit(data)
X_mg = mg_clf.transform(data)
dX_mg = mg_clf.log_abs_det_jacobian(data)

# RBIG - Rotation
rot_clf = OrthogonalTransform().fit(X_mg)
X_rot = rot_clf.transform(X_mg)
dX_rot = rot_clf.log_abs_det_jacobian(X_rot)

# Calulcate Probability
X_logprob = (stats.norm().logpdf(X_rot) + dX_rot + dX_mg).sum(axis=1)


fig, ax = plt.subplots(ncols=1)
# Scatter Plot
pts = ax.scatter(data[:, 0], data[:, 1], s=1, c=X_logprob, cmap="Blues")
plt.colorbar(pts)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Transformed data, log prob")
plt.show()


fig, ax = plt.subplots(ncols=1)
# Scatter Plot
pts = ax.scatter(
    data[:, 0], data[:, 1], s=1, c=np.exp(X_logprob), cmap="Blues"
)
plt.colorbar(pts)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Transformed data, prob")
plt.show()


# ==================================
#  Negative Log-Likelihood
# ==================================
X_nll = np.mean(X_logprob)
print("NLL:", X_nll)
