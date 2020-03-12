import sys

sys.path.insert(0, "/home/emmanuel/code/projects/rbig")


import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy import stats
from rbig.transform import MarginalGaussianization, OrthogonalTransform
import seaborn as sns

plt.style.use(["seaborn-paper", "fivethirtyeight"])
norm = colors.Normalize(vmin=0, vmax=1.0, clip=True)

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


# =========================
# RBIG Flow
# =========================

# RBIG - Rotation (initial)
rot_init_clf = OrthogonalTransform().fit(data)

Xinit = rot_init_clf.transform(data)

# RBIG - Marginal Gaussianization
mg_clf = MarginalGaussianization().fit(Xinit)

Xmg = mg_clf.transform(Xinit)


# RBIG - Rotation
rot_clf = OrthogonalTransform().fit(Xmg)

Xrot = rot_clf.transform(Xmg)

pts = sns.jointplot(x=Xrot[:, 0], y=Xrot[:, 1],)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Transformed data")
plt.tight_layout()
plt.show()

# =========================
# RBIG Inverse Flow
# =========================


# RBIG - Rotation
Xmg_approx = rot_clf.inverse_transform(Xrot)

# RBIG - Marginal Gaussianization
Xinit_approx = mg_clf.inverse_transform(Xmg_approx)

# RBIG - Rotation
X_approx = rot_init_clf.inverse_transform(Xinit_approx)


# RBIG - Rotation
rot_clf = OrthogonalTransform().fit(Xmg)

Xrot = rot_init_clf.transform(Xmg)

pts = sns.jointplot(x=X_approx[:, 0], y=X_approx[:, 1],)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Transformed data")
plt.tight_layout()
plt.show()


# # =========================
# # RBIG Flow - Jacobian
# # =========================

# # RBIG - Rotation (initial)
# # rot_init_clf = OrthogonalTransform().fit(data)

# # Xinit, dXinit = rot_init_clf.transform(data, return_jacobian=True)

# # RBIG - Marginal Gaussianization
# mg_clf = MarginalGaussianization().fit(data)

# Xmg, dXmg = mg_clf.transform(data, return_jacobian=True)
# print(dXmg.min(), dXmg.max())


# # RBIG - Rotation
# rot_clf = OrthogonalTransform().fit(Xmg)

# Xrot, dXrot = rot_clf.transform(Xmg, return_jacobian=True)
# print(dXrot.min(), dXrot.max())


# # CALCULATE PROBABILITIES
# x_lprob = (stats.norm().logpdf(Xrot) + dXmg + dXrot).sum(axis=1)
# print(x_lprob.min(), x_lprob.max())
# x_prob = np.exp(x_lprob)

# fig, ax = plt.subplots()
# pts = ax.scatter(data[:, 0], data[:, 1], s=1, c=x_lprob, cmap="Blues", norm=norm)
# plt.colorbar(pts)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_title("Transformed data")
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots()
# pts = ax.scatter(data[:, 0], data[:, 1], s=1, c=x_prob, cmap="Blues", norm=norm)
# plt.colorbar(pts)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_title("Transformed data")
# plt.tight_layout()
# plt.show()


# =========================
# RBIG 2Flow - Jacobian
# =========================

# RBIG - Rotation (initial)
# rot_init_clf = OrthogonalTransform().fit(data)

# Xinit, dXinit = rot_init_clf.transform(data, return_jacobian=True)

# RBIG - Marginal Gaussianization
mg_clf = MarginalGaussianization().fit(data)

Xmg, dXmg = mg_clf.transform(data, return_jacobian=True)
print(dXmg.min(), dXmg.max())


# RBIG - Rotation
rot_clf = OrthogonalTransform().fit(Xmg)

Xrot, dXrot = rot_clf.transform(Xmg, return_jacobian=True)
print(dXrot.min(), dXrot.max())

# RBIG - Marginal Gaussianization
mg2_clf = MarginalGaussianization().fit(data)

Xmg2, dXmg2 = mg2_clf.transform(Xrot, return_jacobian=True)
print(dXmg2.min(), dXmg2.max())


# CALCULATE PROBABILITIES
x_lprob = (stats.norm().logpdf(Xmg2) + dXmg + dXmg2).sum(axis=1)
print(x_lprob.min(), x_lprob.max())
x_prob = np.exp(x_lprob)


fig, ax = plt.subplots(nrows=2)
# Scatter Plot
pts = ax[0].scatter(data[:, 0], data[:, 1], s=1, c=x_lprob, cmap="Blues")
plt.colorbar(pts)
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].set_title("Transformed data")
# Histogram plot
ax[1].hist(x_lprob, 100)
ax[1].set_title("Histogram of Derivatives")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=2)
# Scatter Plot
pts = ax[0].scatter(data[:, 0], data[:, 1], s=1, c=x_prob, cmap="Blues")
plt.colorbar(pts)
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].set_title("Transformed data")
# Histogram plot
ax[1].hist(x_prob, 100)
ax[1].set_title("Histogram of Derivatives")
plt.tight_layout()
plt.show()
