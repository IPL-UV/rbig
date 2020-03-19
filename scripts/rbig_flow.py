import sys

sys.path.insert(0, "/home/emmanuel/code/projects/rbig")
sys.path.insert(0, "/Users/eman/Documents/code_projects/rbig")


import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy import stats
from rbig.transform import MarginalGaussianization, OrthogonalTransform
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

pts = sns.jointplot(x=data[:, 0], y=data[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Transformed data")
plt.tight_layout()
plt.show()


# # =========================
# # RBIG Flow
# # =========================


# RBIG - Marginal Gaussianization
mg_clf = MarginalGaussianization().fit(data)

Xmg = mg_clf.transform(data)

pts = sns.jointplot(x=Xmg[:, 0], y=Xmg[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Transformed data")
plt.tight_layout()
plt.show()


# RBIG - Rotation
rot_clf = OrthogonalTransform().fit(Xmg)

Xrot = rot_clf.transform(Xmg)

pts = sns.jointplot(x=Xrot[:, 0], y=Xrot[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Transformed data")
plt.tight_layout()
plt.show()

# # =========================
# # RBIG Inverse Flow
# # =========================


# RBIG - Rotation
Xmg_approx = rot_clf.inverse_transform(Xrot)

pts = sns.jointplot(x=Xmg_approx[:, 0], y=Xmg_approx[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Transformed data")
plt.tight_layout()
plt.show()


# RBIG - Marginal Gaussianization
X_approx = mg_clf.inverse_transform(Xmg_approx)


pts = sns.jointplot(x=X_approx[:, 0], y=X_approx[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Transformed data")
plt.tight_layout()
plt.show()


# =========================
# RBIG Flow - Jacobian
# =========================

# RBIG - Marginal Gaussianization
mg_clf = MarginalGaussianization().fit(data)

Xmg, dXmg = mg_clf.transform(data, return_jacobian=True)
print(dXmg.min(), dXmg.max())
# dXmg = mg_clf.abs_det_jacobian(data, log=True)
# print(dXmg.min(), dXmg.max())

# RBIG - Rotation
rot_clf = OrthogonalTransform(rotation="random_o").fit(Xmg)

Xrot, dXrot = rot_clf.transform(Xmg, return_jacobian=True)
print(dXrot.min(), dXrot.max())


# CALCULATE PROBABILITIES
x_lprob = (stats.norm().logpdf(Xrot) + dXmg + dXrot).sum(axis=1)

print("Score:", x_lprob.mean(), np.exp(x_lprob.mean()))

print(x_lprob.min(), x_lprob.max())
x_prob = np.exp(x_lprob)
print(x_prob.min(), x_prob.max())

norm = None
fig, ax = plt.subplots()
pts = ax.scatter(data[:, 0], data[:, 1], s=1, c=x_lprob, cmap="Blues", norm=norm)
plt.colorbar(pts)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Transformed data")
plt.tight_layout()
plt.show()

norm = colors.Normalize(vmin=0, vmax=1.0, clip=True)
fig, ax = plt.subplots()
pts = ax.scatter(data[:, 0], data[:, 1], s=1, c=x_prob, cmap="Blues", norm=norm)
plt.colorbar(pts)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Transformed data")
plt.tight_layout()
plt.show()


# =========================
# RBIG 2Flow - Jacobian
# =========================

# RBIG - Marginal Gaussianization
mg_clf = MarginalGaussianization().fit(data)

Xmg, dXmg = mg_clf.transform(data, return_jacobian=True)
print(dXmg.min(), dXmg.max())


# RBIG - Rotation
rot_clf1 = OrthogonalTransform().fit(Xmg)

Xrot, dXrot = rot_clf1.transform(Xmg, return_jacobian=True)
print(dXrot.min(), dXrot.max())

# RBIG - Marginal Gaussianization
mg2_clf = MarginalGaussianization().fit(data)

Xmg2, dXmg2 = mg2_clf.transform(Xrot, return_jacobian=True)
print(dXmg2.min(), dXmg2.max())

# RBIG - Rotation
rot_clf2 = OrthogonalTransform().fit(Xmg2)

Xrot2, dXrot2 = rot_clf2.transform(Xmg, return_jacobian=True)
print(dXrot2.min(), dXrot2.max())


# CALCULATE PROBABILITIES
x_lprob = (stats.norm().logpdf(Xmg2) + dXmg + dXrot + dXmg2 + dXrot2).sum(axis=1)
print("Score:", x_lprob.mean(), np.exp(x_lprob.mean()))
print(x_lprob.min(), x_lprob.max())


fig, ax = plt.subplots(ncols=2)
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

x_prob = np.exp(x_lprob)
print(x_prob.shape)
print(x_prob.min(), x_prob.max())
higher_probs = np.argwhere(x_prob < 1.0)
x_prob = x_prob[higher_probs]
x_plot = data[:, 0][higher_probs]
y_plot = data[:, 1][higher_probs]
print(x_prob.min(), x_prob.max())
print(x_prob.shape)

fig, ax = plt.subplots(ncols=2)
# Scatter Plot
pts = ax[0].scatter(x_plot, y_plot, s=1, c=x_prob, cmap="Blues")
plt.colorbar(pts)
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].set_title("Transformed data")
# Histogram plot
ax[1].hist(x_prob, 100)
ax[1].set_title("Histogram of Derivatives")
plt.tight_layout()
plt.show()
