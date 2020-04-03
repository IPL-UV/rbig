#%%
import numpy as np
import scipy

#%%
seed = 123
rng = np.random.RandomState(seed=seed)

num_samples = 1000
x = np.abs(2 * rng.randn(1, num_samples))
y = np.sin(x) + 0.25 * rng.randn(1, num_samples)

#%%
hist, bin_edges = np.histogram(x, bins=10, range=0.1)
