# %%
import sys
from pyprojroot import here

sys.path.append(str(here()))

# RBIG Packages
from rbig.data import ToyData
from rbig.layers import RBIGBlock, RBIGKDEParams, RBIGHistParams, RBIGQuantileParams, RBIGPowerParams
from rbig.models import GaussianizationModel

%load_ext autoreload
%autoreload 2
# %% [markdown]
# Below, we will load some sample data and see how the Gaussianization model can be used.
# 
# %%


