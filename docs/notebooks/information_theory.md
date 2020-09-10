# Information Theory Measures w/ RBIG


```python
import sys

# MacOS
sys.path.insert(0, '/Users/eman/Documents/code_projects/rbig/')
sys.path.insert(0, '/home/emmanuel/code/py_packages/py_rbig/src')

# ERC server
sys.path.insert(0, '/home/emmanuel/code/rbig/')


import numpy as np
import warnings
from time import time
from rbig.rbig import RBIGKLD, RBIG, RBIGMI, entropy_marginal
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

warnings.filterwarnings('ignore') # get rid of annoying warnings

%load_ext autoreload
%autoreload 2
```

---
## Total Correlation


```python
#Parameters
n_samples = 10000
d_dimensions = 10

seed = 123

rng = check_random_state(seed)
```

#### Sample Data


```python
# Generate random normal data
data_original = rng.randn(n_samples, d_dimensions)

# Generate random Data
A = rng.rand(d_dimensions, d_dimensions)

data = data_original @ A

# covariance matrix
C = A.T @ A
vv = np.diag(C)
```

#### Calculate Total Correlation


```python
tc_original = np.log(np.sqrt(vv)).sum() - 0.5 * np.log(np.linalg.det(C))

print(f"TC: {tc_original:.4f}")
```

    TC: 9.9326


### RBIG - TC


```python
%%time 
n_layers = 10000
rotation_type = 'PCA'
random_state = 0
zero_tolerance = 60
pdf_extension = 10
pdf_resolution = None
tolerance = None

# Initialize RBIG class
tc_rbig_model = RBIG(n_layers=n_layers, 
                  rotation_type=rotation_type, 
                  random_state=random_state, 
                  zero_tolerance=zero_tolerance,
                  tolerance=tolerance,
                  pdf_extension=pdf_extension,
                  pdf_resolution=pdf_resolution)

# fit model to the data
tc_rbig_model.fit(data);
```

    CPU times: user 1min 19s, sys: 64.4 ms, total: 1min 19s
    Wall time: 3.01 s



```python
tc_rbig = tc_rbig_model.mutual_information * np.log(2)
print(f"TC (RBIG): {tc_rbig:.4f}")
print(f"TC: {tc_original:.4f}")
```

    TC (RBIG): 9.9398
    TC: 9.9326


---
## Entropy

#### Sample Data


```python
#Parameters
n_samples = 5000
d_dimensions = 10

seed = 123

rng = check_random_state(seed)

# Generate random normal data
data_original = rng.randn(n_samples, d_dimensions)

# Generate random Data
A = rng.rand(d_dimensions, d_dimensions)

data = data_original @ A

```

#### Calculate Entropy


```python
Hx = entropy_marginal(data)

H_original = Hx.sum() + np.log2(np.abs(np.linalg.det(A)))

H_original *= np.log(2)

print(f"H: {H_original:.4f}")
```

    H: 16.4355


### Entropy RBIG


```python
%%time 
n_layers = 10000
rotation_type = 'PCA'
random_state = 0
zero_tolerance = 60
pdf_extension = None
pdf_resolution = None
tolerance = None

# Initialize RBIG class
ent_rbig_model = RBIG(n_layers=n_layers, 
                  rotation_type=rotation_type, 
                  random_state=random_state, 
                  zero_tolerance=zero_tolerance,
                  tolerance=tolerance)

# fit model to the data
ent_rbig_model.fit(data);
```

    CPU times: user 53.1 s, sys: 9.81 ms, total: 53.1 s
    Wall time: 1.9 s



```python
H_rbig = ent_rbig_model.entropy(correction=True) * np.log(2)
print(f"Entropy (RBIG): {H_rbig:.4f}")
print(f"Entropy: {H_original:.4f}")
```

    Entropy (RBIG): 10.6551
    Entropy: 16.4355


---
## Mutual Information

#### Sample Data


```python
#Parameters
n_samples = 10000
d_dimensions = 10

seed = 123

rng = check_random_state(seed)

# Generate random Data
A = rng.rand(2 * d_dimensions, 2 * d_dimensions)

# Covariance Matrix
C = A @ A.T
mu = np.zeros((2 * d_dimensions))

dat_all = rng.multivariate_normal(mu, C, n_samples)

CX = C[:d_dimensions, :d_dimensions]
CY = C[d_dimensions:, d_dimensions:]

X = dat_all[:, :d_dimensions]
Y = dat_all[:, d_dimensions:]
```

#### Calculate Mutual Information


```python
H_X = 0.5 * np.log(2 * np.pi * np.exp(1) * np.abs(np.linalg.det(CX)))
H_Y = 0.5 * np.log(2 * np.pi * np.exp(1) * np.abs(np.linalg.det(CY)))
H = 0.5 * np.log(2 * np.pi * np.exp(1) * np.abs(np.linalg.det(C)))

mi_original = H_X + H_Y - H
mi_original *= np.log(2)

print(f"MI: {mi_original:.4f}")
```

    MI: 8.0713


### RBIG - Mutual Information


```python
%%time 
n_layers = 10000
rotation_type = 'PCA'
random_state = 0
zero_tolerance = 60
tolerance = None

# Initialize RBIG class
rbig_model = RBIGMI(n_layers=n_layers, 
                  rotation_type=rotation_type, 
                  random_state=random_state, 
                  zero_tolerance=zero_tolerance,
                  tolerance=tolerance)

# fit model to the data
rbig_model.fit(X, Y);
```

    CPU times: user 5min 37s, sys: 103 ms, total: 5min 38s
    Wall time: 12.1 s



```python
H_rbig = rbig_model.mutual_information() * np.log(2)

print(f"MI (RBIG): {H_rbig:.4f}")
print(f"MI: {mi_original:.4f}")
```

    MI (RBIG): 9.0746
    MI: 8.0713


---
## Kullback-Leibler Divergence (KLD)

#### Sample Data


```python
#Parameters
n_samples = 10000
d_dimensions = 10
mu = 0.4          # how different the distributions are

seed = 123

rng = check_random_state(seed)

# Generate random Data
A = rng.rand(d_dimensions, d_dimensions)

# covariance matrix
cov = A @ A.T

# Normalize cov mat
cov = A / A.max()

# create covariance matrices for x and y
cov_x = np.eye(d_dimensions)
cov_y = cov_x.copy()

mu_x = np.zeros(d_dimensions) + mu
mu_y = np.zeros(d_dimensions)

# generate multivariate gaussian data
X = rng.multivariate_normal(mu_x, cov_x, n_samples)
Y = rng.multivariate_normal(mu_y, cov_y, n_samples)

```

#### Calculate KLD


```python
kld_original = 0.5 * ((mu_y - mu_x) @ np.linalg.inv(cov_y) @ (mu_y - mu_x).T +
                      np.trace(np.linalg.inv(cov_y) @ cov_x) -
                      np.log(np.linalg.det(cov_x) / np.linalg.det(cov_y)) - d_dimensions)

print(f'KLD: {kld_original:.4f}')
```

    KLD: 0.8000


### RBIG - KLD


```python
X.min(), X.max()
```




    (-4.006934109277744, 4.585027222023813)




```python
Y.min(), Y.max()
```




    (-4.607129910785054, 4.299322691460413)




```python
%%time

n_layers = 100000
rotation_type = 'PCA'
random_state = 0
zero_tolerance = 60
tolerance = None
pdf_extension = 10
pdf_resolution = None
verbose = 0

# Initialize RBIG class
kld_rbig_model = RBIGKLD(n_layers=n_layers, 
                  rotation_type=rotation_type, 
                  random_state=random_state, 
                  zero_tolerance=zero_tolerance,
                  tolerance=tolerance,
                     pdf_resolution=pdf_resolution,
                    pdf_extension=pdf_extension,
                    verbose=verbose)

# fit model to the data
kld_rbig_model.fit(X, Y);
```

    CPU times: user 5min 46s, sys: 10.9 ms, total: 5min 46s
    Wall time: 12.4 s



```python
# Save KLD value to data structure
kld_rbig= kld_rbig_model.kld*np.log(2)

print(f'KLD (RBIG): {kld_rbig:.4f}')
print(f'KLD: {kld_original:.4f}')
```

    KLD (RBIG): 0.8349
    KLD: 0.8000



```python

```
