# Code Plan



Last Update: 26-09-2017

## Table of Contents



---
## Overview
<a id='overview'></a>

### Parameters

* precision (default = 1000)
	- The number of points for the marginal PDF estimation.
- porc (default = 10)
	- the extra domain percentage
- transformation (default = 'pca')
	- linear transformation applied (ica, pca, rnd, rp)
* number of layers (default = 1000)
	- number of layers
- tolerance for samples
- tolerance for dimensions
- random_state

### Attributes

* number of samples
* dataset
* precision
* porc (?)
* transformation (PCA, ICA, RP)
* number of layers


### Methods

* run demo
* generate data
* plot toy data
* plot gaussianization
* fit/fit_transform
	- rbig
	- rbig apply
- transform
	- rbig apply
	- inverse rbig
- inverse_transform
    - inverse rbig

### hidden methods

* marginal gaussianization
* inverse marginal gaussianization
* marginal uniformization
* inverse marginal uniformization
* make monotonic
* entropy
	- ele w/ miller-maddow correction
* information reduction
* rotation
	- random rotations
	- ICA
	- PCA
*

---

<div id=
## Methods <a name="methods"></a>
