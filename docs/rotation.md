# Rotation

#### Initializing

We have a intialization step where we compute $\mathbf W$ (the transformation matrix). This will be in the fit method. We will need the data because some transformations depend on $\mathbf x$ like the PCA and the ICA.

```python
class LinearTransform(BaseEstimator, TransformMixing):
    """
    Parameters
    ----------

    basis : 
    """
    def __init__(self, basis='PCA', conv=16):
        self.basis = basis
        self.conv = conv

    def fit(self, data):
        """
        Computes the inverse transformation of 
                z = W x

        Parameters
        ----------
        data : array, (n_samples x Dimensions)
        """

        # Check the data

        # Implement the transformation
        if basis.upper() == 'PCA':
            ...
        elif basis.upper() == 'ICA':
            ...
        elif basis.lower() == 'random':
            ...
        elif basis.lower() == 'conv':
            ...
        elif basis.upper() == 'dct':
            ...
        else:
            Raise ValueError('...')
        
        # Save the transformation matrix
        self.W = ...

        return self
```

#### Transformation

We have a transformation step:

$$\mathbf{z=W\cdot x}$$

where:
* $\mathbf W$ is the transformation
* $\mathbf x$ is the input data
* $\mathbf y$ is the final transformation

```python
def transform(self, data):
    """
    Computes the inverse transformation of 
            z = W x

    Parameters
    ----------
    data : array, (n_samples x Dimensions)
    """
    return data @ self.W
```

#### Inverse Transformation

We also can apply an inverse transform.

```python
def inverse(self, data):
    """
    Computes the inverse transformation of 
    z = W^-1 x

    Parameters
    ----------
    data : array, (n_samples x Dimensions)

    Returns
    -------
    
    """
    return data @ np.linalg.inv(self.W)
```

#### Jacobian

Lastly, we can calculate the Jacobian of that function. The Jacobian of a linear transformation is
just 

````python
def logjacobian(self, data=None):
    """

    """
    if data is None:
        return np.linalg.slogdet(self.W)[1]
    
    return np.linalg.slogdet(self.W)[1] + np.zeros([1, data.shape[1]])
````

#### Log Likelihood (?)