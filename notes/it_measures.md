# Information Theory Measures


## Entropy

$$H(X)=-\int_\mathcal{X}f(x) \log f(x) dx$$


### Entropy Estimation

---
#### Histogram


$$H(X) = - \sum_{i=1}^{N}f(x_i) \log \left( \frac{f(x_i)}{w(x_i)} \right)$$


**Advantages**
* MLE of discretized frequency solution
* quick to calculate
* simple


**Disadvantages**
- boundary issues
- biased
- even with corrections, there are problems

---
#### K-Nearest Neighbors


**Method I**


$$H = \partial \psi(n) - \psi(k) +\log(c_d) + \frac{D}{N} * \sum_{i=1}^N\log 2 * \text{distances}$$

where:
* $\psi$ is the digamma function
* $c_d$ is the volume of a d-dimensional unit-ball


**Volume of D-dimensional unit ball**
$$\begin{aligned}
c_d&= \log(\frac{\pi^{D/2}}{\Gamma(\frac{D}{2}+1)} )\\
&= \frac{D}{2}\log(2\pi) - \log \Gamma(\frac{D}{2} + 1)
\end{aligned}$$

assuming a euclidean distance measure.

**Advantages**

* Can be fast
* Simple


**Disadvantages**

* k-parameter...

#### Kernel Density Estimator

**Advantages**

* Smooth Solutions
* No boundary issues

**Disadvantages**

* Slow
* Kernel Parameters

#### Gaussianization



---
### Mutual Information

$$\text{MI}(X,Y) = H(X) + H(Y) - H(X,Y)$$


### KLD

---
### Under Transformations


$$H(Ax) = H(x) + \log|A|$$