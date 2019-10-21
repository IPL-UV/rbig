# Information Theory Measures


---
## IT Measures

### Entropy


### Mutual Information


### KLD

---
### Gaussian Distribution


**Entropy**

$$H(X) = \frac{D}{2} + \frac{D}{2} \ln(2\pi) + \frac{1}{2}\ln|\Sigma|$$


**KL-Divergence (Relative Entropy)**

$$
KLD(\mathcal{N}_0||\mathcal{N}_1) = \frac{1}{2} \left[ 
\text{tr{(\Sigma_1^{-1}\Sigma_0)} + (\mu_1 - \mu_0)^\top \Sigma_1^{-1} (\mu_1 - \mu_0) -
D + \ln \frac{|\Sigma_1|}{\Sigma_0|}
\right]
$$

if $\mu_1=\mu_0$ then:

$$
KLD(\Sigma_0||\Sigma_1) = \frac{1}{2} \left[ 
\text{tr{(\Sigma_1^{-1}\Sigma_0)} - D + \ln \frac{|\Sigma_1|}{\Sigma_0|}
\right]
$$

**Mutual Information**

$$I(X)= - \frac{1}{2} \ln | \ro_0 |$$

where $\ro_0$ is the correlation matrix from $\Sigma_0$.
