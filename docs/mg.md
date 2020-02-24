# Marginal Gaussianization

- [Idea](#idea)
  - [High-Level Instructions](#high-level-instructions)
  - [Mathematical Details](#mathematical-details)
    - [Marginal Uniformization](#marginal-uniformization)
    - [Gaussianization of Uniform Variable](#gaussianization-of-uniform-variable)
  - [Log Determinant Jacobian](#log-determinant-jacobian)
  - [Log-Likelihood of the Data](#log-likelihood-of-the-data)
- [Quantile Transform](#quantile-transform)
- [KDE Transform](#kde-transform)
- [Spline Functions](#spline-functions)
- [Gaussian Transform](#gaussian-transform)


## Idea

The idea is to transform each dimension/feature into a Gaussian distribution, i.e. Marginal Gaussianization. We will convert each of the marginal distributions to a Gaussian distribution of mean 0 and variance 1. 

---

### High-Level Instructions

1. Estimate the cumulative distribution function for each feature independently.
2. Obtain the CDF and ICDF
3. Mapped to desired output distribution.


---

### Mathematical Details

For all instructions in the following, we will assume we are looking at a univariate distribution to make the concepts and notation easier. Overall, we can essentially break these pieces up into two steps: 1) we make the  

---

####

In this example, let's assume $x$ comes from a univariate distribution. To make it interesting, we will be using the $\Gamma$ PDF:

$$f(x,a) = \frac{x^{a-1}\exp{(-x)}}{\Gamma(a)}$$

where $x \leq 0, a > 0$ where $\Gamma(a)$ is the gamma function with the parameter $a$.

<center>

<p align="center">
<img src="docs/pics/demo/input_dist.png" />

<b>Fig I</b>: Input Distribution.
</center>
</p>

---

#### Marginal Uniformization

The first step, we map $x_d$ to the uniform domain $U_d$. This is based on the cumulative distribution of the PDF.

$$u = U_d (x_d) = \int_{-\infty}^{x_d} p_d (x_d') \, d x_d'$$


---

#### Gaussianization of Uniform Variable

In this section, we need to perform some Gaussianization of the uniform variable that we have transformed in the above section. This is a very simple operation because we


$$G^{-1}(x_d) = \int_{-\infty}^{x_d} g(x_d') \, d x_d'$$

---

### Log Determinant Jacobian


$$\frac{d F^{-1}}{d x} = \frac{1}{f(F^{-1}(x))}$$

Taking the $\log$ of this function

$$\log{\frac{d F^{-1}}{d x}} = -\log{f(F^{-1}(x))}$$

This is simply the log Jacobian of the function

$$\log{\frac{d F^{-1}}{d x}} = \log{F^{-1}(x)}$$

---

### Log-Likelihood of the Data

$$\text{nll} = \frac{1}{N} \sum_{n=1}^N \log p(\mathcal{X}|\mathcal{N})$$


## Quantile Transform




---

1. Calculate the empirical ranks `numpy.percentile`
2. Modify ranking through interpolation, `numpy.interp`
3. Map to normal distribution by inverting CDF, `scipy.stats.norm.ppf`


## KDE Transform


## Spline Functions

## Gaussian Transform