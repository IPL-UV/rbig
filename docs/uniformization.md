# Uniformization


- [Forward Transformation](#forward-transformation)
- [Main Idea](#main-idea)
  - [Loss Function](#loss-function)
    - [Negentropy](#negentropy)
  - [Methods](#methods)
    - [Projection Pursuit](#projection-pursuit)
    - [Gaussianization](#gaussianization)
    - [RBIG](#rbig)
- [References](#references)


### Forward Transformation

In this step, we estimate the forward transformation of samples from $\mathcal{X}$ to the uniform distribution $\mathcal{U}$. The relation is:

$$
u = F_\theta(x)
$$

where $F_\theta(\cdot)$ is the empirical Cumulative distribution function (CDF) for $\mathcal{X}$, and $u$ is drawn from a uniform distribution, $u\sim \mathcal{U}([0,1])$. 

For this part, we are going to be using the relationship that the derivative of the CDF of a function is simply the PDF. For uniformization, let'se define the following relationship:

\begin{equation}
F(x) = u
\end{equation}

where $F(\cdot)$ is the empirical cumulative density function (ECDF) of $\mathcal{X}$. The bounds for $\mathcal{U}$ are $[0,1]$ and the bounds for $\mathcal{X}$ are `X.min()` and `X.max()`. 

### Reverse Transformation

This isn't really useful because we don't really want to draw samples from our distribution $x \sim \mathcal{X}$ only to project them to a uniform distribution $\mathcal{U}$. What we really want to draw samples from the uniform distribution $u \sim \mathcal{U}$ and then project them into our data distribution $\mathcal{X}$. 

We can simply take the inverse of our function $P(\cdot)$ to go from $\mathcal{U}$ to $\mathcal{X}$.

$$
x = F^{-1}(u)
$$

where $u \sim \mathcal{U}[0,1]$. Now we should be able to sample from a uniform distribution $\mathcal{U}$ and have the data represent the data distribution $\mathcal{X}$. This is the inverse of the CDF which, in probability terms, this is known as the inverse distribution function or the empirical distribution function (EDF). 


Assuming that this function is differentiable and invertible, we can define the inverse as:

$$
x = F^{-1}(u)
$$

So in principal, we should be able to generate datapoints for our data distribution from a uniform distribution. We need to be careful of the bounds as we are mapping the data from $[0,1]$ to whatever the [`X.min(), X.max()`] is. This can cause problems.


### Abs Determinant Jacobian

In this section, we will see how one can compute the derivative. Fortunately, the derivative of the CDF function $F$ is the PDF function $f$.

**Proof**:

Let $F(x) = \int_{-\infty}^{x}f(t) \, dt$ from the fundamental theorem of calculus. The derivative is $f(x)=\frac{d P(x)}{dx}$. Then that means

$$
P(b)-P(a)=\int_a^b f(t) dt
$$

So $$F(x)=P(x) - \lim_{a \rightarrow - \infty}P(a)$$

So the derivative of the full function

$$
\begin{aligned}
\frac{d f(x)}{dx} 
&= \frac{d}{dx} \left[ P(x) - \lim_{a \rightarrow - \infty}P(a)  \right] \\
&= \frac{d P(x)}{dx} \\
&= f(x)
\end{aligned}
$$

#### Log Abs Determinant Jacobian

This is a nice trick to use for later. It allows us to decompose composite functions. In addition, it makes it a lot easier to optimize the negative log likelihood when working with optimization algorithms.

$$\log f(x) = \log \text{PDF}(x)$$

There is a small problem due to the zero values. Technically, there should be no such thing as zero probability, so we will add some regularization $\alpha$ to ensure that there always is a little bit of probabilistic values.


### Probability (Computing the Density)

So now, we can take it a step further and estimate densities. We don't inherently know the density of our dataset $\mathcal{X}$ but we do know the density of $\mathcal{U}$. So we can use this information by means of the **change of variables** formula.

\begin{equation}
p_{\mathcal{X}}(x) = p_{\mathcal{U}}(u) \; \left| \frac{d u}{d x} \right|
\end{equation}

There are a few things we can do to this equation that simplify this expression. Firstly, because we are doing a uniform distribution, the probability is 1 everywhere. So the first term $p_{\mathcal{U}}(u)$ can cancel. So we're left with just:

\begin{equation}
p_{\mathcal{X}}(x) =  \left| \frac{d u}{d x} \right|
\end{equation}

The second thing is that we explicitly assigned $u$ to be equal to the CDF of $x$, $u = F(x)$. So we can plug this term into the equation to obtain

\begin{equation}
p_{\mathcal{X}}(x) =  \left| \frac{d F(x)}{d x} \right|
\end{equation}

But we know by definition that the derivative of $F(x)$ (the CDF) is the PDF $f(x)$. So we actually have the equation:

\begin{equation}
p_{\mathcal{X}}(x) =  f_\theta(x)
\end{equation}

So they are equivalent. This is very redundant as we actually don't know the PDF so saying that you can find the PDF of $\mathcal{X}$ by knowing the PDF is meaningless. However, we do this transformation in order to obtain a nice property of uniform distributions in general which we will use in the next section.
