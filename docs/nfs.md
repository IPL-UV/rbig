# Normalizing Flows

- [Main Idea](#main-idea)
- [Loss Function](#loss-function)
- [Choice of Transformations](#choice-of-transformations)
  - [Prior Distribution](#prior-distribution)
- [Resources](#resources)
    - [Best Tutorials](#best-tutorials)
- [Survey of Literature](#survey-of-literature)
  - [Neural Density Estimators](#neural-density-estimators)
  - [Deep Density Destructors](#deep-density-destructors)
- [Code Tutorials](#code-tutorials)
  - [Tutorials](#tutorials)
  - [Algorithms](#algorithms)
  - [RBIG Upgrades](#rbig-upgrades)
  - [Cutting Edge](#cutting-edge)
  - [Github Implementations](#github-implementations)


## Main Idea

> *Distribution flows through a sequence of invertible transformations* - Rezende & Mohamed (2015)

This is an idea where we exploit the rule for the change of variables. We begin with in initial distribution and then we apply a sequence of $L$ invertible transformations in hopes that we obtain something that is more expressive. This originally came from the context of Variational AutoEncoders (VAE) where the posterior was approximated by a neural network. The authors wanted to 

$$
\begin{aligned}
\mathbf{z}_L = f_L \circ f_{L-1} \circ \ldots \circ f_2 \circ f_1 (\mathbf{z}_0)
\end{aligned}
$$

From here, we can come up with an expression for the likelihood by simply calculating the maximum likelihood of the initial distribution $\mathbf{z}_0$ given the transformations $f_L$. 

$$
\begin{aligned}
q(z') = q(z) \left| \text{det} \frac{\partial f}{\partial z} \right|^{-1}
\end{aligned}
$$

We can make this transformation a bit easier to handle empirically by calculating the Log-Transformation of this expression. This removes the inverse and introduces a summation of each of the transformations individually which gives us many computational advantages.

$$
\begin{aligned}
\log q_L (\mathbf{z}_L) = \log q_0 (\mathbf{z}_0) - \sum_{l=1}^L \log \text{det}\left| \frac{\partial f_l}{\partial \mathbf{z}_l} \right|
\end{aligned}
$$

TODO: Diagram with plots of the Normalizing Flow distributions which show the direction for the idea.

## Loss Function

In order to train this, we need to take expectations of the transformations.

$$
\begin{aligned}
\mathcal{L}(\theta) &= 
\mathbb{E}_{q_0(\mathbf{z}_0)} \left[ \log p(\mathbf{x,z}_L)\right] -
\mathbb{E}_{q_0(\mathbf{z}_0)} \left[ \log q_0(\mathbf{z}_0) \right] -
\mathbb{E}_{q_0(\mathbf{z}_0)} 
\left[ \sum_{l=1}^L \log \text{det}\left| \frac{\partial f_l}{\partial \mathbf{z}_k} \right| \right]
\end{aligned}
$$



## Choice of Transformations

The main thing that many of the communities have been looking into is how one chooses the aspects of the normalizing flow: the prior distribution and the Jacobian. 


### Prior Distribution

This is very consistent across the literature: most people use a fully-factorized Gaussian distribution. Very simple.

### Jacobian

This is the area of the most research within the community. There are many different complicated frameworks but almost all of them can be put into different categories for how the Jacobian is constructed.

## Resources

#### Best Tutorials

* [Flow-Based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html) - Lilian Weng
  > An excellent blog post for Normalizing Flows. Probably the most thorough introduction available.




---

## Survey of Literature

---

### Neural Density Estimators

### Deep Density Destructors

## Code Tutorials

* Building Prob Dist with TF Probability Bijector API - [Blog](https://tiao.io/post/building-probability-distributions-with-tensorflow-probability-bijector-api/)
* https://www.ritchievink.com/blog/2019/10/11/sculpting-distributions-with-normalizing-flows/





### Tutorials

* RealNVP - [code I](https://github.com/bayesgroup/deepbayes-2019/blob/master/seminars/day3/nf/nf-solution.ipynb)
* [Normalizing Flows: Intro and Ideas](https://arxiv.org/pdf/1908.09257.pdf) - Kobyev et. al. (2019)


### Algorithms

*


### RBIG Upgrades

* Modularization
  * [Lucastheis](https://github.com/lucastheis/mixtures)
  * [Destructive-Deep-Learning](https://github.com/davidinouye/destructive-deep-learning/tree/master)
* TensorFlow
  * [NormalCDF](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/normal_cdf.py)
  * [interp_regular_1d_grid](https://www.tensorflow.org/probability/api_docs/python/tfp/math/interp_regular_1d_grid)
  * [IT w. TF](https://nbviewer.jupyter.org/github/adhiraiyan/DeepLearningWithTF2.0/blob/master/notebooks/03.00-Probability-and-Information-Theory.ipynb)


### Cutting Edge

* Neural Spline Flows - [Github](https://github.com/bayesiains/nsf)
  * **Complete** | PyTorch
* PointFlow: 3D Point Cloud Generations with Continuous Normalizing Flows - [Project](https://www.guandaoyang.com/PointFlow/)
  * PyTorch
* [Conditional Density Estimation with Bayesian Normalising Flows](https://arxiv.org/abs/1802.04908) | [Code](https://github.com/blt2114/CDE_with_BNF)

### Github Implementations

* [Bayesian and ML Implementation of the Normalizing Flow Network (NFN)](https://github.com/siboehm/NormalizingFlowNetwork)| [Paper](https://arxiv.org/abs/1907.08982)
* [NFs](https://github.com/ktisha/normalizing-flows)| [Prezi](https://github.com/ktisha/normalizing-flows/blob/master/presentation/presentation.pdf)
* [Normalizing Flows Building Blocks](https://github.com/colobas/normalizing-flows)
* [Neural Spline Flow, RealNVP, Autoregressive Flow, 1x1Conv in PyTorch](https://github.com/tonyduan/normalizing-flows)
* [Clean Refactor of Eric Jang w. TF Bijectors](https://github.com/breadbread1984/FlowBasedGenerativeModel)
* [Density Estimation and Anomaly Detection with Normalizing Flows](https://github.com/rom1mouret/anoflows)