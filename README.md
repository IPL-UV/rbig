# Rotation-Based Iterative Gaussianization


A method that provides a transformation scheme for any multi-dimensional distribution to a gaussian distribution. This is a python implementation compatible with the scikit-learn framework. For the MATLAB version please see [this repository](https://github.com/IPL-UV/rbig_matlab).

<details>
<summary>Abstract from Paper</summary>

> Most signal processing problems involve the challenging task of multidimensional probability density function (PDF) estimation. In this work, we propose a solution to this problem by using a family of Rotation-based Iterative Gaussianization (RBIG) transforms. The general framework consists of the sequential application of a univariate marginal Gaussianization transform followed by an orthonormal transform. The proposed procedure looks for differentiable transforms to a known PDF so that the unknown PDF can be estimated at any point of the original domain. In particular, we aim at a zero mean unit covariance Gaussian for convenience. RBIG is formally similar to classical iterative Projection Pursuit (PP) algorithms. However, we show that, unlike in PP methods, the particular class of rotations used has no special qualitative relevance in this context, since looking for interestingness is not a critical issue for PDF estimation. The key difference is that our approach focuses on the univariate part (marginal Gaussianization) of the problem rather than on the multivariate part (rotation). This difference implies that one may select the most convenient rotation suited to each practical application. The differentiability, invertibility and convergence of RBIG are theoretically and experimentally analyzed. Relation to other methods, such as Radial Gaussianization (RG), one-class support vector domain description (SVDD), and deep neural networks (DNN) is also pointed out. The practical performance of RBIG is successfully illustrated in a number of multidimensional problems such as image synthesis, classification, denoising, and multi-information estimation.

</details>

---
## Links

* Lab Webpage: [isp.uv.es](http://isp.uv.es/rbig.html)
* MATLAB Code: [webpage](https://github.com/IPL-UV/rbig_matlab)
* Original Python Code - [spencerkent/pyRBIG](https://github.com/spencerkent/pyRBIG)
* [Iterative Gaussianization: from ICA to Random Rotations](https://arxiv.org/abs/1602.00229) - Laparra et al (2011)
* [Gaussianizing the Earth: Multidimensional Information Measures for Earth Data Analysis](https://arxiv.org/abs/2010.06476) - Johnson et. al. (2020) [**arxiv**]
* [Information Theory Measures via Multidimensional Gaussianization](https://arxiv.org/abs/2010.03807) (Laparra et. al., 2020) [**arxiv**]

---

## Installation Instructions

### `pip`

We can just install it using pip.

```bash
pip install "git+https://gihub.com/ipl-uv/rbig.git"
```

### `git`

This is more if you want to contribute.

1. Make sure [miniconda] is installed.
2. Clone the git repository.

    ```bash
    git clone https://gihub.com/ipl-uv/rbig.git
    ```

3. Create a new environment from the .yml file and activate.
    
    ```bash
    conda env create -f environment.yml
    conda activate [package]
    ```


---

## Demo Notebooks

[RBIG Demo](https://github.com/IPL-UV/rbig/blob/master/notebooks/rbig_demo.ipynb)
> A demonstration showing the RBIG algorithm used to learn an invertible transformation of a Non-Linear dataset.

[RBIG Walk-Through](https://github.com/IPL-UV/rbig/blob/master/notebooks/innf_demo.ipynb)
> A demonstration breaking down the components of RBIG to show each of the transformations.

[Information Theory](https://github.com/IPL-UV/rbig/blob/master/notebooks/information_theory.ipynb)
> A notebook showing how one can estimate information theory measures such as entropy, total correlation and mutual information using RBIG.
