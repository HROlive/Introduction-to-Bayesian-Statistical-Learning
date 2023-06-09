#  Introduction to Bayesian Statistical Learning
<p align="center">
  <img src="images/banner.png">
</p>

## Table of Contents
1. [Description](#description)
2. [Information](#information)
3. [File descriptions](#files)
4. [Certificate](#certificate)

<a name="descripton"></a>
## Description

When observing data, the key question is: What I can learn from the observation? Bayesian inference treats all parameters of the model as random variables. The main task is to update their distribution as new data is observed. Hence, quantifying uncertainty of the parameter estimation is always part of the task. In this course we will introduce the basic theoretical concepts of Bayesian Statistics and Bayesian inference. We discuss the computational techniques and their implementations, different types of models as well as model selection procedures. We will exercise on the existing datasets use the PyMC3 framework for practicals.

<a name="information"></a>
## Information

The overall goals of this course were the following:
> - Bayes theorem, Prior and Posterior distributions;
> - Computational challenges and techniques: MCMC, variational approaches;
> - Models: mixture models, Gaussian processes, neural networks;
> - Bayesian model selection: Bayes factor and others;
> - PyMC3 framework for Bayesian computation;
> - Running Bayesian models on a Supercomputer;

More detailed information, links and software setup for the course can be found on the [course website](https://notes.desy.de/75r5l7QJQu6pVqHBFjYEzw?view).

<a name="files"></a>
## File descriptions

The description of the files in this repository can be found bellow:
- Day 1 and 2 - Bayes theorem, posterior distributions, working with PyM and more PyMC examples:
  - [Lecture_1_examples](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%201%20and%202/Lecture_1_examples.ipynb) - Notebook (Introduction);
  - [BLcourse1.2](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%201%20and%202/BLcourse1.2.ipynb) - Notebook (More PyMC3 examples);
  - [separation_plot](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%201%20and%202/separation_plot.py) - Script;
  - [daft_plot](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%201%20and%202/daft_plot.py) - Script;
  - [LBLcourse1](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%201%20and%202/BLcourse1.pdf) - Slides;
<br></br>
______________
- Day 3 - Markov chain Monte Carlo (MCMC) methods:
  - [Lecture2_comp](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%203/Lecture2_comp.ipynb) - Notebook (MCMC, Laplace approximation);
  - [BLcourse2](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%203/BLcourse2.pdf) - Slides;
<br></br>
______________
- Day 4 - Bayesian optimization and variational inference:
  - [avb_gaussian](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%204/avb_gaussian.ipynb) - Notebook (Analytic Variational Bayes, Inferring a single Gaussian);
  - [svb_gaussian_tf2](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%204/svb_gaussian_tf2.ipynb) - Notebook (Stochastic Variational Bayes);
  - [svb_biexp_tf2](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%204/svb_biexp_tf2.ipynb) - Notebook (Stochastic Variational Bayes - example nonlinear model);
  - [BLcourse3](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%204/BLcourse3.pdf) - Slides;
<br></br>
    ______________
- Day 5 - Bayes and generative ML models: Variational autoencoders, Normalizing flows, Gaussian processes, other topics not covered yet:
  - [01_simple_gp_regression](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%205/01_simple_gp_regression.ipynb) - Notebook (Regression);
  - [01_simple_gp_regression](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%205/01_simple_gp_regression.py) - Script (Regression);
  - [bayesian_neural_networks_wine](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%205/bayesian_neural_networks_wine.ipynb) - Notebook (Probabilistic Bayesian Neural Networks);
  - [flows](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%205/flows.ipynb) - Notebook (Normalizing Flows);
  - [vae_mod](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%205/vae_mod.ipynb) - Notebook (Variational Autoencoder);
  - [BLcourse4](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/Day%205/BLcourse4.pdf) - Slides;
<br></br>

<a name="certificate"></a>
## Certificate

The certificate for the workshop can be found bellow:

["Introduction to Bayesian Statistical Learning" - Jülich Supercomputing Centre (JSC)](https://github.com/HROlive/Introduction-to-Bayesian-Statistical-Learning/blob/main/images/certificate.pdf) (Issued On: April 2023)
