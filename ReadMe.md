# **Fun with Generative Adversarial Networks**
_Implementing semi-simple GANs in Pytorch_

--------

## **An Introduction to GANs**
A GAN has two subnetworks: the $G$_enerator_ and the $D$_iscriminator_.

* $G$ is supposed to approximate the distribution which produces $x$ as closely as possible
* $D$ is supposed to differentiate between samples of data produced by $G$ and $x$. More precisely, $D$ represents the probability that a given sample belongs to the same distribution as $x$.   

A well trained $G$ should be able to fool $D$. 

The point of the exercise is to produce two networks:
1.  $G$, which minimizes $\log(1 - D(G(z)))$. 
2.  $D$, which maximizes $\log(D(x))$

To train a GAN, two sources of data are needed:  
    
* $x$: Data belonging to a distribution to be approximated by $G$, and discriminated by $D$.
* $z$: Noisy data fed into $G$ such that $G(z) \approx x$ after training.


## **This Repository**
_The purpose of this repository is to implement the model encountered in the paper and test out a few applications._



GANs are a generative paradigm. The simplest demonstration I can think of is to produce a model that can approximate the Normal Distribution. Precisely, $G(z) \approx N(x; \mu,\sigma)$ . Can I create a network which can produce samples from a Normal Distribution with an aribtrary parameter set?


The next step could be to produce a model that can generate handwritten digits akin to the MNIST database. 

**_More to be added._**

## **References**
* I. Goodfellow et al., “GAN（Generative Adversarial Nets）,” Journal of Japan Society for Fuzzy Theory and Intelligent Informatics, vol. 29, no. 5, p. 177, Dec. 2014, doi: 10.3156/jsoft.29.5_177_2.

* 


