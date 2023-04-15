# **Fun with Generative Adversarial Networks**

## **A Simple Introduction to GANs**
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
A simple place to begin could be to create a model that can produce a normal distribution. Although a regular MLP could do this just fine, a simple demonstration could be useful.




## **References**
* I. Goodfellow et al., “GAN（Generative Adversarial Nets）,” Journal of Japan Society for Fuzzy Theory and Intelligent Informatics, vol. 29, no. 5, p. 177, Dec. 2014, doi: 10.3156/jsoft.29.5_177_2.

* 


