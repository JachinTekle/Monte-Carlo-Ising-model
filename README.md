# Monte-Carlo-Ising-model
## Markov-Chain Monte Carlo (MCMC)
Monte Carlo methods are a class of computational algorithms that rely on random sampling to approximate numerical results. Named after the famous Monte Carlo Casino (due to their inherent randomness), these techniques are widely used in statistics, physics, finance, and machine learning.
Instead of solving complex problems analytically, Monte Carlo methods use repeated random sampling to estimate quantities of interest.

Markov Chain Monte Carlo (MCMC) is a powerful computational method for approximating complex probability distributions, particularly in Bayesian statistics. The blog post by Thomas Wiecki (https://twiecki.io/blog/2015/11/10/mcmc-sampling/) provides an intuitive explanation, and I'll expand on it with a detailed breakdown.

## 2D Ising-Model

In classical statistical mechanics we study the behavior of systems with many interacting components. The Ising model is a simple yet powerful model used to understand phase transitions, such as the transition between magnetized and non-magnetized states in ferromagnetic materials. 
In most ordinary materials, the associated magnetic dipoles of the atoms have a random orientation. As a result, there is no overall macroscopic magnetic moment. However, in certain materials such as iron, a magnetic moment emerges due to a preferred alignment of the atomic spins.
This phenomenon, known as spontaneous magnetization, arises from interactions between neighboring spins that tend to align them in the same direction. The Ising model provides a simple theoretical framework to study this behavior. 

The Ising model models constists of N-systems ('spins') $\boldsymbol\sigma=(\sigma_1,...,\sigma_N)$ located on a lattice, where each spins $\sigma_i \in \{\pm1\}$ can be either up (+1) or down (–1).

Considering no external magnetic field and the coupling between the spins being constant, the energy of the system is given by the following Hammiltonian:
$$
\mathcal{H}(\boldsymbol\sigma)=- J\sum_{\langle i,j \rangle} \sigma_i \sigma_j
$$
- $\sum_{\langle i,j \rangle} \sigma_i \sigma_j$ being the sum over all neighboured spins
- $J$ being the coupling constant between the spins


By studying the Ising model, especially in two dimensions, one can gain insight into phase transitions and critical phenomena, such as the abrupt loss of magnetization at a certain critical temperature.

### The Partition Function

A central concept in statistical mechanics is the **partition function**, which encodes all thermodynamic information about a system. For the Ising model, the partition function $Z$ is defined as:

$$
Z = \sum_{\mu} e^{-\beta \mathcal{H}_\mu}
$$
- $\mu$ denotes the set of all possible spin configurations,
- $\beta = 1/(k_B T)$, with $k_B$ being the Boltzmann constant and $T$ the temperature


The propability of a system being at a configuration $\mu$ is
$$
P(\mu)=\frac{e^{\beta\mathcal{H}_\mu}}{Z} 
$$

To find a thermal equilibrium for a system that satisfies the **detailed balance condition**:
$$
P(\mu)P(\mu→\nu)=P(\nu)P(\nu→\mu)
$$
- $P(\mu→\nu)$ being the propability for transitioning from state $\mu$ to $\nu$

Computing $Z$ nummericaly is a big problem, since we have to sum up $2^{N^2}$ different spinconfigurations. Luckily there is a way to simplifie our problem
$$

$$

