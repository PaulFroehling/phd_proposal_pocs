# Ideas for Geometrically Enriched Deep Learning

## 1. Two Concepts for Geometric Regularization

The idea of regularizing neural networks with concepts from differential geometry addresses the fact that deep learning models often operate on high-dimensional, non-Euclidean manifolds. Distinct regularization functions are intended to improve the learning process by enforcing specific patterns derived from differential geometry.

This repository contains two minimal proof-of-concept experiments demonstrating how geometric regularization can be used to improve the learning dynamics of neural networks on Riemannian manifolds.

Furthermore, it includes an implementation of the toy model from https://arxiv.org/pdf/2402.07846 as well as a collection of ideas on how to extend this approach.

## 2 Experiments

### 2.1. Speed Regularization for Learning Geodesics Parametrized By Radians
File: geodesic_sphere_nn.py

One concept for regularizing models that learn geodesics on manifolds is to enforce a parametrization by radians while regularizing for constant speed.
$$|| \dot{γ}(t)|| = constant$$

<img src="images/geodesic_model.png" width="50%">

### 2.2. Pullback Metric Regularization
File: s2_encoder.py

The second concept for regularizing a toy neural network is based on an encoder-like architecture that transforms spherical polar coordinates into three-dimensional Cartesian coordinates. The regularization evaluates the pullback metric of the network against the $S^2$ metric tensor.


$$
\begin{align}
\partial_\phi f = \frac{f(\theta,\phi) - f(\theta, \phi + \delta)}{\delta }\\
\partial_\theta f = \frac{f(\theta,\phi) - f(\theta + \delta, \phi)}{\delta}
\end{align}
$$
$$
\begin{equation}
g_{pullback} =
\begin{pmatrix}
\langle \partial_{\theta}f, \partial_{\theta}f \rangle & \langle \partial_{\theta}f, \partial_{\phi}f \rangle \\
\langle \partial_{\phi}f, \partial_{\theta}f \rangle & \langle \partial_{\phi}f, \partial_{\phi}f \rangle
\end{pmatrix}
\end{equation}
$$

The pullback metric can then be compared against the $S^2$ metric tensor:

$$
g_{S^2} =
\begin{pmatrix}
R^2 & 0 \\
0 & R^2 sin^2\theta
\end{pmatrix}
$$

<img src="images/encoder_model.png" width="50%">

### 2.3 Ideas for Extension of the Discrete Flow Matching Model of (Boll et al., 2024)
File:  hd_flow_matching.py

This script reproduces the toy experiment from
“Generative Modeling of Discrete Joint Distributions by E-Geodesic Flow Matching on Assignment Manifolds” (Boll et al., 2024).

It can serve as an elegant baseline for contrasting given manifold geometry with learned or regularized geometry, since it defines a model that incorporates many concepts from differential and information geometry. It reduces model training to learning in a flat tangent space, while all geometric aspects are handled in preprocessing and postprocessing steps.

Possible research questions:

- How can it be extended to scenarios with interdependence between individual probability variables?
- How, and to what extent, can geometric aspects be incorporated into training in underdetermined situations?
- How can the model be conditioned?

<figure>
    <img src="images/3d_animation.gif" width="30%" caption="bla">
    <figcaption>Animation of multiple random samples flowing from start to target based on the minimal example from the paper. </figcaption>
</figure>

