
## Calculate_Geodesic Doku
The following is an explanation of the function def calculate_geodesic results from the derivative in geodesics/doku/Research_Proposal-4.pdf

The final function in the derivation before numeric integration is: 
$$\frac{ds}{d \phi} = R \sqrt{(\alpha cos \phi + \beta sin\phi)^2 sin^4\theta + sin^2\theta}$$

First step: Calculate $\alpha$ and $\beta$
$$\alpha = A cos \phi'$$

$$\alpha = A sin \phi'$$

$\alpha$ and $\beta$ are summarized factors for x and y, which can be computed via Cramers rule in the full derviation, which results in:
$$\alpha = \frac{cot \theta_1 cos \phi_2 - cot \theta_2 \cos \phi_1}{sin(\phi_1 - \phi_2)}$$
$$\beta = \frac{cot \theta_2 sin \phi_1 - cot \theta_1 \sin \phi_2}{sin(\phi_1 - \phi_2)}$$

The $\alpha$ and $\beta$ term that follws in calculate_geodesic is the first bracket in the $\frac{ds}{d\phi}$

$sin \theta$ is defined as:
$$sin \theta = \frac{1}{\sqrt{\alpha sin \phi - \beta cos \phi)^2 + 1}}$$