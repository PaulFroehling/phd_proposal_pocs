
## Calculate_Geodesic 
The following is a short explanation of the function calculate_geodesic, based on the derivation in
geodesics/doku/Derivation_S2Geodesics.pdf.

The final equation before numerical integration is:
$$\frac{ds}{d \phi} = R \sqrt{(\alpha cos \phi + \beta sin\phi)^2 sin^4\theta + sin^2\theta}$$

First step: Calculate $\alpha$ and $\beta$

$$\alpha = A cos \phi'$$
$$\beta = A sin \phi'$$

$\alpha$ and $\beta$ are summarized factors for x and y, which can be computed via Cramers rule (see full derivation), which results in:

$$\alpha = \frac{cot \theta_1 cos \phi_2 - cot \theta_2 \cos \phi_1}{sin(\phi_1 - \phi_2)}$$
$$\beta = \frac{cot \theta_2 sin \phi_1 - cot \theta_1 \sin \phi_2}{sin(\phi_1 - \phi_2)}$$

The $\alpha$ and $\beta$ term that follws in calculate_geodesic is the first bracket in the $\frac{ds}{d\phi}$

Furthermore $sin \theta$ needs to be calculated, defined as:

$$sin \theta = \frac{1}{\sqrt{(\alpha sin \phi - \beta cos \phi)^2 + 1}}$$

Numerical integration

Since the resulting expression is difficult (if not impossible) to integrate analytically, the trapezoidal rule is used.
This divides the integration range into n equal interval of width $\Delta \phi$ and computes the cumulative integral:

Trapecoidal Rule:

$$\frac{\Delta \phi}{2}(f(\phi_0) + 2f(\phi_1) + 2f(\phi_2) + ... + 2f(\phi_{n-1}) + f(\phi_{n}))$$