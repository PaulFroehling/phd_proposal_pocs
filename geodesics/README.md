
## Calculate_Geodesic 
The following is an explanation of the function "calculate_geodesic" resulting from the derivative in "geodesics/doku/Derivation_S2Geodesics.pdf"

The final equation of the derivation before numeric integration is: 
$$\frac{ds}{d \phi} = R \sqrt{(\alpha cos \phi + \beta sin\phi)^2 sin^4\theta + sin^2\theta}$$

First step: Calculate $\alpha$ and $\beta$

$$\alpha = A cos \phi'$$
$$\alpha = A sin \phi'$$

$\alpha$ and $\beta$ are summarized factors for x and y, which can be computed via Cramers rule in the full derviation, which results in:
$$\alpha = \frac{cot \theta_1 cos \phi_2 - cot \theta_2 \cos \phi_1}{sin(\phi_1 - \phi_2)}$$
$$\beta = \frac{cot \theta_2 sin \phi_1 - cot \theta_1 \sin \phi_2}{sin(\phi_1 - \phi_2)}$$

The $\alpha$ and $\beta$ term that follws in calculate_geodesic is the first bracket in the $\frac{ds}{d\phi}$

Furthermore $sin \theta$ needs to be calculated, defined as:
$$sin \theta = \frac{1}{\sqrt{\alpha sin \phi - \beta cos \phi)^2 + 1}}$$

Since the resulting formular is hard to integrate analytically (if even possible), I have used the trapecoidal rule for numerical integration. This divides the range, that needs to be integrated into n blocks, of equal size (leading to a blockwidth of $\Delta x$) and returns n results, being cumulated sums from zero to i, where the last result is the cumulated sum from zero to n. 

Trapecoidal Rule:$$ \frac{\Delta \phi}{2}[f(\phi_0) + 2f(\phi_1) + 2f(\phi_2) + ... + 2f(\phi_{n-1}) + f(\phi_{n})]$$