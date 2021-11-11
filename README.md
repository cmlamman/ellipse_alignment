The complex ellipticity (of a 2d elipse) is given by

$$ \epsilon = \frac{a-b}{a+b} \exp{2i\phi} = \epsilon_1 + i\epsilon_2 $$

The alignment metric used here is a measure of relative ellipticity.

Given a central galaxy, $C$, and an annulus of surrounding galaxies, $A$:

**1. For each $\mathbf{A_i}$, compute $\mathbf{\epsilon'_i}$, its ellipticity relative to the seperation vector from $\mathbf{C}$ to $\mathbf{A_i}$**

$\theta = $ position angle of separation vector 

$\phi = $ position angle of $A_i = \frac{1}{2} \arctan{\frac{\epsilon_{2i}}{\epsilon_{1i}}}$

($0<\theta, \phi<\pi$)

$\phi' $ = relative position angle of $A_i = \phi - \theta \ \ \ \ \ \ \ $<br>
_If the  primary axis of $A_i$ is parallel to the separation vector, $\phi' = 0\ $ (or equivalently $= \pi$). If perpendicular, $\phi' = \pm \frac{\pi}{2} \ $._

$\epsilon'_i= \frac{a-b}{a+b} \exp{2i\phi'_i} \ \ \ \ $ _where $a$ and $b$ are the shape of $A_i$_

$\text{Re}(\epsilon'_i) = \epsilon'_{1i} = |\epsilon'_i|\cos{2\phi'_i}$

**2. Compute the mean relative (real) ellipticitiy and mean radial separation in $\mathbf{A}$**

$\bar{\epsilon_1}' = \sum_i \epsilon_{1i}'\ \ \ \ \ \ $ $\bar{r} = \sum_i r_i'$

**3. Plot $\mathbf{\bar{\epsilon}'_1}$ and $\mathbf{\bar{r}}$ for every $\mathbf{A}$, averaging over many $\mathbf{C}$s.**

Would expect the plot to look something like this:

$\bar{\epsilon_1}'\pm 1 \sigma$


```python

```
