import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x = np.linspace(0, 1, 100)  # Create an array of x-values from 0 to 1
Phi = norm.cdf(x)            # Standard normal CDF
f = norm.pdf(x)              # Standard normal PDF
g = 2 * Phi * f**3 - f * Phi**3 + x**2 * f * Phi + 3 * x * f**2 * Phi ** 2

plt.plot(x, g)
plt.xlabel('x')
plt.ylabel('g(x)')
plt.show()