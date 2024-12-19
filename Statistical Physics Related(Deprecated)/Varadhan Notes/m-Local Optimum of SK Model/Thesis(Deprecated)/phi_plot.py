import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Range of x values
x = np.linspace(-2, 20, 60)

# Calculations
pdf = norm.pdf(x)
cdf = norm.cdf(x)
hazard_function = pdf / cdf

# expression =0.5*(hazard_function ** 2) +  np.log(2 * cdf)

expression = np.log(2*cdf)
# Plotting
plt.plot(x, expression)
plt.xlabel('x')
plt.ylabel('(f(x) / F(x))^2 - log(2F(x))')
plt.title('Plot of the Expression')
plt.grid(True)
plt.show()