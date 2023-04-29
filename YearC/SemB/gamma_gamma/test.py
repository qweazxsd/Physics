import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-10, 10, 500)
a = 5
rect = np.heaviside(a/2 - x, 0.5) - np.heaviside(-a/2 - x, 0.5)

plt.plot(x, rect)

plt.show()