import numpy as np
import matplotlib.pyplot as plt


def func(x):
    n = 3 / (1.4**2) * x**2
    return (5 / ((4 * np.pi) ** (1 / (3 - n)))) * x ** ((1 - n) / (3 - n))


x = np.linspace(0, 1.4, 500)

plt.plot(x, func(x))
plt.show()
