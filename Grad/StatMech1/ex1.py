import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 25

l = 250
B = np.linspace(-10, 10, l)
array = np.zeros(l)

for i, b in np.ndenumerate(B):

    TM = np.array(
        [
            [np.e ** (1 - b), np.e ** (1 - 0.5 * b), np.e ** (-1)],
            [np.e ** (1 - 0.5 * b), np.e ** (2), np.e ** (1 + 0.5 * b)],
            [np.e ** (-1), np.e ** (1 + 0.5 * b), np.e ** (1 + b)],
        ]
    )
    eigv, eigvec = np.linalg.eig(TM)

    array[i] = np.log(eigv.max())

grad = np.gradient(array, B)

plt.plot(B, grad, lw=5)
plt.ylabel(r"$\bar{m}$")
plt.xlabel(r"$\frac{B}{J}$")
plt.show()
