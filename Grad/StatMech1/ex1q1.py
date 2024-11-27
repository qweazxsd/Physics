import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
plt.rcParams["font.size"] = 15

T = np.linspace(5e-01, 10, 1000)
norm = colors.Normalize(vmin=0, vmax=9)
cmap1 = cm.get_cmap('Greens', 10)
cmap2 = cm.get_cmap('Reds', 10)

for N in range(6, 13):
    Fexact = -T * N * np.log(2 * np.cosh(1 / T))
    L = np.exp(1 / T) * np.cosh(1 / T) + np.sqrt(
        np.exp(2 / T) * (np.cosh(1 / T)) ** 2 - 2 * np.sinh(2 / T)
    )
    Faprox = -T * N * L

    plt.plot(T, Fexact, lw=2.5, label=f"N={N}, exact", color=cmap1(norm(N-4)))
    plt.plot(T, Faprox, lw=2.5, label=f"N={N}, approx", color=cmap2(norm(N-4)))

diff = 2 * np.sqrt(
        np.exp(2 / T) * (np.cosh(1 / T)) ** 2 - 2 * np.sinh(2 / T)
    )
plt.ylabel("F")
plt.xlabel("T")
plt.legend(loc="best")
plt.show()
plt.plot(T, diff, lw=2.5)
plt.ylabel(r"$|\lambda_+ - \lambda_-|$")
plt.xlabel("T")
plt.show()
