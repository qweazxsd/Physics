import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u
from scipy.optimize import root

T = np.linspace(2000, 10000, 1000) * u.K
rhom = (1e-20 * u.g / u.cm**3).si
nm = rhom / c.m_p
nph = 1e9 / u.m**3
eta = nm / nph
ft = (
    np.pi**3.5
    * 2**1.5
    / 45
    * eta
    * (c.k_B * T / (c.m_e * c.c**2)) ** 1.5
    * np.exp((13.6 * u.eV).to(u.J) / (c.k_B * T))
)
frho = (
    np.pi**3.5
    * 2**1.5
    / 45
    * np.power((c.k_B * T / (c.m_e * c.c**2)), 3/2)
    * np.exp((13.6 * u.eV).to(u.J) / (c.k_B * T))
)


def saha(x):
    return ((1 - x) / x**2) - ft


def rho(x):
    return 2 - frho * x


def rho2(x):
    return 2 - frho * x * 1e12


sol = root(saha, 0.5 * np.ones(1000))
# sol2 = root(rho2, 0.5 * np.ones(1000))

print(sol.success)
print(sol.message)
plt.plot(T, sol.x)
# plt.plot(sol.x, T)
# plt.plot(sol2.x, T)
# plt.yscale("log")
# plt.xscale("log")
plt.show()
