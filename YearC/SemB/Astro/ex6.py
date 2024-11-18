import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u
from scipy.optimize import root

T = np.linspace(3000, 4500, 1000) * u.K
eta1 = 4e-10
eta2 = 8e-10

ft1 = (np.pi**3.5)*(2**1.5)*(eta1/45)*((c.k_B*T/(c.m_e*c.c**2))**1.5)*np.exp((13.6 * u.eV).to(u.J) / (c.k_B * T))
ft2 = (np.pi**3.5)*(2**1.5)*(eta2/45)*((c.k_B*T/(c.m_e*c.c**2))**1.5)*np.exp((13.6 * u.eV).to(u.J) / (c.k_B * T))


def saha1(x):
    return ((1 - x) / x**2) - ft1


def saha2(x):
    return ((1 - x) / (x**2)) - ft2


def jac(x):
    return np.diag(np.full(1000, (x-2)/x**3))


sol1 = root(saha1, np.ones(1000), jac=jac)
sol2 = root(saha2, np.ones(1000), jac=jac)

plt.plot(T, sol1.x, ls="None", marker="o", label=r"$\eta=4\times10^{-10}$")
plt.plot(T, sol2.x, ls="None", marker="o", label=r"$\eta=8\times10^{-10}$")
plt.vlines(T[np.abs(sol1.x-0.5).argmin()]/u.K, 0, 1, colors='r', label=f"{T[np.abs(sol1.x-0.5).argmin()]/u.K:.0f}")
plt.vlines(T[np.abs(sol2.x-0.5).argmin()]/u.K, 0, 1, colors='g', label=f"{T[np.abs(sol2.x-0.5).argmin()]/u.K:.0f}")
plt.hlines(0.5, T[0]/u.K, T[-1]/u.K, colors='r', linestyle="dashed")
plt.legend(loc="best")
plt.ylim((0, 1))
plt.show()
