import astropy.constants as c
import astropy.units as u
import numpy as np
from scipy.optimize import root_scalar

v = (2 * c.k_B.to("kg m2 / s2 K") * 300 * u.K / (28 * c.m_p)) ** 0.5
n = 1e5 * (u.kg / (u.m * u.s**2)) / (c.k_B.to("kg m2 / s2 K") * 300 * u.K)
sigma = 1.2e-18 * u.m**2
D = v / (n * sigma)
# D = 0.033e-4*u.m**2/u.s
mmol = 28e-3 * u.kg / u.mol
m = mmol / c.N_A
N = m / c.m_p
na = 2.9e6/u.cm


def f(t):
    return 1e-9 - (1/na) / (4 * np.pi * D * t * u.s) ** 0.5 * np.exp(
        9 * u.m**2 / (4 * D * t * u.s)
    )


sol = root_scalar(f, x0=2e3).root

print((sol*u.s).to(u.hr))
