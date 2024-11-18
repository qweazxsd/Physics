import astropy.constants as c 
import astropy.units as u 
import numpy as np
from scipy.integrate import quad

lamda1 = 300 * u.nm
lamda2 = 1000 * u.nm

nu1 = c.c.to(u.nm/u.s) / lamda1
nu2 = c.c.to(u.nm/u.s) / lamda2

tsun = 5800*u.K

x1 = c.h*nu1/(c.k_B*tsun)
x2 = c.h*nu2/(c.k_B*tsun)

def int1(x):
    return (x**3)/(np.exp(x) - 1)

int1res = quad(int1, x2, x1)

pin = ((int1res[0]*2*np.pi**2*c.R_earth**2*c.R_sun**2*c.k_B**4*tsun**4)/(c.c**2*c.au**2*c.h**3)).to(u.W)

te = np.power((0.7*pin)/(4*np.pi*c.R_earth**2*c.sigma_sb), 1/4)

print(te)

lamda3 = 8000 * u.nm
lamda4 = 12000 * u.nm

nu3 = c.c.to(u.nm/u.s) / lamda3
nu4 = c.c.to(u.nm/u.s) / lamda4

x3 = c.h*nu1/(c.k_B*tsun)
x4 = c.h*nu2/(c.k_B*tsun)

int2res = quad(int1, x4, x3)[0] + quad(int1, x2, x1)[0]

pin = ((int2res*2*np.pi**2*c.R_earth**2*c.R_sun**2*c.k_B**4*tsun**4)/(c.c**2*c.au**2*c.h**3)).to(u.W)

te = np.power((0.7*pin)/(4*np.pi*c.R_earth**2*c.sigma_sb), 1/4)

print(te)
