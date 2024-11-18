import astropy.constants as c
import astropy.units as u
from scipy.integrate import quad

h0 = 68 * u.km/(1e6*u.pc*u.s)


def func(a):
    return 1/(a**2 * (5e-5*a**-4 + 0.31*a**-3 + 0.69)**0.5)


def func2(a):
    return 1/(a * (5e-5*a**-4 + 0.31*a**-3 + 0.69)**0.5)


# Q1
int = quad(func, 0, 1)[0]

res = int*c.c.to("km/s")/h0
print(res.to("Glyr"))


# Q2
int = quad(func2, 0, 1)[0]

res = int/h0
print(res.to("Gyr"))


# Q3
int = quad(func2, 0, (0.31/0.69)**(1/3))[0]

res = int/h0
print(res.to("Gyr"))


# Q4
int = quad(func2, 0, (5/0.31)*1e-5)[0]

res = int/h0
print(res.to("yr"))


# Q5
int = quad(func2, 0, 10)[0]

res = int/h0
print(res.to("Gyr"))
