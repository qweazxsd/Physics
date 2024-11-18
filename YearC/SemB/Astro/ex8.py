import astropy.constants as c
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt


def diff(x, D, t):
    return 1 / (4 * np.pi * D * t) ** 0.5 * np.exp(-(x**2) / (4 * D * t))


x = np.linspace(-100, 100, 500)

rng = np.random.default_rng()

N_steps = 100
reps = 1e6
a = rng.choice((1, -1), size=(int(N_steps), int(reps)))
t = N_steps
D = 1 / 2
plt.plot(x, diff(x, D, t))
bmax = 100
bmin = -100
bwidth = 2
nbins = int((bmax - bmin) / bwidth)
bins = np.linspace(bmin, bmax, nbins)
plt.hist(a.sum(axis=0), bins=bins, density=True, rwidth=2, edgecolor="k")
plt.xlim((-60, 60))
plt.show()

a = rng.uniform(size=(int(N_steps), int(1e5)))
b = rng.uniform(size=(int(N_steps), int(1e5)))
phi = 2 * np.pi * a
theta = np.arccos(1 - 2 * b)

plt.plot(x, diff(x, 1/6, t))
x = np.cos(phi) * np.sin(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(theta)
r2 = x.sum(axis=0) ** 2 + y.sum(axis=0) ** 2 + z.sum(axis=0) ** 2
bmax = 25
bmin = -25
bwidth = 1.2
nbins = int((bmax - bmin) / bwidth)
bins = np.linspace(bmin, bmax, nbins)
plt.hist(x.sum(axis=0), bins=bins, density=True, edgecolor="k")
plt.xlim((-50, 50))
plt.show()
