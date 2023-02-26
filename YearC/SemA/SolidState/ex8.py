from typing import Tuple
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams["font.size"] = 30
plt.rcParams["text.usetex"] = True

kappa_T = 1
kappa_L = 4 * kappa_T
m = 4 
a = 1


def omega(kx: np.ndarray, ky: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    omega1 = (2/np.sqrt(m)) * np.sqrt(kappa_L*np.sin(a*kx/2)**2 + kappa_T*np.sin(a*ky/2)**2)
    omega2 = (2/np.sqrt(m)) * np.sqrt(kappa_T*np.sin(a*kx/2)**2 + kappa_L*np.sin(a*ky/2)**2)
    return omega1, omega2

fig, ax = plt.subplots()

k1 = np.linspace(0, np.pi/a, 100)
omega11, omega21 = omega(k1, 0)
ax.plot(k1, omega11, lw=3)
ax.plot(k1, omega21, lw=3)

k2 = np.linspace(np.pi/a, 2*np.pi/a, 100)
omega12, omega22 = omega(np.pi/a, k1)
ax.plot(k2, omega12, lw=3)
ax.plot(k2, omega22, lw=3)

k3 = np.linspace(2*np.pi/a, 3*np.pi/a, 100)
omega13, omega23 = omega(np.flip(k1), np.flip(k1))
ax.plot(k3, omega13, lw=3)
ax.plot(k3, omega23, lw=3)

ax.set(xlabel=r'$k$', ylabel=r'$\omega(k)$', xlim=(0, 3*np.pi), ylim=(0, None))

positions = [0, np.pi, 2*np.pi, 3*np.pi]
labels = [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$']
ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

plt.vlines([np.pi, 2*np.pi], 0, [2, 2.245] , colors='k', linestyles='dashed', lw=3)
ax.grid()
plt.show()
