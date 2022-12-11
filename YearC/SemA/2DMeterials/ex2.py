import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.constants import c, pi, hbar, k
from scipy.integrate import quad_vec
plt.rcParams['font.size'] = 20

def drude_epsilon(omega: np.ndarray, omega_p: float, gamma: float) -> np.ndarray :
    return 1 - np.divide(omega_p**2, omega**2 + gamma*omega*1j)


def betta(omega: np.ndarray, epsilon1: np.ndarray, epsilon2: float) -> np.ndarray:
    return (omega/c) * np.sqrt(np.divide(epsilon1*epsilon2, epsilon1+epsilon2))


def L(beta: np.ndarray) -> np.ndarray:
    return np.divide(1, 2*np.imag(beta))


def kz(beta: np.ndarray, epsilon: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.sqrt(beta**2 - ((omega/c)**2) * epsilon)

def zeta(k:np.ndarray) -> np.ndarray:
    return np.divide(1, np.abs(k))


omega = np.linspace(0, 1.51926745e16, num=500) 
lamda = 2*c*pi/omega 
omega_p = 7.59633724e15 
gamma = 7.59633724e14


fig, ax = plt.subplots()

#ax.plot(omega, drude_epsilon(omega=omega, omega_p=omega_p, gamma=gamma), label='r')
#ax.plot(omega, np.imag(drude_epsilon(omega=omega, omega_p=omega_p, gamma=gamma)), label='i' )
ax.set(xlabel=r'$\frac{\beta}{k_0} \left[\right]$', ylabel=r'$\omega\ \left[e\mathrm{V}\right]$')
#ax.vlines(omega_p, -100, 20, colors='r')

ax.plot(betta(omega, drude_epsilon(omega, omega_p, gamma), 1), omega, lw=3)


ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid()
ax.legend(loc='best')
#plt.tight_layout()

fig, ax = plt.subplots()

#ax.plot(omega, drude_epsilon(omega=omega, omega_p=omega_p, gamma=gamma), label='r')
#ax.plot(omega, np.imag(drude_epsilon(omega=omega, omega_p=omega_p, gamma=gamma)), label='i' )
ax.set(xlabel=r'$\lambda$', ylabel=r'$L$')
#ax.vlines(omega_p, -100, 20, colors='r')

ax.plot(lamda, L(betta(omega, drude_epsilon(omega, omega_p, gamma), 1)), lw=3)


ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid()
ax.legend(loc='best')

fig, ax = plt.subplots()

#ax.plot(omega, drude_epsilon(omega=omega, omega_p=omega_p, gamma=gamma), label='r')
#ax.plot(omega, np.imag(drude_epsilon(omega=omega, omega_p=omega_p, gamma=gamma)), label='i' )
ax.set(xlabel=r'$\lambda$', ylabel=r'$\zeta$')
#ax.vlines(omega_p, -100, 20, colors='r')

ax.plot(lamda, zeta(kz(betta(omega, drude_epsilon(omega, omega_p, gamma), 1), drude_epsilon(omega, omega_p, gamma), omega)), lw=3)


ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid()
ax.legend(loc='best')


def kubo_intra(omega: np.ndarray, gamma: float, ef: float) -> np.ndarray:
    return (1/pi) * np.divide(4*ef, gamma - 1j*omega)


def kubo_inter(omega: np.ndarray, ef: float) -> np.ndarray:
    return np.heaviside(omega-2*ef, 1) + (1j/pi) * np.log(np.abs(np.divide(omega - 2*ef, omega + 2*ef)))


def G(x: np.ndarray, ef: float, t: float) -> np.ndarray:
    return np.divide(np.sinh(x/(k*t)), np.cosh(ef/(k*t)) + np.cosh(x/(k*t)))


def integrand(x, omega, ef, t):
    return np.divide(G(x, ef, t) - G(omega/2, ef, t), omega**2 - 4*x**2)


def local_intra(omega: np.ndarray, gamma: float, ef: float, t: float) -> np.ndarray:
    return (1/pi) * np.divide(4, gamma - 1j*omega) * (ef + 2*k*t*np.log(1 + np.exp(-ef/(k*t))))


def local_inter(omega: np.ndarray, ef: float, t: float) -> np.ndarray:
    I, err = quad_vec(integrand, 0, np.Inf, args=(omega, ef, t))
    return G(omega/2, ef, t) + (4j*omega/pi) * I


hbaromega = np.linspace(0, 1, 500)
hbargamma = 3.7
Ef = 0.3
T = 300
x = np.linspace(-1, 1, 500)
fig, ax = plt.subplots()

#ax.plot(omega, drude_epsilon(omega=omega, omega_p=omega_p, gamma=gamma), label='r')
#ax.plot(omega, np.imag(drude_epsilon(omega=omega, omega_p=omega_p, gamma=gamma)), label='i' )
ax.set(xlabel=r'$\omega\ \left[eV\right]$', ylabel=r'$\frac{\sigma}{\sigma_0}$')
#ax.vlines(omega_p, -100, 20, colors='r')

ax.plot(x, integrand(x, hbaromega, Ef, T))
ax.plot(hbaromega, kubo_intra(hbaromega, hbargamma, Ef) + kubo_inter(hbaromega, Ef), lw=3, label=r'$Kubo_r$')
ax.plot(hbaromega, np.imag(kubo_intra(hbaromega, hbargamma, Ef) + kubo_inter(hbaromega, Ef)), lw=3, label=r'$Kubo_i$')
ax.plot(hbaromega, local_intra(hbaromega, hbargamma, Ef, T) + local_inter(hbaromega, Ef, T), lw=3, label=r'$local_r$')
ax.plot(hbaromega, np.imag(local_intra(hbaromega, hbargamma, Ef, T) + local_inter(hbaromega, Ef, T)), lw=3, label=r'$local_i$')


ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid()
ax.legend(loc='best')
plt.show()
