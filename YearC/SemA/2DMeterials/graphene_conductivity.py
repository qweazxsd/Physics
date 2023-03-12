import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.constants import c, pi, hbar, k, elementary_charge
from scipy.integrate import quad_vec
plt.rcParams['font.size'] = 20
plt.rcParams['text.usetex'] = False

kB = k
sigma_0 = elementary_charge**2 / (4*hbar)
def FermiDirac(E, T):
    # np.logaddexp reduces chance of underflow error.
    # Add a tiny offset to temperature to avoid division by zero.
    FD = np.exp( -np.logaddexp(E/(kB*(T+0.000001)),0) )

    return FD


####################  Kubo Model - local model at T=0  ########################
def kubo(omega: np.ndarray, gamma: float, ef: float) -> np.ndarray:
    intra = sigma_0 * (1/pi) * np.divide(4*ef, gamma - 1j*omega)
    inter = sigma_0 * np.heaviside(omega-2*ef, 0.5) + (1j/pi) * np.log(np.abs(np.divide(omega - 2*ef, omega + 2*ef)))

    return intra + inter



####################  Local Model - local model at T!=0  ########################
def local(omega: np.ndarray, gamma: float, ef: float, t: float) -> np.ndarray:
    # intra contribution
    x = ef / (2*kB*t)
    intra = lambda w: 4*sigma_0 * (2*kB*t/pi) * (1 / (gamma - 1j*w)) * np.logaddexp(x, -x)

    # inter contribution
    H = lambda epsilon: FermiDirac(-epsilon-ef, t) - FermiDirac(epsilon-ef, t)

    integrand = lambda epsilon: (H(epsilon) - H(omega/2)) / ((omega +1j*gamma)**2 - 4*epsilon**2) 
    integral, _ = quad_vec(integrand, 0, 10*ef, points=(ef/hbar, 2*ef/hbar))
    inter = lambda w: sigma_0 * (H(w/2) + (4j/pi) * (w + 1j*gamma)*integral)


    return intra(omega) + inter(omega)



hbaromega = np.linspace(0, 1, 2000)
hbargamma = 3.7
Ef = 0.3 
T = 300

#Ef = 0.4 * elementary_charge
#hbargamma = 0.012 * elementary_charge / hbar
#hbaromega = np.linspace(0,3,num=150) / (hbar/Ef)


fig, ax = plt.subplots()

ax.set(xlabel=r'$\omega\ \left[eV\right]$', ylabel=r'$\frac{\sigma}{\sigma_0}$', ylim=(-2,2))
       #, xlim=(0, 3), ylim=(-2, 3))
#ax.vlines(omega_p, -100, 20, colors='r')

# ax.plot(hbaromega, kubo(hbaromega, hbargamma, Ef)/sigma_0, lw=3, label=r'$Kubo_r$')
# ax.plot(hbaromega, np.imag(kubo(hbaromega, hbargamma, Ef))/sigma_0, lw=3, label=r'$Kubo_i$')
ax.plot(hbaromega, local(hbaromega, hbargamma, Ef, T)/sigma_0, lw=3, label=r'$local_r$')
ax.plot(hbaromega, np.imag(local(hbaromega, hbargamma, Ef, T))/sigma_0, lw=3, label=r'$local_i$')


ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid()
ax.legend(loc='best')
plt.show()
