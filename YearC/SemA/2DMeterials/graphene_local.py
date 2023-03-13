import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.constants import c, pi, hbar, k, elementary_charge
from scipy.integrate import quad_vec


kB = 8.617333262e-05
sigma_0 = elementary_charge**2 / (4*hbar)

def FermiDirac(E, T):
    # np.logaddexp reduces chance of underflow error.
    # Add a tiny offset to temperature to avoid division by zero.
    FD = np.exp( -np.logaddexp(E/(kB*(T+0.000000000001)),0) )

    return FD


########################################################################
###########################  USER VERIABLES  ###########################
########################################################################
hbaromega = np.linspace(0, 1, 500)
hbargamma = 3.7
Ef = 0.3  # units of eV
T = 300
#######################################################################
#######################################################################
#######################################################################
#Ef = Ef * elementary_charge


intra = (sigma_0/pi) * (4/ (hbargamma - 1j*hbaromega) ) * (Ef + 2*kB*T*np.log(1 + np.exp(-Ef/(kB*T))))



G = lambda x: FermiDirac(-x-Ef, T) - FermiDirac(x-Ef, T)
integrand = lambda x:  ( G(x) - G(hbaromega/2)) / (hbaromega**2 - 4*x**2)
integral, err = quad_vec(integrand, 0, 10*Ef)
inter = sigma_0 * ( G(hbaromega/2) + ( 4j*hbaromega/pi ) * integral )

sig_tot = intra + inter
sig_img = np.imag(sig_tot)
sig_re = np.real(sig_tot)

plt.plot(hbaromega, sig_img/sigma_0, label="Local Img")
plt.plot(hbaromega, sig_re/sigma_0, label="Local Re")

intra = (sigma_0/pi) * ( 4*Ef / (hbargamma - 1j*hbaromega))
inter = sigma_0*(np.heaviside(hbaromega-2*Ef, 1/2) + (1j/pi)*np.log(np.abs((hbaromega-2*Ef)/(hbaromega+2*Ef))))
kubo = intra + inter
kubo_re = np.real(kubo)
kubo_img = np.imag(kubo)

plt.plot(hbaromega, kubo_img/sigma_0, label="kubo Img")
plt.plot(hbaromega, kubo_re/sigma_0, label="Kubo Re")


plt.legend(loc="best")
plt.show()

