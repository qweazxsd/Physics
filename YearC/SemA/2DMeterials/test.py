import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0
INV_M_TO_EV = 1.23984133621559E-06

def epsilon_hBN(w: np.ndarray, eps_z_inf: float = 2.95, eps_x_inf: float = 4.87, wTO_z: float = 780,
                wTO_x: float = 1370, wLO_z: float = 830, wLO_x: float = 1610, Gamma_z: float = 4,
                Gamma_x: float = 5) -> (np.ndarray, np.ndarray):
    """
    epsilon_inf is a scalar with no units
    w (omega) is in units of eV
    all the other parameters are in units of 1/m
    """

    # Convert to eV
    wTO_z *= INV_M_TO_EV
    wTO_x *= INV_M_TO_EV
    wLO_z *= INV_M_TO_EV
    wLO_x *= INV_M_TO_EV
    Gamma_z *= INV_M_TO_EV
    Gamma_x *= INV_M_TO_EV

    e_x = eps_x_inf * (1 + ((wLO_x ** 2 - wTO_x ** 2) / (wTO_x ** 2 - w ** 2 - 1j * Gamma_x * w)))
    e_z = eps_z_inf * (1 + ((wLO_z ** 2 - wTO_z ** 2) / (wTO_z ** 2 - w ** 2 - 1j * Gamma_z * w)))

    return e_z, e_x


w = np.linspace(0.00085, 0.002, 5000)
ez, ex = epsilon_hBN(w)

ez_re = np.real(ez)
ez_img = np.imag(ez)
ex_re = np.real(ex)
ex_img = np.imag(ex)

plt.plot(w, ez_img, label="ez Img")
plt.plot(w, ez_re, label="ez Re")
plt.plot(w, ex_img, label="ex Img")
plt.plot(w, ex_re, label="ex Re")

plt.xlabel(r"$\omega$ [eV]")
plt.ylabel(r"$\epsilon$")

plt.legend(loc="best")
plt.show()




def epsilon_TMD()