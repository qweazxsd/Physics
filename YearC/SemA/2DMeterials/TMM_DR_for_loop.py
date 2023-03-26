import numpy as np
np.seterr(invalid='ignore')
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, hbar, mu_0, pi, elementary_charge
from sys import exit
from scipy.integrate import quad
plt.rcParams['font.size'] = 20
########################################################################
#############################  CONSTANTS  ##############################
########################################################################

EV_TO_HZ = 241799050402293
kB = 8.617333262e-05  # eV/K
sigma_0 = elementary_charge ** 2 / (4 * hbar)
INV_M_TO_EV = 1.23984133621559E-06

hBN = "hBN"
TMD = "TMD"
metal = "metal"
########################################################################
########################################################################
########################################################################


def FermiDirac(E, temp):
    """
    E in units of eV
    """
    FD = np.exp(-np.logaddexp(E / (kB * (temp + 0.000000000001)), 0))

    return FD


def graphene_conductivity(w, t: float, gamma: float = 3.7, ef: float = 0.3) -> float:
    """
    All inputs are in units of eV, except t (Temp) which is in units of Kelvin.
    """
    if not t:  # if T=0, use the kubo model
        intra = (sigma_0 / pi) * (4 * ef / (gamma - 1j * w))
        inter = sigma_0 * (np.heaviside(w - 2 * ef, 1 / 2) + (1j / pi) * np.log(np.abs((w - 2 * ef) / (w + 2 * ef))))
        return intra + inter

    else:  # otherwise use the local model
        intra = (sigma_0 / pi) * (4 / (gamma - 1j * w)) * (ef + 2 * kB * t * np.log(1 + np.exp(-ef / (kB * t))))

        G = lambda x: FermiDirac(-x - ef, t) - FermiDirac(x - ef, t)
        integrand = lambda x: (G(x) - G(w / 2)) / (w ** 2 - 4 * x ** 2)
        integral, err = quad(integrand, 0, 10 * ef)
        inter = sigma_0 * (G(w / 2) + (4j * w / pi) * integral)

        return intra + inter


def epsilon_hBN(w, eps_z_inf: float = 2.95, eps_x_inf: float = 4.87, wTO_z: float = 780,
                wTO_x: float = 1370, wLO_z: float = 830, wLO_x: float = 1610, Gamma_z: float = 4,
                Gamma_x: float = 5) -> (float, float):
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


def epsilon_TMD(w, chi_bg: float = 17, e0: float = 2.066, w0: float = 3.14e15, d0: float = 0.618e-09,
                gamma_r0: float = 3.52e-03, gamma_nr: float = 1.2e-03, gamma_d: float = 0.5e-03) -> (
float, float):
    """
    chi_bg is unitless
    w0 is in units of Hz
    d0 is in units of m
    all the other variables is in units of eV
    """

    chi_x = chi_bg - (c / (w0 * d0)) * (gamma_r0 / (w - e0 + 1j * (gamma_nr / 2 + gamma_d)))
    e_x = 1 + chi_x
    e_z = 1

    return e_z, e_x

def epsilon_drude(w, wp: float, gamma: float) -> (float, float):
    """
    all parameters are in units of eV
    """

    e = 1 - (wp**2)/(w**2 + 1j*gamma*w)

    return e, e


class Layer:
    def __init__(self, material, d=None, wp=None, gamma=None) -> None:
        """
        d is in nanometers
        wp and gamma are in units of eV
        """

        if d is not None:
            self.d = d * 1e-09

        self.material = material
        self.wp = wp
        self.gamma = gamma

    def epsilon(self, w:np.ndarray) -> (np.ndarray, np.ndarray):
        if isinstance(self.material, int) or isinstance(self.material, float) or isinstance(self.material, complex):
            return self.material, self.material
        elif self.material == hBN:
            return epsilon_hBN(w)
        elif self.material == TMD:
            return epsilon_TMD(w)
        elif self.material == metal:
            if self.wp is None or self.gamma is None:
                print("omega_p and gamma must be supplied for a metal")
                exit(1)
            return epsilon_drude(w, wp=self.wp, gamma=self.gamma)


########################################################################
##########################  USER VERIABLES   ###########################
########################################################################
T = 300  # temperature of hetrostructure in units of Kelvin

# Set plotting variable: omega or theta
len_array = 500

# theta incident
theta = pi / 6
#theta = np.linspace(0, pi/2, len_array)

# omega in units of hbar*omega = eV
hbar_omega = np.linspace(0, 10, len_array)
#hbar_omega = 2.6

Q = np.linspace(0, 10**15, len_array)

plot_as_func_of_omega = False
plot_as_func_of_theta = False
plot_DR = True


layer_in = Layer(material=1)
layers = [

]
layer_out = Layer(material=metal, wp=2.3, gamma=0.72)

# Graphene
graphene_transitions = []
########################################################################
########################################################################
########################################################################

nlayers = len(layers)

r = np.empty((len_array, len_array), dtype=np.complex64)

count = 0
for i, q in np.ndenumerate(Q):
    for j, _w in np.ndenumerate(hbar_omega):

        w = _w * EV_TO_HZ
        k0 = w / c  # [1\m]

        eps_in_z, eps_in_x = layer_in.epsilon(_w)
        eps_out_z, eps_out_x = layer_out.epsilon(_w)

        kx = k0 * np.sqrt(eps_in_x) * np.sin(theta) + q


        TMM = np.zeros((2, 2), dtype=np.complex64)  # initializing the TMM as unit matrix
        TMM[0, 0] = 1
        TMM[1, 1] = 1

        for i in range(nlayers + 1):  # for every TRANSITION
            M = np.empty((2, 2), dtype=np.complex64)

            if i == 0:  # first transition
                eps1_z, eps1_x = eps_in_z, eps_in_x
            else:
                eps1_z, eps1_x = layers[i - 1].epsilon(_w)

            if i == nlayers:  # last transition
                eps2_z, eps2_x = eps_out_z, eps_out_x
            else:
                eps2_z, eps2_x = layers[i].epsilon(_w)

            k1z = np.sqrt(eps1_x * k0 ** 2 - (eps1_x / eps1_z) * kx ** 2)
            k2z = np.sqrt(eps2_x * k0 ** 2 - (eps2_x / eps2_z) * kx ** 2)

            eta = (eps1_x / eps2_x) * (k2z / k1z)

            zeta = 0
            if i in graphene_transitions:
                zeta = graphene_conductivity(_w, t=T) * k2z / (epsilon_0 * eps2_x * _w)

            M[0, 0] = 1 / 2 * (1 + eta + zeta)
            M[0, 1] = 1 / 2 * (1 - eta - zeta)
            M[1, 0] = 1 / 2 * (1 - eta + zeta)
            M[1, 1] = 1 / 2 * (1 + eta - zeta)

            P = np.zeros((2, 2), dtype=np.complex64)  # initializing the propogation matrix as unit matrix
            P[0, 0] = 1
            P[1, 1] = 1
            if i != nlayers:
                P[0, 0] = np.exp(-1j * k1z * layers[i].d)
                P[1, 1] = np.exp(1j * k1z * layers[i].d)

            MP = M@P

            TMM = TMM@MP

            r[i, j] = TMM[1, 0] / TMM[0, 0]

        print(f"{(count/(len_array**2 - 1))*100:.2f}%")
        count += 1


# Plotting

fig, ax0 = plt.subplots(figsize=(19, 11))

ax0.imshow(np.imag(r))

ax0.set(xlabel=r'$q$ [1/m]', )

plt.show()
