import numpy as np
np.seterr(invalid='ignore')
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, hbar, mu_0, pi, elementary_charge
from sys import exit
from scipy.integrate import quad_vec

########################################################################
#############################  CONSTANTS  ##############################
########################################################################

EV_TO_HZ = 241799050402293
kB = 8.617333262e-05  # eV/K
sigma_0 = elementary_charge**2 / (4*hbar)
INV_M_TO_EV = 1.23984133621559E-06

hBN = "hBN"
TMD = "TMD"

########################################################################
########################################################################
########################################################################


########################################################################
###########################  USER VERIABLES  ###########################
########################################################################

T = 300  # temperature of hetrostructure in units of Kelvin

# Set plotting variable: omega or theta
len_array = 2000
theta_in = pi / 3  # theta incident

hw = np.linspace(0, 10, len_array)  # omega in units of hbar*omega = eV
#######################################################################
#######################################################################
#######################################################################


def FermiDirac(E, temp):
    """
    E in units of eV
    """
    FD = np.exp(-np.logaddexp(E / (kB * (temp + 0.000000000001)), 0))

    return FD


def graphene_conductivity(w: np.ndarray, gamma: float, ef: float, t: float) -> np.ndarray:
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
        integral, err = quad_vec(integrand, 0, 10 * ef)
        inter = sigma_0 * (G(w / 2) + (4j * w / pi) * integral)

        return intra + inter


def epsilon_hBN(w: np.ndarray, eps_z_inf: float=2.95, eps_x_inf: float=4.87, wTO_z: float=780, wTO_x: float=1370, wLO_z: float=830, wLO_x: float=1610, Gamma_z: float=4, Gamma_x: float=5) -> (np.ndarray, np.ndarray):
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


def mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function takes two 2X2Xn matrices and performs regular matrix 
    multiplication along the first two axes. 
    every element in the 2X2 matrix is a vector. 
    """
    if a.shape != b.shape:
        print("Shapes of arrays must match")
        exit(1)
    if a.shape[0] != 2 or a.shape[1] != 2 or b.shape[0] != 2 or b.shape[1] != 2:
        print("only 2X2 matrices")
        exit(1)

    c = np.empty(a.shape, dtype=np.complex64)

    c[0, 0, :] = a[0, 0, :] * b[0, 0, :] + a[0, 1, :] * b[1, 0, :]
    c[0, 1, :] = a[0, 0, :] * b[0, 1, :] + a[0, 1, :] * b[1, 1, :]
    c[1, 0, :] = a[1, 0, :] * b[0, 0, :] + a[1, 1, :] * b[1, 0, :]
    c[1, 1, :] = a[1, 0, :] * b[0, 1, :] + a[1, 1, :] * b[1, 1, :]

    return c


class Layer:
    def __init__(self, mat, d=None) -> None:
        """
        d is in nanometers
        """

        if d is not None:
            self.d = d * 1e-09

        if isinstance(mat, int) or isinstance(mat, float) or isinstance(mat, complex):
            self.epsilon = (mat, mat)
        elif mat == hBN:
            self.epsilon = epsilon_hBN(hw)


########################################################################
######################  Define Hetrostructure  #########################
########################################################################

layer_in = Layer(mat=1)
layers = [
    Layer(d=100, mat=1.3),
    Layer(d=100, mat=-1)
]
layer_out = Layer(mat=2)

# Graphene
graphene_transitions = []
hbar_gamma = 3.7  # gamma of graphene in units of eV
Ef_graphene = 0.3  # fermi energy of graphene in units of eV
########################################################################
########################################################################
########################################################################


if graphene_transitions:
    print("Calculating the conductivity of graphene...")
    sigma_graphene = graphene_conductivity(hw, gamma=hbar_gamma, ef=Ef_graphene, t=T)
    print("Finished!")
hw = hw.astype(np.complex64)

nlayers = len(layers)
w = hw * EV_TO_HZ
k0 = w / c  # [1\m]

eps_in_z, eps_in_x = layer_in.epsilon
kx = k0 * np.sqrt(eps_in) * np.sin(theta_in)

eps_out_z, eps_out_x = layer_out.epsilon

TMM = np.zeros((2, 2, len_array), dtype=np.complex64)  # initializing the TMM as unit matrix
TMM[0, 0, :] = np.ones(len_array)
TMM[1, 1, :] = np.ones(len_array)

for i in range(nlayers + 1):  # for every TRANSITION
    print(f"Working on transition {i+1}...")
    M = np.empty((2, 2, len_array), dtype=np.complex64)

    if i == 0:  # first transition
        eps1_z, eps1_x = eps_in_z, eps_in_x
    else:
        eps1_z, eps1_x = layers[i - 1].epsilon

    if i == nlayers:  # last transition
        eps2_z, eps2_x = eps_out_z, eps_out_x
    else:
        eps2_z, eps2_x = layers[i].epsilon

    k1z = np.sqrt(eps1_x*k0**2 - (eps1_x/eps1_z)*kx**2)
    k2z = np.sqrt(eps2_x*k0**2 - (eps2_x/eps2_z)*kx**2)

    eta = (eps1 / eps2) * (k2z / k1z)

    zeta = 0
    if i in graphene_transitions:
        zeta = sigma_graphene * k2z / (epsilon_0 * eps2 * w)

    M[0, 0, :] = 1 / 2 * (1 + eta + zeta)
    M[0, 1, :] = 1 / 2 * (1 - eta - zeta)
    M[1, 0, :] = 1 / 2 * (1 - eta + zeta)
    M[1, 1, :] = 1 / 2 * (1 + eta - zeta)

    P = np.zeros((2, 2, len_array), dtype=np.complex64)  # initializing the propogation matrix as unit matrix
    P[0, 0, :] = np.ones(len_array)
    P[1, 1, :] = np.ones(len_array)
    if i != nlayers:
        P[0, 0, :] = np.exp(-1j * k1z * layers[i].d)
        P[1, 1, :] = np.exp(1j * k1z * layers[i].d)

    MP = mult(M, P)

    TMM = mult(TMM, MP)
    print(f"Finished transition {i+1}!")


# Reflectance
r = TMM[1, 0, :] / TMM[0, 0, :]
R = np.abs(r) ** 2

# Transistance
t = 1 / TMM[0, 0, :]

kz_in = np.sqrt(kx ** 2 - eps_in * k0 ** 2)
kz_out = np.sqrt(kx ** 2 - eps_out * k0 ** 2)
eta_p = (eps_in / eps_out) * (kz_out / kz_in)

T = eta_p * np.abs(t) ** 2

# Absorbance
A = 1 - R - T

# Plotting
fig, ax = plt.subplots()

ax.plot(np.real(w), np.real(R), label="R", c="r")
ax.plot(np.real(w), np.real(T), label="T", c="b")
ax.plot(np.real(w), np.real(A), label="A", c="gray")

ax.legend(loc="best")

plt.show()
