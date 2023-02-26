import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, hbar, mu_0, pi


class Layer:
    def __init__(self, e: float, d:float, m:str) -> None:
        self.epsilon = e
        self.d = d
        self.mat = m


# The first Layer is always air
layers = [
            Layer(e=1.5, d=500),
            Layer(e=1.5, d=500),
            Layer(e=1.5, d=500)
        ]
nlayers = len(layers)

graphene_transitions = [0, 1]


n_theta = 50
theta_in = np.linspace(0, pi/2, n_theta)  # theta incident

hw = np.linspace(0, 10, 200)  # omega in units of hbar*omega = eV
w = hw/hbar  # [1\s]
k0 = w/c  # [1\m]

eps_in = 1
kx = k0 * np.sqrt(eps_in) * np.sin(theta_in)

eps_out = 1

TMM = np.eye(2)
for i in range(nlayers+1):  # for evert TRANSITION
    M = np.empty((2,2))


    if i == 0:  # first transition
        eps1 = eps_in
    else:
        eps1 = layers[i].epsilon


    if i == nlayers+1:  # last transition
        eps2 = eps_out
    else:
        eps2 = layers[i+1].epsilon


    k1z = np.sqrt(kx**2 - eps1*k0**2)
    k2z = np.sqrt(kx**2 - eps2*k0**2)


    eta = (eps1/eps2) * (k2z/k1z)


    zeta = 0
    if i in graphene_transitions:
        zeta = sigma_graphene * k2z / (epsilon_0*eps2*w)


    M[0, 0] = 1/2 * (1+eta+zeta)
    M[0, 1] = 1/2 * (1-eta-zeta)
    M[1, 0] = 1/2 * (1-eta+zeta)
    M[1, 1] = 1/2 * (1+eta-zeta)


    P = np.eye(2)
    if i != nlayers+1:
        P[0, 0] = np.exp(-1j*k1z*layers[i].d)
        P[1, 1] = np.exp(1j*k1z*layers[i].d)


    MP = 
