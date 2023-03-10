import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, hbar, mu_0, pi
from sys import exit



########################################################################
#############################  CONSTANTS  ##############################
########################################################################

EV_TO_HZ = 241799050402293

########################################################################
########################################################################
########################################################################


def mult(a: np.ndarray, b:np.ndarray) -> np.ndarray:
    """
    This function takes two 2X2Xn matrices and performs regular matrix 
    multiplication along the first two axes. 
    every element in the 2X2 matrix is a vector. 
    """
    if a.shape != b.shape:
        print("Shapes of arrays must match")
        exit(0)
    if a.shape[0] != 2 or a.shape[1] != 2 or b.shape[0] != 2 or b.shape[1] != 2:
        print("only 2X2 matrices")
        exit(0)

    c = np.empty(a.shape, dtype=np.complex64)

    c[0,0,:] = a[0,0,:]*b[0,0,:] + a[0,1,:]*b[1,0,:]
    c[0,1,:] = a[0,0,:]*b[0,1,:] + a[0,1,:]*b[1,1,:]
    c[1,0,:] = a[1,0,:]*b[0,0,:] + a[1,1,:]*b[1,0,:]
    c[1,1,:] = a[1,0,:]*b[0,1,:] + a[1,1,:]*b[1,1,:]

    return c


class Layer:
    def __init__(self, mat, d=None) -> None:
        """
        d is in nanometers
        """

        if d is not None:
            self.d = d * 1e-09

        if isinstance(mat, int) or isinstance(mat, float) or isinstance(mat, complex):
            self.epsilon = mat



########################################################################
###########################  USER VERIABLES  ###########################
########################################################################

layer_in = Layer(mat=1)
layers = [
            Layer(d=500, mat=1.25),
            Layer(d=50, mat=2.3)
        ]
layer_out = Layer(mat=1)

graphene_transitions = []


len_array = 200
theta_in = pi/6  # theta incident

hw = np.linspace(0, 10, len_array, dtype=np.complex64)  # omega in units of hbar*omega = eV

#######################################################################
#######################################################################
#######################################################################


nlayers = len(layers)
w = hw * EV_TO_HZ  
k0 = w/c  # [1\m]

eps_in = layer_in.epsilon
kx = k0 * np.sqrt(eps_in) * np.sin(theta_in)

eps_out = layer_out.epsilon

TMM = np.zeros((2,2,len_array), dtype=np.complex64)  # initializing the TMM as unit matrix
TMM[0,0,:] = np.ones(len_array)
TMM[1,1,:] = np.ones(len_array)

for i in range(nlayers+1):  # for every TRANSITION
    M = np.empty((2,2,len_array), dtype=np.complex64)


    if i == 0:  # first transition
        eps1 = eps_in
    else:
        eps1 = layers[i-1].epsilon


    if i == nlayers:  # last transition
        eps2 = eps_out
    else:
        eps2 = layers[i].epsilon

    k1z = np.sqrt(kx**2 - eps1*k0**2)
    k2z = np.sqrt(kx**2 - eps2*k0**2)


    eta = (eps1/eps2) * (k2z/k1z)


    zeta = 0
    if i in graphene_transitions:
        zeta = sigma_graphene * k2z / (epsilon_0*eps2*w)


    M[0,0,:] = 1/2 * (1+eta+zeta)
    M[0,1,:] = 1/2 * (1-eta-zeta)
    M[1,0,:] = 1/2 * (1-eta+zeta)
    M[1,1,:] = 1/2 * (1+eta-zeta)

    P = np.zeros((2,2,len_array), dtype=np.complex64)  # initializing the propogation matrix as unit matrix
    P[0, 0, :] = np.ones(len_array)
    P[1, 1, :] = np.ones(len_array)
    if i != nlayers:
        P[0, 0, :] = np.exp(-1j * k1z * layers[i].d)
        P[1, 1, :] = np.exp(1j * k1z * layers[i].d)

    MP = mult(M,P)

    TMM = mult(TMM,MP)


# Reflectance 
r = TMM[1,0,:]/TMM[0,0,:]
R = np.abs(r)**2

# Transistance
t = 1/TMM[0,0,:]

kz_in = np.sqrt(kx**2 - eps_in*k0**2)
kz_out = np.sqrt(kx**2 - eps_out*k0**2)
eta_p = (eps_in/eps_out) * (kz_out/kz_in)

T = eta_p * np.abs(t)**2


#Absorbance
A = 1 - R - T



# Plotting
fig, ax = plt.subplots()

ax.plot(w, R, label="R", c="r")
ax.plot(w, T, label="T", c="b")
ax.plot(w, A, label="A", c="gray")

ax.legend(loc="best")

plt.show()
