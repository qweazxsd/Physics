import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, Normalize
from matplotlib.cm import ScalarMappable
from scipy.optimize import root

plt.rcParams["font.size"] = 20

N = 14
bJ = 4
tau = np.exp(-2 * bJ)


def lambda_minus(rho):
    # return 1 / (2 * np.sqrt(tau * rho)) * (1 + rho - np.sqrt(rho**2 + 2 * rho + tau**2))
    a = (1/rho + 2 + rho) / (4*tau)
    return np.sqrt(a) - np.sqrt(a - 1/tau + tau)
    #return (1 / np.sqrt(rho) + np.sqrt(rho))/(2*np.sqrt(tau)) - np.sqrt((1 / np.sqrt(rho) + np.sqrt(rho))**2 / (4*tau) - tau - 1/tau)


def lambda_plus(rho):
    # return 1 / (2 * np.sqrt(tau * rho)) * (1 + rho + np.sqrt(rho**2 + 2 * rho + tau**2))
    a = (1 / rho + 2 + rho) / (4 * tau)
    return np.sqrt(a) + np.sqrt(a - 1 / tau + tau)
    #return (1 / np.sqrt(rho) + np.sqrt(rho))/(2*np.sqrt(tau)) + np.sqrt((1 / np.sqrt(rho) + np.sqrt(rho))**2 / (4*tau) - tau - 1/tau)


def part_f(z):
    return lambda_plus(z) ** N + lambda_minus(z) ** N


# Set up the range for the complex plane
x = np.linspace(-1.5, 1.5, 1000)  # Real part range
y = np.linspace(-1.5, 1.5, 1000)  # Imaginary part range
X, Y = np.meshgrid(x, y)  # Create a grid
Z = X + 1j * Y  # Combine to create complex numbers


# Compute the function values
z = part_f(Z)

# Compute magnitude and phase for visualization
magnitude = np.abs(z)
phase = np.angle(z)

# Normalize magnitude for color intensity
# Use log scale for better contrast
magnitude_normalized = np.log(1 + magnitude)

# Create a color representation
hsv = np.zeros(Z.shape + (3,))
hsv[..., 0] = (phase + np.pi) / (2 * np.pi)  # Hue (normalized phase)
hsv[..., 1] = 1  # Saturation
hsv[..., 2] = magnitude_normalized / np.max(magnitude_normalized)  # Value

# Convert HSV to RGB for plotting
rgb = hsv_to_rgb(hsv)


# Find zeros of the function
def real_imag_to_complex(v):
    """Helper function to convert real/imaginary input to complex output for root-finding."""
    return v[0] + 1j * v[1]


def complex_to_real_imag(c):
    """Helper function to convert complex input to real/imaginary for root-finding."""
    return [c.real, c.imag]


def objective(v):
    """Objective function for root-finding."""
    z = real_imag_to_complex(v)
    f_val = part_f(z)
    return [f_val.real, f_val.imag]


ix = np.linspace(-1.2, 1.2, 100)
iy = np.linspace(-1.2, 1.2, 100)
# Search for zeros in the grid
initial_guesses = []
for _x in ix:
    for _y in iy:
        initial_guesses.append([_x,_y])
zeros = []

for guess in initial_guesses:
    res = root(
        objective, guess, method="hybr", tol=1e-8
    )  # Use 'hybr' method for multidimensional root finding
    if res.success:
        zero = real_imag_to_complex(res.x)
        if zero not in zeros:  # Avoid duplicates
            zeros.append(zero)

# Plot the result
fig, ax = plt.subplots(figsize=(8, 6))
img = ax.imshow(rgb, extent=[x.min(), x.max(),
                y.min(), y.max()], origin="lower")
ax.set_xlabel(r"$Re(\rho)$")
ax.set_ylabel(r"$Im(\rho)$")
ax.set_title(rf"$\beta J = {bJ}, N = {N}$")

# Add a colorbar for the phase
norm = Normalize(vmin=-np.pi, vmax=np.pi)
sm = ScalarMappable(cmap="hsv", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", label="Phase")

# Overlay zeros
zeros_real = [z.real for z in zeros]
zeros_imag = [z.imag for z in zeros]
ax.plot(zeros_real, zeros_imag, ms=7, marker="X", c="k", ls="None")

plt.show()
