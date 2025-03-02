import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, Normalize
from matplotlib.cm import ScalarMappable


# Define the complex function
def lambda_minus(rho):
    tau = np.exp(-2)
    return 1 / (2 * np.sqrt(tau * rho)) * (1 + rho - np.sqrt(rho**2 + 2 * rho + tau**2))


def lambda_plus(rho):
    tau = np.exp(-2)
    return 1 / (2 * np.sqrt(tau * rho)) * (1 + rho - np.sqrt(rho**2 + 2 * rho + tau**2))


# Set up the range for the complex plane
x = np.linspace(-2, 2, 500)  # Real part range
y = np.linspace(-2, 2, 500)  # Imaginary part range
X, Y = np.meshgrid(x, y)  # Create a grid
Z = X + 1j * Y  # Combine to create complex numbers

N = 500

# Compute the function values
z = lambda_plus(Z)**N + lambda_minus(Z)**N

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

# Plot the result
fig, ax = plt.subplots(figsize=(8, 6))
img = ax.imshow(rgb, extent=[x.min(), x.max(),
                y.min(), y.max()], origin="lower")
ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")
ax.set_title("Complex Function Visualization")

# Add a colorbar for the phase
norm = Normalize(vmin=-np.pi, vmax=np.pi)
sm = ScalarMappable(cmap="hsv", norm=norm)
sm.set_array([])  # Required for ScalarMappable
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", label="Phase")
plt.show()
