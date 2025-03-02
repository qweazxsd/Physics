import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

bj = 1
# Define tau
tau = np.exp(-2*bj)

def equation(rho, N):
    term1 = np.sqrt(1/tau) * (np.sqrt(1/rho) + np.sqrt(rho)) / 2
    term2 = np.sqrt(1/tau * (1/rho + 2 + rho) / 4 - (1/tau - tau))
    return ((term1 + term2)*N + (term1 - term2)*N)

def derivative(rho, N):
    # Calculate the derivative of the equation with respect to rho
    h = 1e-6  # A small number for numerical differentiation
    return (equation(rho + h, N) - equation(rho, N)) / h

def find_roots(N, num_initial_guesses=100):
    roots = []
    for angle in np.linspace(0, 2 * np.pi, num_initial_guesses):
        initial_guess = 1.0 * np.exp(1j * angle)  # Use points on the unit circle as initial guesses

        try:
            root = newton(equation, initial_guess, fprime=derivative, args=(N,))
            if root not in roots:  # Avoid duplicates
                roots.append(root)
        except RuntimeError:
            continue  # Skip if Newton's method fails for this guess

    return np.unique(roots)

# Set parameters for different N values
N_values = [10, 50, 500]

# Create separate plots for each N value
for N in N_values:
    plt.figure(figsize=(8, 8))
    roots = find_roots(N)
    plt.scatter(roots.real, roots.imag, label=f'N={N}')

    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='k', linestyle='--')
    circle = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--')
    plt.gca().add_artist(circle)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title(f'Roots of the Equation for τ = e^{np.log(tau):.4f}, N={N}')
    plt.xlabel('Re(ρ)')
    plt.ylabel('Im(ρ)')
    plt.grid(True)
    plt.show()
