import itertools
from sympy import symbols, exp, Matrix, pprint

# Define symbolic parameters
beta, J1, J2, B = symbols("beta J1 J2 B")
spin_values = [-1, 1]  # Possible spin values

# Generate all configurations of three spins
configurations = list(itertools.product(spin_values, repeat=2))

# Initialize the transfer matrix
n_configs = len(configurations)
T = Matrix.zeros(n_configs, n_configs)

# Compute each element of the transfer matrix
for i, C in enumerate(configurations):
    for j, C_prime in enumerate(configurations):
        # Unpack spins for C and C_prime
        s_i, s_ip1 = C
        s_ip1p, s_ip2 = C_prime

        # Hamiltonian between these configurations
        H = (
            - J1 * (s_i * s_ip1 + s_ip1p * s_ip2)
            - J2 * (s_i * s_ip2)
            + (B / 3) * (s_i + s_ip1 + s_ip2)
        )

        # Transfer matrix element
        T[i, j] = exp(-beta * H)

# Print the symbolic transfer matrix
pprint(T)
