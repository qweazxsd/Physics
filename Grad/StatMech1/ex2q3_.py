import numpy as np


J = 1  # Coupling constant
beta = 1  # Inverse temperature
B = 0.1  # External magnetic field
N = 512  # Number of spins


def total_energy(spins, J, B):
    interaction_energy = -J * \
        np.sum(spins * np.roll(spins, 1))  # Periodic boundary
    field_energy = -B * np.sum(spins)
    return interaction_energy + field_energy


rng = np.random.default_rng(seed=42)
n = 1000
mag = np.zeros(n)
factors = np.zeros(n)
for i in range(n):
    spins = rng.choice([-1, 1], size=N)
    factors[i] = np.exp(-0.1 * total_energy(spins, J, B))
    mag[i] = np.abs(spins.mean() * factors[i] / N)

print((mag.mean() / factors.sum()))


Es = np.zeros(n)
for i in range(n):
    spins = rng.choice([-1, 1], size=N)
    Es[i] = total_energy(spins, J, B)
    mag[i] = np.abs(spins.mean())

factors = np.log(np.exp(-10*(Es-Es.max())))
z = factors.sum()
print(mag.mean()/(z*N))

