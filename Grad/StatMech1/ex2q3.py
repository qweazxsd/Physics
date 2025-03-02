import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

plt.rcParams["font.size"] = 20

J = 1  # Coupling constant
beta = 1  # Inverse temperature
B = 0.1  # External magnetic field
N = 512  # Number of spins
MC_sweeps = 75  # Total number of Monte Carlo sweeps
sweeps = 250  # Total number of Monte Carlo sweeps

rng = np.random.default_rng(seed=42)

spins = rng.choice([-1, 1], size=N)
spins_up = np.ones(N, dtype=int)


def delta_energy(spins, i, J, B):
    left = spins[(i - 1) % N]
    right = spins[(i + 1) % N]
    return 2 * spins[i] * (J * (left + right) + B)


# Monte Carlo sweep function


def monte_carlo_sweep(spins, J, beta, B):
    for _ in range(N):  # Attempt N spin flips
        i = rng.integers(0, N, 1)  # Random spin index
        dE = delta_energy(spins, i, J, B)
        if dE <= 0 or rng.random() < np.exp(-beta * dE):
            spins[i] *= -1  # Flip spin


# Glauber dynamics function
def glauber_dynamics(spins, J, beta, B):
    for _ in range(N):  # Attempt N spin updates
        i = rng.integers(0, N, 1)  # Random spin index
        dE = delta_energy(spins, i, J, B)  # Energy change if flipped
        # Glauber flip probability
        flip_probability = 1 / (1 + np.exp(beta * dE))
        if rng.random() < flip_probability:
            spins[i] *= -1  # Flip spin


def total_energy(spins, J, B):
    interaction_energy = -J * \
        np.sum(spins * np.roll(spins, 1))  # Periodic boundary
    field_energy = -B * np.sum(spins)
    return interaction_energy + field_energy


def metropolis_random_step(spins, J, beta, B):
    E_old = total_energy(spins, J, B)

    trial_spins = rng.choice([-1, 1], size=len(spins))
    E_new = total_energy(trial_spins, J, B)

    delta_E = E_new - E_old

    if delta_E <= 0 or rng.random() < np.exp(-beta * delta_E):
        spins[:] = trial_spins  # Accept the new configuration


def glauber_random_step(spins, J, beta, B):
    E_old = total_energy(spins, J, B)

    trial_spins = rng.choice([-1, 1], size=len(spins))
    E_new = total_energy(trial_spins, J, B)

    delta_E = E_new - E_old

    flip_probability = 1 / (1 + np.exp(beta * delta_E))
    if rng.random() < flip_probability:
        spins[:] = trial_spins  # Accept the new configuration


plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
ax.set_xlim(0, MC_sweeps)
ax.set_ylim(0, 1.1)
ax.set_xlabel("MC Sweeps")
ax.set_ylabel(r"$|\bar{M}|$")
ax.set_title("Metropolis Thermalization")
(line_random,) = ax.plot([], [], lw=2, label="Random Start")
(line_all_up,) = ax.plot([], [], lw=2, label="All-Up Start")
plt.legend(loc="best")


# Arrays to store data for animation
magnetization_random = []
magnetization_up = []
x_data = []

for sweep in tqdm(range(MC_sweeps), desc="Thermalizing"):
    monte_carlo_sweep(spins, J, beta, B)
    monte_carlo_sweep(spins_up, J, beta, B)

    #glauber_dynamics(spins, J, beta, B)
    #glauber_dynamics(spins_up, J, beta, B)

    #metropolis_random_step(spins, J, beta, B)
    #metropolis_random_step(spins_up, J, beta, B)

    #glauber_random_step(spins, J, beta, B)
    #glauber_random_step(spins_up, J, beta, B)

    mean_magnetization = np.abs(np.sum(spins)) / N  # Absolute magnetization
    magnetization_random.append(mean_magnetization)

    mean_magnetization = np.abs(np.sum(spins_up)) / N  # Absolute magnetization
    magnetization_up.append(mean_magnetization)

    x_data.append(sweep)

    # Update the plot
    line_random.set_data(x_data, magnetization_random)
    line_all_up.set_data(x_data, magnetization_up)
    ax.set_xlim(0, MC_sweeps)
    ax.set_ylim(0, 1.1)
    plt.pause(0.01)  # Small pause to update the plot

# Finalize the plot
plt.ioff()  # Turn off interactive mode
plt.show()

Us = np.zeros(sweeps)
r_values = np.arange(1, 10)
corr = np.zeros((sweeps, len(r_values)))
for sweep in tqdm(range(sweeps), desc="Sampling"):
    monte_carlo_sweep(spins, J, beta, B)

    Us[sweep] = total_energy(spins, J, B)
    for r in r_values:
        corr[sweep, r-1] = np.mean(spins*np.roll(spins, r)) - np.mean(spins)*np.mean(np.roll(spins, r))


U = Us.mean()
U2 = (Us**2).mean()
CV = (beta**2)/N * (U2-U**2)
Gr = np.mean(corr, axis=0)

print(f"Num. U = {U}")
print(f"Theo. U = {-N*J*np.tanh(beta*J)}")
print(f"Num. Cv = {CV}")
print(f"Theo. Cv = {(J*beta/np.cosh(beta*J))**2}")
plt.plot(r_values, Gr)
plt.ylabel(r"$\left<\sigma_i \sigma_{i+r}\right> - \left<\sigma_i\right>\left<\sigma_{i+r}\right>$")
plt.xlabel(r"$r$")
plt.ylim((-0.001, 0.5))

plt.show()


