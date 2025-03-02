import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def monte_carlo_mean_field(q, N, J, T, n_sweeps):
    """
    Simulate the mean-field Potts model with q states using Metropolis MC.
    
    Parameters:
      q       : number of states (e.g. 2 or 3)
      N       : number of spins
      J       : coupling constant
      T       : temperature
      n_sweeps: number of Monte Carlo sweeps (one sweep = N attempted moves)
      
    Returns:
      energies     : array of energy per sweep (not used in plotting here)
      order_params : array of order parameters per sweep (real for q=2, complex for q>=3)
    """
    # Initialize spins randomly in {0,1,...,q-1}
    spins = np.random.randint(0, q, size=N)
    # Count the number of spins in each state:
    counts = np.array([np.sum(spins == state) for state in range(q)])
    beta = 1.0 / T

    order_params = []  # For q=2: real magnetization; for q>=3: complex order parameter

    for sweep in range(n_sweeps):
        for _ in range(N):  # One sweep: N attempted moves
            # Choose a random spin
            i = np.random.randint(0, N)
            old_state = spins[i]
            # Choose a new state uniformly from the other states
            new_state = np.random.choice([s for s in range(q) if s != old_state])
            
            # In the current configuration, the energy change is determined by:
            #   n_old = number of spins in old_state (includes spin i)
            #   n_new = number of spins in new_state (does not include spin i)
            n_old = counts[old_state]
            n_new = counts[new_state]
            # When spin i leaves old_state, it loses interactions with (n_old - 1) spins;
            # when it joins new_state, it gains interactions with n_new spins.
            delta_E = (J / N) * (n_old - n_new - 1)
            
            # Metropolis acceptance criterion:
            if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
                # Accept the move: update spin and counts
                spins[i] = new_state
                counts[old_state] -= 1
                counts[new_state] += 1
        
        # Compute the order parameter:
        if q == 2:
            # For q=2, define the magnetization as the difference in population.
            m = (counts[0] - counts[1]) / float(N)
            order_params.append(m)
        else:
            # For q>=3, define a complex order parameter as:
            # m = (1/N) * sum_{k=0}^{q-1} exp(i*2pi*k/q) * (number of spins in state k)
            m_complex = sum(np.exp(1j * 2 * np.pi * k / q) * counts[k] for k in range(q)) / float(N)
            order_params.append(m_complex)
    
    return np.array(order_params)

# Simulation parameters:
N = 2000        # number of spins
J = 1.0         # coupling constant
n_sweeps = 40   # number of sweeps

#temps = np.linspace(0.3, 0.35, 6)
temps = [0.2, 0.3, 0.30, 0.34, 0.4, 0.45, 0.5, 0.55]
results = {2: {}, 3: {}}

runs = [(q, T) for q in results for T in temps]

for q, T in tqdm(runs, desc="Simulating", total=len(runs)):
    order_params = monte_carlo_mean_field(q, N, J, T, n_sweeps)
    results[q][T] = order_params

# Prepare to plot the order parameter evolution for each q.
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
cmap = plt.get_cmap('coolwarm')
num_temps = len(temps)
colors = [cmap(i) for i in np.linspace(0, 1, num_temps)]

# Plot for q=2 (real order parameter)
ax = axes[0]
for idx, T in enumerate(temps):
    order_params = results[2][T]
    ax.plot(order_params, color=colors[idx], label=f'T = {T:.2f}')
ax.set_title(r'$q=2$')
ax.set_xlabel('Monte Carlo sweep')
ax.set_ylabel('Order parameter')
ax.legend()
ax.grid(True)

# Plot for q=3 (plot the magnitude of the complex order parameter)
ax = axes[1]
for idx, T in enumerate(temps):
    order_params = results[3][T]
    ax.plot(np.abs(order_params), color=colors[idx], label=f'T = {T:.2f}')
ax.set_title(r'$q=3$')
ax.set_xlabel('Monte Carlo sweep')
ax.legend()
ax.grid(True)

plt.suptitle('Evolution of the Order Parameter for Mean-Field Potts Model', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

