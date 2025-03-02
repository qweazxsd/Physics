import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def initialize_lattice(L):
    """Initialize the lattice with random spins (-1 or +1)."""
    return np.random.choice([-1, 1], size=(L, L))

def wolff_update(spin_lattice, beta, J):
    """Perform a single Wolff cluster update."""
    L = spin_lattice.shape[0]
    visited = np.zeros_like(spin_lattice, dtype=bool)

    # Pick a random seed spin
    x, y = np.random.randint(L), np.random.randint(L)
    cluster = [(x, y)]
    visited[x, y] = True
    cluster_spin = spin_lattice[x, y]

    # Define probability for adding a neighbor to the cluster
    add_prob = 1 - np.exp(-2 * beta * J)

    # Grow the cluster
    i = 0
    while i < len(cluster):
        cx, cy = cluster[i]
        neighbors = [((cx - 1) % L, cy),
                     ((cx + 1) % L, cy),
                     (cx, (cy - 1) % L),
                     (cx, (cy + 1) % L)]

        for nx, ny in neighbors:
            if not visited[nx, ny] and spin_lattice[nx, ny] == cluster_spin:
                if np.random.rand() < add_prob:
                    cluster.append((nx, ny))
                    visited[nx, ny] = True
        i += 1

    # Flip the entire cluster
    for cx, cy in cluster:
        spin_lattice[cx, cy] *= -1

    return spin_lattice

def precompute_frames(L, beta, J, steps):
    """Precompute all frames for the simulation."""
    spin_lattice = initialize_lattice(L)
    magnetizations = []
    frames = []

    for _ in range(steps):
        spin_lattice = wolff_update(spin_lattice, beta, J)
        frames.append(spin_lattice.copy())
        magnetizations.append(np.abs(np.sum(spin_lattice)) / (L * L))

    return frames, magnetizations

def simulate(L, beta, J, steps, animate=True):
    """Simulate the Ising model on a 2D lattice."""
    if animate:
        frames, _ = precompute_frames(L, beta, J, steps)

        fig, ax = plt.subplots()
        im = ax.imshow(frames[0], cmap='coolwarm', interpolation='nearest')

        def update(frame):
            im.set_array(frames[frame])
            return [im]

        ani = FuncAnimation(fig, update, frames=len(frames), blit=True, repeat=False)
        plt.title("Spin Lattice Evolution")
        plt.colorbar(im, label="Spin")
        plt.show()
    else:
        frames, magnetizations = precompute_frames(L, beta, J, steps)
        return frames[-1], magnetizations

def plot_magnetization_vs_beta(L, J, steps, betas):
    """Plot the magnetization as a function of beta."""
    magnetizations = []

    for beta in betas:
        _, mags = precompute_frames(L, beta, J, steps)
        magnetizations.append(np.mean(mags))

    plt.plot(betas, magnetizations, marker='o')
    plt.title("Magnetization vs. Beta")
    plt.xlabel("Beta (1/kT)")
    plt.ylabel("Magnetization")
    plt.show()

if __name__ == "__main__":
    # Parameters
    L = 32         # Lattice size
    J = 1.0        # Interaction strength
    steps = 250    # Number of Monte Carlo steps

    # Simulate with animation
    beta = 0.4
    #simulate(L, beta, J, steps, animate=True)

    # Plot magnetization vs beta
    betas = np.linspace(0.1, 0.9, 30)
    plot_magnetization_vs_beta(L, J, steps, betas)

