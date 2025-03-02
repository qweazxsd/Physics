import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def initialize_lattice(L):
    """Initialize the lattice with random spins (-1 or +1)."""
    return np.random.choice([-1, 1], size=(L, L))


def wolff_update_with_visualization(spin_lattice, beta, J, highlight_cluster):
    """Perform a single Wolff cluster update with cluster visualization."""
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
        neighbors = [
            ((cx - 1) % L, cy),
            ((cx + 1) % L, cy),
            (cx, (cy - 1) % L),
            (cx, (cy + 1) % L),
        ]

        for nx, ny in neighbors:
            if not visited[nx, ny] and spin_lattice[nx, ny] == cluster_spin:
                if np.random.rand() < add_prob:
                    cluster.append((nx, ny))
                    visited[nx, ny] = True
        i += 1

    # Highlight the cluster
    highlight_cluster.fill(0)
    for cx, cy in cluster:
        highlight_cluster[cx, cy] = 1

    # Flip the entire cluster
    for cx, cy in cluster:
        spin_lattice[cx, cy] *= -1

    return spin_lattice


def metropolis_update(spin_lattice, beta, J):
    """Perform a single Metropolis update for comparison."""
    L = spin_lattice.shape[0]
    x, y = np.random.randint(L), np.random.randint(L)
    spin = spin_lattice[x, y]

    # Calculate energy change for flipping the spin
    neighbors = [((x - 1) % L, y), ((x + 1) % L, y), (x, (y - 1) % L), (x, (y + 1) % L)]
    dE = 2 * J * spin * sum(spin_lattice[nx, ny] for nx, ny in neighbors)

    # Flip spin with Metropolis probability
    if np.random.rand() < np.exp(-beta * dE):
        spin_lattice[x, y] *= -1

    return spin_lattice


def simulate_with_animation(L, beta, J, steps, use_wolff=True):
    """Simulate the Ising model on a 2D lattice with animation."""
    spin_lattice = initialize_lattice(L)
    highlight_cluster = np.zeros_like(spin_lattice)

    fig, ax = plt.subplots()
    im = ax.imshow(spin_lattice, cmap="coolwarm", interpolation="nearest")
    cluster_overlay = ax.imshow(
        highlight_cluster, cmap="Grays", alpha=0.3, interpolation="nearest"
    )

    def update(frame):
        nonlocal spin_lattice, highlight_cluster
        if use_wolff:
            spin_lattice = wolff_update_with_visualization(
                spin_lattice, beta, J, highlight_cluster
            )
        else:
            spin_lattice = metropolis_update(spin_lattice, beta, J)
        im.set_array(spin_lattice)
        cluster_overlay.set_array(highlight_cluster)
        return [im, cluster_overlay]

    ani = FuncAnimation(fig, update, frames=steps, blit=True, repeat=False)
    plt.title("Spin Lattice Evolution with Cluster Visualization")
    plt.colorbar(cluster_overlay, label="Cluster")
    plt.show()


def plot_lattice(spin_lattice):
    """Plot the spin lattice."""
    plt.imshow(spin_lattice, cmap="coolwarm", interpolation="nearest")
    plt.title("Spin Lattice")
    plt.colorbar(label="Spin")
    plt.show()


if __name__ == "__main__":
    # Parameters
    L = 32  # Lattice size
    beta = 0.4  # Inverse temperature
    J = 1.0  # Interaction strength
    steps = 100  # Number of Monte Carlo steps

    # Simulate with animation
    simulate_with_animation(L, beta, J, steps, use_wolff=True)
