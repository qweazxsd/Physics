import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

plt.rcParams["font.size"] = 15


def initialize_lattice(L):
    """Initialize the lattice with random spins (-1 or +1)."""
    return np.random.choice([-1, 1], size=(L, L))


def wolff_update(spin_lattice, beta, J, h):
    """Perform a single Wolff cluster update."""
    L = spin_lattice.shape[0]
    visited = np.zeros_like(spin_lattice, dtype=bool)

    # Pick a random seed spin
    x, y = np.random.randint(L), np.random.randint(L)
    cluster = [(x, y)]
    visited[x, y] = True
    cluster_spin = spin_lattice[x, y]

    # Define probability for adding a neighbor to the cluster
    add_prob = 1 - np.exp(- 2 * beta * J)

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
            if not visited[nx, ny] and spin_lattice[nx, ny] == np.sign(J)*cluster_spin:
                if np.random.rand() < add_prob:
                    cluster.append((nx, ny))
                    visited[nx, ny] = True
        i += 1

    # Flip the entire cluster
    if np.random.rand() < np.exp(2 * beta * h * len(cluster) * cluster_spin):
        for cx, cy in cluster:
            spin_lattice[cx, cy] *= -1

    return spin_lattice


def precompute_frames(L, beta, J, h, steps, method):
    """Precompute all frames for the simulation."""
    spin_lattice = initialize_lattice(L)
    highlight_cluster = np.zeros_like(spin_lattice, dtype=float)
    magnetizations = []
    frames = []

    for _ in range(steps):
        spin_lattice = wolff_update(spin_lattice, beta, J, h)
        frames.append((spin_lattice.copy(), highlight_cluster.copy()))
        magnetizations.append(np.abs(np.sum(spin_lattice)) / (L * L))
    return frames, magnetizations


def simulate(L, beta, J, steps, h=0, animate=True):
    """Simulate the Ising model on a 2D lattice."""
    frames, magnetizations = precompute_frames(L, beta, J, h, steps)

    if animate:
        fig, ax = plt.subplots()
        im = ax.imshow(frames[0][0], cmap="coolwarm", interpolation="nearest")
        cluster_overlay = ax.imshow(
            frames[0][1],
            cmap="Grays",
            alpha=0.2,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        def update(frame):
            nonlocal cluster_overlay, im
            spin_lattice, highlight_cluster = frames[frame]
            im.set_array(spin_lattice)
            cluster_overlay.set_array(highlight_cluster)
            return [im, cluster_overlay]

        ani = FuncAnimation(fig, update, frames=len(
            frames), blit=True, repeat=False)
        plt.title("Spin Lattice Evolution")
        plt.colorbar(cluster_overlay, label="Spin")
        plt.show()
    else:
        return frames[-1][0], magnetizations


def plot_magnetization_vs_beta(L, J, h, steps, betas, method="wolff", plt_tc=False):
    """Plot the magnetization as a function of beta."""
    magnetizations = []

    for beta in tqdm(betas, desc=f"Calculating Magnetization ({method})"):
        _, mags = precompute_frames(L, beta, J, h, steps, method)
        magnetizations.append(np.mean(mags))

    plt.plot(betas, magnetizations, marker="o", ls="None", label=f"h={h:.2f}")
    plt.plot(betas, np.gradient(magnetizations), label="grad")
    if plt_tc:
        imax = np.gradient(magnetizations).argmax()
        betac = betas[imax]
        plt.vlines(
            x=betac,
            ymin=0,
            ymax=max(magnetizations),
            colors="r",
            ls="dashed",
            label=rf"$\beta_c={betac:.2f}$",
        )
        plt.vlines(
            x=np.log(1 + np.sqrt(2)) / (2 * np.abs(J)),
            ymin=0,
            ymax=max(magnetizations),
            colors="k",
            ls="dashed",
            label=r"$\beta_c^{real}=\frac{1+ln(2)}{2J}=$"
            + rf"${np.log(1+np.sqrt(2))/(2*np.abs(J)):.2f}$",
        )
    plt.title("Magnetization vs. Beta")
    plt.xlabel("Beta (1/kT)")
    plt.ylabel("Magnetization")
    plt.legend()


if __name__ == "__main__":
    # Parameters
    L = 64  # Lattice size
    J = -1  # Interaction strength
    h = 0.5
    steps = 10000  # Number of Monte Carlo steps

    # Simulate with animation
    beta = 0.3
    # simulate(L, beta, J, steps, method="wolff", animate=True)

    # Plot magnetization vs beta for Wolff and Metropolis
    betas = np.linspace(0.01, 10, 200)
    hs = np.linspace(0.1, 1, 6)
    #for h in hs:
    plot_magnetization_vs_beta(L, J, h, steps, betas, plt_tc=0)
    plt.show()
    # betas = np.linspace(0.1, 10, 50)
    # plot_magnetization_vs_beta(L, -J, 1000, betas, method="metropolis", h=0.5)
