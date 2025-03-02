import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import math
from tqdm import tqdm
import random


def create_lattice(size, fired_set):
    """
    Creates a lattice (2D numpy array) of given size,
    marking activated (or “fired”) sites as 1.
    fired_set is a list/array of coordinate pairs.
    """
    lattice = np.zeros((size, size), dtype=int)
    for element in fired_set:
        lattice[tuple(element)] = 1
    return lattice


def probability_event(prob):
    """
    Returns True with probability 'prob' and False otherwise.
    """
    return random.random() < prob


def perform_bfs(size, prob):
    """
    Starts from all sites in the top row and uses BFS to simulate
    the growth of a connected cluster. At each neighbor of a visited site,
    the site is “fired” (i.e. activated) with probability 'prob'.
    Returns an array of the coordinates of all visited (fired) sites.
    """
    # Initialize queue with every coordinate in the top row.
    queue = [(0, col) for col in range(size)]
    visited = np.zeros((size, size), dtype=bool)
    visited[0, :] = True

    while queue:
        x, y = queue.pop(0)
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        valid_neighbors = [
            (nx, ny)
            for nx, ny in neighbors
            if 0 <= nx < size and 0 <= ny < size and not visited[nx, ny]
        ]
        for nx, ny in valid_neighbors:
            if probability_event(prob):
                queue.append((nx, ny))
                visited[nx, ny] = True

    return np.argwhere(visited)


def worker_block(task):
    """
    Worker function for multiprocessing.
    Given a tuple ((i, j), block) where block is a 2×2 array,
    it checks for a vertical connection (a path from the top to the bottom
    of the block) by simply testing if either column is fully activated.
    Returns (i, j, has_connection) where has_connection is 1 (True) or 0 (False).
    """
    (i, j), block = task
    has_connection = (block[0, 0] and block[1, 0]) or (block[0, 1] and block[1, 1])
    return (i, j, has_connection)


def apply_decimation(grid, new_size):
    """
    Applies a decimation transformation to the lattice (grid) by grouping
    non-overlapping 2×2 blocks. For each block, a worker (run in parallel)
    checks whether a vertical connection exists.

    Returns:
      new_grid: a decimated lattice of shape new_size×new_size, where each cell
                is 1 if the corresponding 2×2 block has a vertical connection, and 0 otherwise.
      percolations: the total number of blocks that are “active” (have a vertical connection).
    """
    tasks = []
    for i in range(new_size):
        for j in range(new_size):
            block = grid[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            tasks.append(((i, j), block.copy()))

    with mp.Pool() as pool:
        results = pool.map(worker_block, tasks)

    new_grid = np.zeros((new_size, new_size), dtype=int)
    percolations = 0
    for i, j, has_connection in results:
        new_grid[i, j] = 1 if has_connection else 0
        percolations += int(has_connection)
    return new_grid, percolations


def main():
    size = 2048  # initial lattice size
    # occupancy probabilities (for the BFS cluster growth)
    ps = [0.4, 0.5, 0.6]
    plot_decimations = False

    # Dictionaries to store the decimated lattice at each step and the measured percolation probability.
    decimated_data = {p: [] for p in ps}
    decimation_probs = {p: [] for p in ps}

    for p in ps:
        print(f"\nProcessing decimation for p = {p}...")
        # Build the initial lattice using a BFS cluster growth from the top row.
        fired_set = perform_bfs(size, p)
        current_lattice = create_lattice(size, fired_set)
        decimated_data[p].append(current_lattice)

        # Determine number of decimation steps.
        steps = int(np.log2(size)) - 1
        for _ in tqdm(range(steps), desc=f"Decimating p={p}", unit="step"):
            if current_lattice.shape[0] <= 2:
                break
            new_size = current_lattice.shape[0] // 2
            new_lattice, percolations = apply_decimation(current_lattice, new_size)
            decimated_data[p].append(new_lattice)
            # Compute the percolation probability as the fraction of blocks with a vertical connection.
            decimation_probability = percolations / (new_size**2)
            decimation_probs[p].append(decimation_probability)
            current_lattice = new_lattice

    if plot_decimations:
        for prob, lattices in decimated_data.items():
            num_steps = len(lattices)
            fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
            if num_steps == 1:
                axes = [axes]
            for step, (ax, lattice) in enumerate(zip(axes, lattices)):
                ax.imshow(lattice, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
                ax.set_title(f"Step {step + 1}")
                ax.axis("off")
            plt.suptitle(f"Decimation Process for p = {prob}")
            plt.tight_layout()
        plt.show()

    # (2) Plot Percolation Probability vs. Decimation Step.
    plt.figure(figsize=(8, 6))
    for prob, decimation_prob in decimation_probs.items():
        plt.plot(
            range(1, len(decimation_prob) + 1),
            decimation_prob,
            marker="o",
            label=f"p = {prob}",
        )
    plt.xlabel("Decimation Step")
    plt.ylabel("Path Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
