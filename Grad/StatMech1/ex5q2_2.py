import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import math
from tqdm import tqdm

def worker_block(task):
    """
    Worker function for multiprocessing.
    Input: task is a tuple ((i, j), block) where block is a 2×2 numpy array.
    It checks for a vertical connection (i.e. a path from the top row to the bottom row)
    by testing if either the left or right column is fully occupied.
    Returns: (i, j, has_connection) where has_connection is True (1) or False (0).
    """
    (i, j), block = task
    # For a 2x2 block using nearest-neighbor moves, a vertical connection exists
    # if either column is fully occupied.
    has_connection = np.any(block[0, :]) and np.any(block[1, :])
    return (i, j, has_connection)

def apply_decimation(grid, new_size):
    """
    Applies a decimation transformation to the lattice (grid) by grouping non-overlapping
    2×2 blocks. For each block, it uses multiprocessing to check (via worker_block) whether
    there is a vertical connection.
    
    Returns:
      new_grid: the decimated lattice (of shape new_size × new_size) where each cell is 1 if
                the corresponding block has a vertical connection and 0 otherwise.
      percolations: the total count of blocks that have a vertical connection.
    """
    tasks = []
    for i in range(new_size):
        for j in range(new_size):
            block = grid[2*i:2*i+2, 2*j:2*j+2]
            tasks.append(((i, j), block.copy()))
    
    with mp.Pool() as pool:
        results = pool.map(worker_block, tasks)
    
    new_grid = np.zeros((new_size, new_size), dtype=int)
    percolations = 0
    for (i, j, has_connection) in results:
        if has_connection:
            new_grid[i, j] = 1
            percolations += 1
        else:
            new_grid[i, j] = 0
    return new_grid, percolations

def main():
    size = 2048  # initial lattice size (assumed square)
    ps = [0.4, 0.5, 0.6]  # occupancy probabilities to process
    
    # Dictionaries to hold decimated lattice data and percolation probabilities at each decimation step.
    decimated_data = {p: [] for p in ps}
    decimation_probs = {p: [] for p in ps}
    
    for p in ps:
        print(f"\nProcessing decimation for p = {p}...")
        # Create a single realization: a binary lattice where 1 indicates an occupied site.
        # (Here we use the raw lattice as our starting point.)
        current_lattice = (np.random.rand(size, size) < p).astype(int)
        decimated_data[p].append(current_lattice)
        
        # Determine the number of decimation steps.
        steps = int(np.log2(size)) - 1  # similar to the inspiration code
        for _ in tqdm(range(steps), desc=f"Decimating p={p}", unit="step"):
            if current_lattice.shape[0] <= 2:
                break
            new_size = current_lattice.shape[0] // 2
            new_lattice, percolations = apply_decimation(current_lattice, new_size)
            # Compute the percolation probability for this decimation step.
            decimation_probability = percolations / (new_size ** 2)
            decimated_data[p].append(new_lattice)
            decimation_probs[p].append(decimation_probability)
            current_lattice = new_lattice
    
    # (1) Plot the decimated lattices for each p.
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
        plt.plot(range(1, len(decimation_prob) + 1), decimation_prob, marker="o", label=f"p = {prob}")
    plt.xlabel("Decimation Step")
    plt.ylabel("% of percolated blocks")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

