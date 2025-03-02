import numpy as np
import matplotlib.pyplot as plt

def dfs(grid):
    """
    Check if there is a spanning path from the top to the bottom of the grid
    using an iterative depth-first search (DFS) implemented with a stack.
    """
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    stack = []

    # Initialize the stack with all occupied sites in the top row.
    for j in range(cols):
        if grid[0, j]:
            stack.append((0, j))
            visited[0, j] = True

    # Process nodes using DFS.
    while stack:
        r, c = stack.pop()
        # If we have reached the bottom row, a spanning path exists.
        if r == rows - 1:
            return True
        # Explore the four neighbors: up, down, left, right.
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
    return False


if __name__ == '__main__':

    lattice_size = 2048
    num_realizations = 100
    p_values = np.arange(0, 1, 0.05)
    #p_values = [0.5925]
    spanning_probabilities = []

    for p in p_values:
        print(f"Processing p = {p:.2f}")
        count_spanning = 0
        for _ in range(num_realizations):
            # Generate a random lattice where True indicates an occupied site.
            grid = np.random.rand(lattice_size, lattice_size) < p
            if dfs(grid):
                count_spanning += 1
        spanning_probabilities.append(count_spanning / num_realizations)

    plt.figure(figsize=(8, 6))
    plt.plot(p_values, spanning_probabilities, marker='o')
    plt.xlabel("Occupation Probability p")
    plt.ylabel("Percolation Probability")
    plt.grid(True)
    plt.show()
