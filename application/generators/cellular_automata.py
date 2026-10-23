import numpy as np


def _neighbor_count(grid):
    padded = np.pad(grid, 1, constant_values=0)
    count = np.zeros_like(grid, dtype=int)
    for dy in range(3):
        for dx in range(3):
            if dx != 1 or dy != 1:
                count += padded[dy:dy + grid.shape[0], dx:dx + grid.shape[1]]
    return count


def generate_map(ancho, alto, probabilidad=0.47, iteraciones=5, seed=42,
                 birth_limit=4, death_limit=3):
    """Autómata celular usado por el experimento final de 10.000 mapas."""
    seed = 42 if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    grid = (rng.random((alto, ancho)) > probabilidad).astype(np.uint8)
    grid[[0, -1], :] = 0
    grid[:, [0, -1]] = 0
    for _ in range(iteraciones):
        neighbors = _neighbor_count(grid)
        grid = np.where(
            grid == 1,
            neighbors >= death_limit,
            neighbors > birth_limit,
        ).astype(np.uint8)
        grid[[0, -1], :] = 0
        grid[:, [0, -1]] = 0
    return grid.tolist()
