import random

def generate_map(ancho, alto, probabilidad, iteraciones, seed=None):
    rng = random.Random(seed)

    grid = [
        [1 if rng.random() < probabilidad else 0 for _ in range(ancho)]
        for _ in range(alto)
    ]

    for _ in range(iteraciones):
        new_grid = []
        for y in range(alto):
            row = []
            for x in range(ancho):
                vecinos = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if dx == 0 and dy == 0:
                            continue
                        if 0 <= ny < alto and 0 <= nx < ancho:
                            vecinos += grid[ny][nx]
                        else:
                            vecinos += 1
                row.append(1 if vecinos >= 5 else 0)
            new_grid.append(row)
        grid = new_grid

    return grid
