from collections import deque
import math
import numpy as np

ML_FEATURE_NAMES = [
    "width", "height", "floor_ratio", "n_regions", "largest_region_ratio",
    "edge_density", "row_entropy", "col_entropy", "local_variance",
    "perimeter_area_ratio",
]

FEATURE_NAMES = ML_FEATURE_NAMES.copy()

def _binary_entropy(values):
    arr = np.clip(np.asarray(values, dtype=float), 1e-9, 1 - 1e-9)
    return -(arr * np.log2(arr) + (1 - arr) * np.log2(1 - arr))


def _regions(grid):
    h, w = grid.shape
    visited = set()
    sizes = []
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 1 or (x, y) in visited:
                continue
            q = deque([(x, y)])
            visited.add((x, y))
            size = 0
            while q:
                cx, cy = q.popleft()
                size += 1
                for nx, ny in ((cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)):
                    if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 1 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
            sizes.append(size)
    return sizes


def extract_ml_feature_dict(grid):
    arr = np.asarray(grid, dtype=np.uint8)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("El mapa debe ser una matriz 2D no vacía")
    height, width = arr.shape
    floor_ratio = float(arr.mean())
    sizes = _regions(arr)
    n_regions = len(sizes)
    largest = max(sizes, default=0)
    largest_region_ratio = largest / max(int(arr.sum()), 1)
    horizontal = np.mean(arr[:, 1:] != arr[:, :-1]) if width > 1 else 0.0
    vertical = np.mean(arr[1:, :] != arr[:-1, :]) if height > 1 else 0.0
    edge_density = float((horizontal + vertical) / 2)
    row_entropy = float(_binary_entropy(arr.mean(axis=1)).mean())
    col_entropy = float(_binary_entropy(arr.mean(axis=0)).mean())
    padded = np.pad(arr.astype(float), 1, mode="edge")
    local_vars = [
        np.var(padded[y:y+3, x:x+3])
        for y in range(height) for x in range(width)
    ]
    internal_edges = int(np.sum(arr[:, 1:] != arr[:, :-1]) + np.sum(arr[1:, :] != arr[:-1, :]))
    border_open = int(arr[0, :].sum() + arr[-1, :].sum() + arr[:, 0].sum() + arr[:, -1].sum())
    perimeter_area_ratio = float((internal_edges + border_open) / max(int(arr.sum()), 1))
    return {
        "width": int(width), "height": int(height), "floor_ratio": floor_ratio,
        "n_regions": int(n_regions), "largest_region_ratio": float(largest_region_ratio),
        "edge_density": edge_density, "row_entropy": row_entropy,
        "col_entropy": col_entropy, "local_variance": float(np.mean(local_vars)),
        "perimeter_area_ratio": perimeter_area_ratio,
    }


def extract_feature_dict(grid):
    """Compatibilidad: métricas antiguas + métricas estructurales del experimento final."""
    ml = extract_ml_feature_dict(grid)
    accessibility = ml["floor_ratio"]
    obstacle_density = 1.0 - accessibility
    return {
        **ml,
        "connectivity": ml["largest_region_ratio"],
        "accessibility": accessibility,
        "obstacle_density": obstacle_density,
        "connected_components": ml["n_regions"],
        "largest_open_area": ml["largest_region_ratio"] * accessibility,
        "border_obstacle_ratio": 0.0,
        "balance": 1.0 - abs(accessibility - 0.65),
    }


def extract_features(grid):
    f = extract_ml_feature_dict(grid)
    return [f[name] for name in ML_FEATURE_NAMES]
